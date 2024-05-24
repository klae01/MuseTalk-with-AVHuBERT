import glob
import itertools
import os
import tempfile

import cv2
import imageio
import numpy as np
import torch
from diffusers import AutoencoderKL
from moviepy.editor import AudioFileClip, VideoFileClip

from adaptor.av2av_unit import AVSpeechToUnitExtractor
from adaptor.model import WhisperCNN
from musetalk.models.unet import UNet
from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox_iter
from musetalk.utils.utils import datagen_iter, get_file_type, get_video_fps
from musetalk.whisper.audio2feature import Audio2Feature


class RepeatingIterator:
    def __init__(self, iterable, reverse=False):
        self.iterable = iterable
        self.reverse = reverse
        self.iter = iter(iterable)
        self.cache = []

    def __iter__(self):
        top = 0
        while True:
            if top < len(self.cache):
                yield from self.cache[top:]
                top = len(self.cache)
            try:
                item = next(self.iter)
            except StopIteration:
                break
            self.cache.append(item)
        yield from itertools.cycle(
            self.cache[::-1] + self.cache if self.reverse else self.cache
        )


class AVHubertInference:
    MAVHUBERT_LARGE_NOISE = "mavhubert_large_noise.pt"
    WHISPER_CNN = "whisper_cnn_0019.pth"
    WHISPER_TINY = "whisper/tiny.pt"
    SD_VAE_FT_MSE = "sd-vae-ft-mse/"
    MUSETALK_JSON = "musetalk/musetalk.json"
    MUSETALK_BIN = "musetalk/pytorch_model.bin"

    def __init__(self, checkpoints_dir):
        self.checkpoints_dir = checkpoints_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tensor_options = {
            "dtype": torch.float,
            "device": self.device,
            "non_blocking": True,
        }
        self.model_path_avhubert = self.get_checkpoint_path(self.MAVHUBERT_LARGE_NOISE)
        self.model_path_whisper_cnn = self.get_checkpoint_path(self.WHISPER_CNN)

    def get_checkpoint_path(self, subpath):
        return os.path.join(self.checkpoints_dir, subpath)

    def remove_module_prefix(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def initialize_backbone_model(self):
        self.audio_processor = Audio2Feature(
            self.get_checkpoint_path(self.WHISPER_TINY)
        )
        del self.audio_processor.model
        self.vae = AutoencoderKL.from_pretrained(
            self.get_checkpoint_path(self.SD_VAE_FT_MSE)
        ).to(**self.tensor_options)
        unet_modules = UNet(
            self.get_checkpoint_path(self.MUSETALK_JSON),
            self.get_checkpoint_path(self.MUSETALK_BIN),
        )
        self.unet = unet_modules.model.to(**self.tensor_options)
        self.pe = unet_modules.pe.to(**self.tensor_options)
        self.timesteps = torch.tensor([0], device=self.device)

    def initialize_bridge_model(self):
        self.avhubert_processor = AVSpeechToUnitExtractor(
            self.get_checkpoint_path(self.MAVHUBERT_LARGE_NOISE),
            torch.cuda.is_available(),
        )
        self.avhubert_to_whisper = WhisperCNN(1001, 128, 512, 384, 5, 5, 16)
        state_dict = torch.load(self.get_checkpoint_path(self.WHISPER_CNN))
        state_dict = self.remove_module_prefix(state_dict)
        self.avhubert_to_whisper.load_state_dict(state_dict, strict=True)
        self.avhubert_to_whisper.to(self.device)

    def extract_frames(self, video_path):
        def frame_generator(video_capture):
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if not ret:
                    break
                yield frame
            video_capture.release()

        def image_generator(image_list):
            for img_path in image_list:
                img = cv2.imread(img_path)
                yield img

        if get_file_type(video_path) == "video":
            video_capture = cv2.VideoCapture(video_path)
            fps = get_video_fps(video_path)
            return frame_generator(video_capture), fps
        else:
            input_img_list = sorted(
                glob.glob(os.path.join(video_path, "*.[jpJP][pnPN]*[gG]"))
            )
            fps = 25

            return image_generator(input_img_list), fps

    @torch.no_grad()
    def extract_audio_features(self, audio_path, fps):
        avhubert_units = self.avhubert_processor.process_audio(audio_path)
        length = avhubert_units.size(0)
        output_length = (length + [0, 1, 2, 0, 1, 2, 0, 2][length % 16 // 2]) * 5
        avhubert_units = avhubert_units + 1
        avhubert_units = torch.nn.functional.pad(avhubert_units, (0, 5))
        whisper_feature = self.avhubert_to_whisper(
            avhubert_units.to(self.device)[None]
        )[0]
        whisper_feature = whisper_feature[:output_length, :].reshape(-1, 5, 384)
        whisper_feature = whisper_feature.cpu().numpy()
        whisper_chunks = self.audio_processor.feature2chunks(
            feature_array=whisper_feature, fps=fps
        )
        return whisper_chunks

    @torch.no_grad()
    def extract_bbox_and_frame(self, frame_iter, bbox_shift, batch_size=1):
        frame_iter, frame_iter1 = itertools.tee(frame_iter)
        for bbox, frame in zip(
            get_landmark_and_bbox_iter(frame_iter, bbox_shift, batch_size), frame_iter1
        ):
            if bbox == coord_placeholder:
                continue
            bbox = tuple(
                map(
                    lambda v, mi, mx: max(mi, min(v, mx)),
                    bbox,
                    [0, 0, 0, 0],
                    list(frame.shape[:2]) * 2,
                )
            )
            yield bbox, frame

    @torch.no_grad()
    def extract_latents(self, coord_frame_iter, batch_size=8):
        batches = []

        for bbox, frame in coord_frame_iter:
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(
                crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4
            )
            batches.append(crop_frame)
            if len(batches) == batch_size:
                yield from self.vae_encode_batch(batches).unsqueeze(1)
                batches = []

        # Process any remaining images
        if batches:
            yield from self.vae_encode_batch(batches).unsqueeze(1)

    @torch.no_grad()
    def vae_encode_batch(self, batch_frames):
        # Convert batch_frames to a tensor
        batch_frames = np.stack(batch_frames)
        batch_frames = torch.from_numpy(batch_frames).to(**self.tensor_options)
        batch_frames = batch_frames.flip(-1)
        batch_frames = batch_frames.permute(0, 3, 1, 2) / 255.0
        batch_frames = torch.repeat_interleave(batch_frames, 2, 0)
        batch_frames[::2, :, 128:].zero_()
        batch_frames = (batch_frames - 0.5) / 0.5

        # Encode latents in a batch
        init_latent_dist = self.vae.encode(batch_frames).latent_dist
        init_latents = self.vae.config.scaling_factor * init_latent_dist.sample()
        return init_latents.unflatten(0, [-1, 2]).flatten(1, 2)

    @torch.no_grad()
    def run_inference(self, whisper_chunks, latent_cycle, batch_size=1):
        for whisper_batch, latent_batch in datagen_iter(
            whisper_chunks, latent_cycle, batch_size=batch_size
        ):
            audio_feature_batch = self.pe(
                torch.from_numpy(whisper_batch).to(**self.tensor_options)
            )
            pred_latents = self.unet(
                latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            yield from self.vae_decode_batch(pred_latents)

    @torch.no_grad()
    def vae_decode_batch(self, latents):
        latents = latents / self.vae.config.scaling_factor
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.permute(0, 2, 3, 1).float()
        image = (image * 255).round().flip(-1).to(dtype=torch.uint8, device="cpu")
        return image.numpy()

    def postprocess_images(self, res_frame_iter, frame_cycle, coord_cycle):
        for bbox, ori_frame, res_frame in zip(coord_cycle, frame_cycle, res_frame_iter):
            x1, y1, x2, y2 = bbox
            res_frame = cv2.resize(
                res_frame.astype(np.uint8),
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_AREA,
            )
            combine_frame = ori_frame.copy()
            combine_frame[y1:y2, x1:x2] = res_frame
            yield combine_frame

    def create_video(self, result_images, fps, audio_path, output_vid_name):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmpfile:
            output_video = tmpfile.name
            imageio.mimwrite(
                output_video,
                [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in result_images],
                "FFMPEG",
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
            )

            video_clip = VideoFileClip(output_video)
            audio_clip = AudioFileClip(audio_path)
            video_clip = video_clip.set_audio(audio_clip)
            video_clip.write_videofile(
                output_vid_name, codec="libx264", audio_codec="aac", fps=fps
            )
            video_clip.close()
        print(f"Result saved to {output_vid_name}")

    def run_pipeline(self, audio_path, video_path, bbox_shift=0, batch_size=16):
        result_dir = "./results/output"
        input_basename = os.path.basename(video_path).split(".")[0]
        audio_basename = os.path.basename(audio_path).split(".")[0]
        output_basename = f"{input_basename}_{audio_basename}"
        os.makedirs(result_dir, exist_ok=True)
        output_vid_name = os.path.join(result_dir, output_basename + ".mp4")

        frame_iter, fps = self.extract_frames(video_path)
        whisper_chunks = self.extract_audio_features(audio_path, fps)

        gen1, gen2, gen3 = itertools.tee(
            self.extract_bbox_and_frame(frame_iter, bbox_shift, batch_size), 3
        )
        coord_cycle = RepeatingIterator((item[0] for item in gen1), reverse=True)
        frame_cycle = RepeatingIterator((item[1] for item in gen2), reverse=True)
        latent_cycle = RepeatingIterator(
            self.extract_latents(gen3, batch_size), reverse=True
        )

        res_frame_iter = self.run_inference(
            whisper_chunks, latent_cycle, batch_size=batch_size
        )
        result_image_iter = self.postprocess_images(
            res_frame_iter, frame_cycle, coord_cycle
        )
        self.create_video(list(result_image_iter), fps, audio_path, output_vid_name)


def batch(iterable, n=1):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            break
        yield chunk


# Usage example:
# avhubert_inference = AVHubertInference("path/to/checkpoints")
# avhubert_inference.initialize_backbone_model()
# avhubert_inference.initialize_bridge_model()
# avhubert_inference.run_pipeline("path_to_audio.mp3", "path_to_video.mp4", 0)
