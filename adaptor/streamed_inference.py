import glob
import os
import shutil

import cv2
import imageio
import numpy as np
import torch
from av2av_unit import AVSpeechToUnitExtractor
from moviepy.editor import AudioFileClip, VideoFileClip

from adaptor.model import WhisperCNN
from musetalk.utils.blending import get_image
from musetalk.utils.preprocessing import (
    coord_placeholder,
    get_bbox_range,
    get_landmark_and_bbox,
    read_imgs,
)
from musetalk.utils.utils import datagen, get_file_type, get_video_fps, load_all_model

CheckpointsDir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "models")


class AVHubertInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path_avhubert = os.path.join(
            CheckpointsDir, "mavhubert_large_noise.pt"
        )
        self.model_path_whisper_cnn = os.path.join(
            CheckpointsDir, "whisper_cnn_0019.pth"
        )
        self.audio_processor, self.vae, self.unet, self.pe = load_all_model()
        self.timesteps = torch.tensor([0], device=self.device)

        self.initialize_model()

    def remove_module_prefix(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def initialize_model(self):
        self.avhubert_processor = AVSpeechToUnitExtractor(
            self.model_path_avhubert, torch.cuda.is_available()
        )
        self.avhubert_to_whisper = WhisperCNN(1001, 128, 512, 384, 5, 5, 16)
        state_dict = self.remove_module_prefix(torch.load(self.model_path_whisper_cnn))
        self.avhubert_to_whisper.load_state_dict(state_dict, strict=True)
        self.avhubert_to_whisper.to(self.device)

    def extract_frames(self, video_path, result_dir, input_basename):
        # Use a temporary directory to avoid conflicts
        temp_dir = os.path.join(result_dir, f"{input_basename}_frames")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        result_img_save_path = os.path.join(result_dir, input_basename)
        os.makedirs(result_img_save_path, exist_ok=True)

        if get_file_type(video_path) == "video":
            reader = imageio.get_reader(video_path)
            for i, im in enumerate(reader):
                imageio.imwrite(f"{temp_dir}/{i:08d}.png", im)
            input_img_list = sorted(
                glob.glob(os.path.join(temp_dir, "*.[jpJP][pnPN]*[gG]"))
            )
            fps = get_video_fps(video_path)
        else:
            input_img_list = sorted(
                glob.glob(os.path.join(video_path, "*.[jpJP][pnPN]*[gG]"))
            )
            fps = 25
        return input_img_list, fps, result_img_save_path

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
    def preprocess_images(self, input_img_list, bbox_shift):
        coord_list = []
        frame_list = []
        input_latent_list = []
        for bbox, frame in zip(*get_landmark_and_bbox(input_img_list, bbox_shift)):
            if bbox == coord_placeholder:
                continue
            frame_list.append(frame)
            bbox = tuple(
                map(
                    lambda v, mi, mx: max(mi, min(v, mx)),
                    bbox,
                    [0, 0, 0, 0],
                    list(frame.shape[:2]) * 2,
                )
            )
            coord_list.append(bbox)
            x1, y1, x2, y2 = bbox
            crop_frame = frame[y1:y2, x1:x2]
            crop_frame = cv2.resize(
                crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4
            )
            latents = self.vae.get_latents_for_unet(crop_frame)
            input_latent_list.append(latents)
        return coord_list, frame_list, input_latent_list

    @torch.no_grad()
    def run_inference(self, whisper_chunks, input_latent_list, frame_list, coord_list):
        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]
        res_frame_list = []
        for whisper_batch, latent_batch in datagen(
            whisper_chunks, input_latent_list_cycle, batch_size=8
        ):
            audio_feature_batch = self.pe(
                torch.from_numpy(whisper_batch).to(self.unet.device)
            )
            pred_latents = self.unet.model(
                latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            recon = self.vae.decode_latents(pred_latents)
            res_frame_list.extend(recon)
        return res_frame_list, frame_list_cycle, coord_list_cycle

    def postprocess_images(
        self, res_frame_list, frame_list_cycle, coord_list_cycle, result_img_save_path
    ):
        for i, res_frame in enumerate(res_frame_list):
            bbox = coord_list_cycle[i % len(coord_list_cycle)]
            ori_frame = frame_list_cycle[i % len(frame_list_cycle)]
            x1, y1, x2, y2 = bbox
            res_frame = cv2.resize(
                res_frame.astype(np.uint8),
                (x2 - x1, y2 - y1),
                interpolation=cv2.INTER_AREA,
            )
            combine_frame = get_image(ori_frame, res_frame, bbox)
            cv2.imwrite(f"{result_img_save_path}/{str(i).zfill(8)}.png", combine_frame)

    def create_video(self, result_img_save_path, fps, audio_path, output_vid_name):
        output_video = "temp.mp4"
        images = [
            imageio.imread(f"{result_img_save_path}/{file}")
            for file in sorted(os.listdir(result_img_save_path))
            if file.endswith(".png")
        ]
        imageio.mimwrite(
            output_video,
            images,
            "FFMPEG",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
        )

        video_clip = VideoFileClip(output_video)
        audio_clip = AudioFileClip(audio_path)
        video_clip = video_clip.set_audio(audio_clip)
        video_clip.write_videofile(
            output_vid_name, codec="libx264", audio_codec="aac", fps=25
        )

        os.remove("temp.mp4")
        print(f"Result saved to {output_vid_name}")

    def run_pipeline(self, audio_path, video_path, bbox_shift=0):
        result_dir = "./results/output"
        input_basename = os.path.basename(video_path).split(".")[0]
        audio_basename = os.path.basename(audio_path).split(".")[0]
        output_basename = f"{input_basename}_{audio_basename}"
        output_vid_name = os.path.join(result_dir, output_basename + ".mp4")

        input_img_list, fps, result_img_save_path = self.extract_frames(
            video_path, result_dir, input_basename
        )
        whisper_chunks = self.extract_audio_features(audio_path, fps)
        coord_list, frame_list, input_latent_list = self.preprocess_images(
            input_img_list, bbox_shift
        )
        res_frame_list, frame_list_cycle, coord_list_cycle = self.run_inference(
            whisper_chunks, input_latent_list, frame_list, coord_list
        )
        self.postprocess_images(
            res_frame_list, frame_list_cycle, coord_list_cycle, result_img_save_path
        )
        self.create_video(result_img_save_path, fps, audio_path, output_vid_name)


# Usage example:
# avhubert_inference = AVHubertInference()
# avhubert_inference.run_pipeline("path_to_audio.mp3", "path_to_video.mp4", 0)
