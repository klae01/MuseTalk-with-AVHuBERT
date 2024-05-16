import glob
import os
import pickle
import re
import shutil
import time
from argparse import Namespace

import cv2

# import imageio
import ffmpeg
import gdown
import numpy as np
import requests
import torch
import tqdm

# from moviepy.editor import *
from huggingface_hub import snapshot_download

ProjectDir = os.path.abspath(os.path.dirname(__file__))
CheckpointsDir = os.path.join(ProjectDir, "models")


def download_model():
    if not os.path.exists(CheckpointsDir):
        os.makedirs(CheckpointsDir)
        print("Checkpoint Not Downloaded, start downloading...")
        tic = time.time()
        snapshot_download(
            repo_id="TMElyralab/MuseTalk",
            local_dir=CheckpointsDir,
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True,
            resume_download=False,
        )
        # weight
        os.makedirs(f"{CheckpointsDir}/sd-vae-ft-mse/")
        snapshot_download(
            repo_id="stabilityai/sd-vae-ft-mse",
            local_dir=CheckpointsDir + "/sd-vae-ft-mse",
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True,
            resume_download=False,
        )
        # dwpose
        os.makedirs(f"{CheckpointsDir}/dwpose/")
        snapshot_download(
            repo_id="yzd-v/DWPose",
            local_dir=CheckpointsDir + "/dwpose",
            max_workers=8,
            local_dir_use_symlinks=True,
            force_download=True,
            resume_download=False,
        )
        # vae
        url = "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt"
        response = requests.get(url)
        if response.status_code == 200:
            file_path = f"{CheckpointsDir}/whisper/tiny.pt"
            os.makedirs(f"{CheckpointsDir}/whisper/")
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"Request failed with status code: {response.status_code}")
        # gdown face parse
        url = "https://drive.google.com/uc?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812"
        os.makedirs(f"{CheckpointsDir}/face-parse-bisent/")
        file_path = f"{CheckpointsDir}/face-parse-bisent/79999_iter.pth"
        gdown.download(url, file_path, quiet=False)
        # resnet
        url = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
        response = requests.get(url)
        if response.status_code == 200:
            file_path = f"{CheckpointsDir}/face-parse-bisent/resnet18-5c106cde.pth"
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"Request failed with status code: {response.status_code}")

        toc = time.time()
        print(f"Download took {toc-tic} seconds")

    else:
        print("Model already downloaded.")


download_model()  # for huggingface deployment.

from musetalk.utils.utils import get_video_fps, load_all_model

# load model weights
audio_processor, vae, unet, pe = load_all_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = torch.tensor([0], device=device)


def process_audio(audio_path, output_path):
    """
    Process the audio file and save the whisper_chunks as an npy file.
    """
    whisper_feature = audio_processor.audio2feat(audio_path)

    # Save whisper_chunks as an npy file
    np.save(output_path, whisper_feature)
    print(f"Whisper chunks saved to {output_path}")

    return output_path


# # Example usage
# audio_path = '/mnt/hard3/rhs/intern/sample_folder/output_audio_t1.wav'
# process_audio(audio_path)

import glob

audio_path = "/mnt/hard3/rhs/intern/audio"
featu_path = "/mnt/hard3/rhs/intern/audio_feature"
finish = []
for input_path in glob.glob(os.path.join(audio_path, "*", "*", "*.wav")):
    output_path = os.path.join(featu_path, os.path.relpath(input_path, audio_path))
    output_path = os.path.splitext(output_path)[0] + ".npy"
    output_dir = os.path.dirname(output_path)
    finish.append(os.path.exists(output_path))

print(np.argmin(finish), np.sum(finish))

restart = max(0, np.argmin(finish) - 128)
for input_path in tqdm.tqdm(
    glob.glob(os.path.join(audio_path, "*", "*", "*.wav"))[restart:], mininterval=10
):
    output_path = os.path.join(featu_path, os.path.relpath(input_path, audio_path))
    output_path = os.path.splitext(output_path)[0] + ".npy"
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    process_audio(input_path, output_path)
