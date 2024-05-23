#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh

# Initialize conda environment for MuseTalk
conda create -n musetalk python=3.10 -y
conda activate musetalk


# Install required packages from requirements.txt
python3 -m pip install -U pip
python3 -m pip install --upgrade pandas
python3 -m pip install -r requirements.txt
# python3 -m pip install flask Flask-SocketIO scikit-image librosa
# python3 -m pip install git+https://github.com/hhj1897/face_alignment.git git+https://github.com/hhj1897/face_detection.git
# python3 -m pip install cffi cython 'hydra-core>=1.0.7,<1.1' 'omegaconf<2.1' 'numpy>=1.21.3' regex 'sacrebleu>=1.4.12' 'torch>=1.13' tqdm bitarray 'torchaudio>=0.8.0' scikit-learn packaging

# Install mmlab packages
python3 -m pip install --no-cache-dir -U openmim
python3 -m mim install mmengine mmcv>=2.0.1 mmdet>=3.1.0 mmpose>=1.1.0

# # Download ffmpeg-static
# # (Assuming that ffmpeg-static is already downloaded or you will download it manually)
# export FFMPEG_PATH=/path/to/ffmpeg
# # Example:
# # export FFMPEG_PATH=/musetalk/ffmpeg-4.4-amd64-static

# Download weights and organize them in the models directory
mkdir -p models/musetalk
mkdir -p models/dwpose
mkdir -p models/face-parse-bisent
mkdir -p models/sd-vae-ft-mse
mkdir -p models/whisper

# Download the weights
# MuseTalk weights
wget -P models/musetalk https://huggingface.co/TMElyralab/MuseTalk/resolve/main/pytorch_model.bin
wget -P models/musetalk https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk.json

# sd-vae-ft-mse weights
wget -P models/sd-vae-ft-mse https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin
wget -P models/sd-vae-ft-mse https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json

# whisper weights
wget -P models/whisper https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt

# dwpose weights
wget -P models/dwpose https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth

# face-parse-bisent weights
wget -P models/face-parse-bisent https://github.com/zllrunning/face-parsing.PyTorch/releases/download/79999_iter.pth/79999_iter.pth
wget -P models/face-parse-bisent https://download.pytorch.org/models/resnet18-5c106cde.pth

# Verify installation
echo "MuseTalk environment setup completed."

# End of script
