import argparse
import os
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F

from av2av.av2unit.inference import load_model as load_av2unit_model
from av2av.fairseq.fairseq import utils


class AVSpeechToUnitExtractor:
    def __init__(self, av2unit_model_path, use_cuda=False):
        self.use_cuda = use_cuda
        self.av2unit_model, self.av2unit_task = load_av2unit_model(
            av2unit_model_path, "audio", use_cuda=use_cuda
        )

    def process_audio(self, audio_path):
        # Load and preprocess audio
        audio_feats = self._load_audio_features(audio_path)

        # Prepare the sample
        collated_audios, _, _ = self.av2unit_task.dataset.collater_audio(
            [audio_feats], len(audio_feats)
        )
        sample = {"source": {"audio": collated_audios, "video": None}}
        sample = utils.move_to_cuda(sample) if self.use_cuda else sample

        # Run inference
        pred = self.av2unit_task.inference(self.av2unit_model, sample)
        return pred.int()

    def _load_audio_features(self, audio_path):
        task = self.av2unit_task
        _, audio_feats = task.dataset.load_feature((None, audio_path))
        audio_feats = (
            torch.from_numpy(audio_feats.astype(np.float32))
            if audio_feats is not None
            else None
        )
        if task.dataset.normalize and "audio" in task.dataset.modalities:
            with torch.no_grad():
                audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
        return audio_feats


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio_path", type=str, required=True, help="Path to the input audio file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the AV2Unit model checkpoint",
    )
    args = parser.parse_args()

    # Use CUDA if available
    use_cuda = torch.cuda.is_available()
    print("Use cuda", use_cuda)

    # Create the unit extractor
    extractor = AVSpeechToUnitExtractor(args.model_path, use_cuda=use_cuda)

    # Process the input audio file and print the src_unit
    src_unit = extractor.process_audio(args.audio_path)
    print(f"src_unit: {src_unit}")
