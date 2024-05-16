import os

import numpy as np
import torch
import torch.nn as nn
from snake import Snake


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(Block, self).__init__()
        padding = (dilation * (kernel_size - 1)) // 2
        self.In1 = nn.InstanceNorm1d(in_channels, affine=True)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        self.In2 = nn.InstanceNorm1d(out_channels, affine=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            dilation=1,
        )
        self.ln3 = nn.InstanceNorm1d(out_channels, affine=True)
        nn.init.orthogonal_(self.conv1.weight)
        nn.init.orthogonal_(self.conv2.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.constant_(self.ln3.weight, 0.2)

        self.model = nn.Sequential(
            Snake([in_channels, 1]),
            self.conv1,
            self.In2,
            Snake([out_channels, 1]),
            self.conv2,
            self.ln3,
        )

    def forward(self, x):
        return x + self.model(x)


class WhisperCNN(nn.Module):
    def __init__(
        self, num_tokens, embedding_dim, hidden_dim, output_dim, kernel_sizes, dilations
    ):
        super(WhisperCNN, self).__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_dim)
        self.conv_transpose = nn.ConvTranspose1d(
            embedding_dim, hidden_dim, kernel_size=15, stride=5, padding=5
        )

        self.backbone = nn.Sequential(
            Block(hidden_dim, hidden_dim, kernel_sizes, dilations),
            Block(hidden_dim, hidden_dim, kernel_sizes, dilations),
            Block(hidden_dim, hidden_dim, kernel_sizes, dilations),
            Block(hidden_dim, hidden_dim, kernel_sizes, dilations),
        )

        self.conv = nn.Conv1d(hidden_dim, output_dim, kernel_size=1)
        self.In1 = nn.InstanceNorm1d(hidden_dim, affine=True)
        self.In2 = nn.InstanceNorm1d(hidden_dim, affine=True)
        self.ln = nn.LayerNorm(output_dim)

        nn.init.orthogonal_(self.conv_transpose.weight)
        nn.init.orthogonal_(self.conv.weight)
        nn.init.zeros_(self.conv_transpose.bias)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv_transpose(x)
        x = self.In1(x)
        x = self.backbone(x)
        x = self.In2(x)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        return x
