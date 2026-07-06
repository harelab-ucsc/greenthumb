"""
File:
    models.py

Description:
    Temporally-encoded models with regression heads.

Authors:
    Taylor Kergan
    nubby

Date:
    6 Jul 2026

Version:
    1.0.1
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


def masked_mean(sequence: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Computes a length-aware mean along the time dimension."""
    if sequence.dim() != 3:
        raise ValueError("Expected sequence tensor of shape (batch, time, features).")
    batch_size, max_time, _ = sequence.shape
    device = sequence.device
    mask = torch.arange(max_time, device=device).expand(batch_size, max_time)
    mask = mask < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).type_as(sequence)
    summed = (sequence * mask).sum(dim=1)
    denom = lengths.clamp(min=1).unsqueeze(-1).type_as(sequence)
    return summed / denom


class LSTMEstimator(nn.Module):
    """
    LSTMEstimator

    Baseline bi-directional LSTM estimator.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        bidirectional = True
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, num_classes),
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            inputs, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # Concatenate the final states from both directions.
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.head(last_hidden)


class TemporalBlock(nn.Module):
    """Two-layer residual block used inside the TCN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = ((kernel_size - 1) * dilation) // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class TemporalConvNet(nn.Module):
    """Stack of residual temporal convolution blocks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        in_channels = input_dim
        for layer_idx in range(num_layers):
            dilation = 2 ** layer_idx
            layers.append(TemporalBlock(in_channels, hidden_dim, kernel_size, dilation, dropout))
            in_channels = hidden_dim
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs.transpose(1, 2)  # (batch, features, time)
        out = self.network(x)
        return out.transpose(1, 2)  # back to (batch, time, features)


class TemporalConvNetEstimator(nn.Module):
    """Estimator head on top of a TCN backbone."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 160,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.tcn = TemporalConvNet(input_dim, hidden_dim, num_layers, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        features = self.tcn(inputs)
        pooled = masked_mean(features, lengths)
        return self.head(pooled)


class PositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEstimator(nn.Module):
    """Compact Transformer encoder estimator."""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 16,
        dropout: float = 0.1,
        num_classes: int = 3,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    # TODO: Find where dimension mismatch between layers is occurring.
    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = torch.arange(inputs.size(1), device=inputs.device).expand(inputs.size(0), inputs.size(1))
        mask = mask >= lengths.unsqueeze(1)
        #print(inputs)
        projected = self.input_projection(inputs)
        encoded = self.positional_encoding(projected)
        encoded = self.encoder(encoded, src_key_padding_mask=mask)
        pooled = masked_mean(encoded, lengths)
        return self.head(pooled)
