import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any

# Custom library imports
from pipelinegen_histones.models.mixins import EncoderDecoderMixin
from pipelinegen.core.models.factory import ModelFactory


class SimpleHistonesAutoencoder(EncoderDecoderMixin):
    def __init__(self, latent_dim: int = 8):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(60, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 60),
            nn.Sigmoid(),  # Use Sigmoid to ensure output values are between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    @classmethod
    def build(cls, config: Dict[str, Any]):
        return cls(latent_dim=config.get("latent_dim", 8))


class ConvHistonesAutoencoder(EncoderDecoderMixin):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=0),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(
                16, 32, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose1d(
                32, 64, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1792, 60),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x[:, None, :]
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.squeeze()
        return x

    def encode(self, x: torch.Tensor):
        x = x[:, None, :]
        x = self.encoder(x)
        # reshape to 1D
        x = x.flatten(1)
        return x.squeeze()

    def decode(self, x: torch.Tensor):
        x = x[:, None, :]
        x = self.decoder(x)
        x = x.squeeze()
        return x

    @classmethod
    def build(cls, config: Dict[str, Any]):
        return cls()


ModelFactory.register_builder(
    "SimpleHistonesAutoencoder", SimpleHistonesAutoencoder.build
)
ModelFactory.register_builder("ConvHistonesAutoencoder", ConvHistonesAutoencoder.build)
