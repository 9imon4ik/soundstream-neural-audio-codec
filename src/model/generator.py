import torch
import torch.nn as nn


class SoundStreamGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential()

        self.quantizer = nn.Sequential()

        self.decoder = nn.Sequential()

    def forward(self, x):
        encoded = self.encoder(x)
        quantized = self.quantizer(encoded)
        decoded = self.decoder(quantized)

        return {
            "reconstructed_audio": decoded,
        }
