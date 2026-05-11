import torch.nn as nn


class SoundStreamDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.discriminator = nn.Sequential()

    def forward(self, x):
        logits = self.discriminator(x)

        return {
            "logits": logits,
            "features": [],
        }
