from torch import nn


class SoundStreamDiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, real_outputs, fake_outputs):
        loss = max(0, 1.0 - real_outputs).mean() + max(0, 1.0 + fake_outputs).mean()

        return loss
