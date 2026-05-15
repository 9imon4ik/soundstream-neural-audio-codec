import torch
from torch import nn


class SoundStreamDiscriminatorLoss(nn.Module):
    def forward(self, real_logits, fake_logits):
        loss_real = sum(
            torch.relu(1 - real_logit).mean() for real_logit in real_logits
        ) / len(real_logits)
        loss_fake = sum(
            torch.relu(1 + fake_logit).mean() for fake_logit in fake_logits
        ) / len(fake_logits)

        return loss_real + loss_fake
