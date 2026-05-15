import math

import torch
import torchaudio
from torch import nn


class SoundStreamGeneratorLoss(nn.Module):
    def __init__(self, lambda_adv, lambda_feat, lambda_rec, lambda_com):
        super().__init__()

        self.lambda_adv = lambda_adv
        self.lambda_feat = lambda_feat
        self.lambda_rec = lambda_rec
        self.lambda_com = lambda_com

        self.window_lengths = [2**window_power for window_power in range(6, 12)]
        self.mel_transforms = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=16000,
                    n_fft=window_length,
                    hop_length=window_length // 4,
                    n_mels=64,
                )
                for window_length in self.window_lengths
            ]
        )

    def _reconstruction_loss(self, real_audio, reconstructed_audio):
        loss_rec = 0.0

        for window_length, mel_transform in zip(
            self.window_lengths, self.mel_transforms
        ):
            real_mel = mel_transform(real_audio)
            reconstructed_mel = mel_transform(reconstructed_audio)

            loss_rec += (real_mel - reconstructed_mel).abs().mean()

            alpha_s = math.sqrt(window_length) / 2
            loss_rec += (
                alpha_s
                * (
                    (
                        torch.log(real_mel.clamp(min=1e-5))
                        - torch.log(reconstructed_mel.clamp(min=1e-5))
                    )
                    ** 2
                ).mean()
            )

        return loss_rec

    def forward(
        self,
        real_audio,
        reconstructed_audio,
        real_logits,
        fake_logits,
        real_features,
        fake_features,
        commitment_loss,
    ):
        loss_adv = sum(
            torch.relu(1 - fake_logit).mean() for fake_logit in fake_logits
        ) / len(fake_logits)

        loss_feat = 0.0
        num_feature_maps = 0

        for real_branch_features, fake_branch_features in zip(
            real_features, fake_features
        ):
            for real_feature, fake_feature in zip(
                real_branch_features, fake_branch_features
            ):
                loss_feat += (real_feature.detach() - fake_feature).abs().mean()
                num_feature_maps += 1

        loss_feat = loss_feat / num_feature_maps

        real_audio = real_audio.squeeze(1)
        reconstructed_audio = reconstructed_audio.squeeze(1)
        loss_rec = self._reconstruction_loss(real_audio, reconstructed_audio)

        return {
            "loss_generator": (
                self.lambda_adv * loss_adv
                + self.lambda_feat * loss_feat
                + self.lambda_rec * loss_rec
                + self.lambda_com * commitment_loss
            ),
            "loss_adversarial": loss_adv,
            "loss_feature_matching": loss_feat,
            "loss_reconstruction": loss_rec,
        }
