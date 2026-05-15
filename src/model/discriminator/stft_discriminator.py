import torch
import torch.nn as nn

from src.model.discriminator.modules import ResidualUnit


class STFTDiscriminator(nn.Module):
    def __init__(
        self,
        window_length,
        hop_length,
        frequency_bins,
        hidden_channels,
        negative_slope,
        input_kernel_size,
        residual_kernel_size,
        channel_multipliers,
        strides,
    ):
        super().__init__()

        self.window_length = window_length
        self.hop_length = hop_length

        self.input_conv = nn.Conv2d(
            in_channels=2,
            out_channels=hidden_channels,
            kernel_size=input_kernel_size,
            padding=(input_kernel_size[0] // 2, input_kernel_size[1] // 2),
        )
        self.input_activation = nn.LeakyReLU(negative_slope=negative_slope)

        channels = hidden_channels
        residual_units = []
        for i in range(len(channel_multipliers)):
            residual_units.append(
                ResidualUnit(
                    channels,
                    residual_kernel_size,
                    channel_multipliers[i],
                    strides[i],
                    negative_slope,
                )
            )
            channels *= channel_multipliers[i]
        self.residual_units = nn.ModuleList(residual_units)

        self.logits_conv = nn.Conv2d(
            in_channels=hidden_channels * 16,
            out_channels=1,
            kernel_size=(1, frequency_bins // (2 ** len(strides))),
        )

    def forward(self, x):
        x = x.squeeze(1)

        frames = x.unfold(dimension=1, size=self.window_length, step=self.hop_length)
        frames = frames * torch.hann_window(
            self.window_length,
            device=x.device,
            dtype=x.dtype,
        )

        stft = torch.fft.rfft(frames, dim=-1)
        stft = torch.view_as_real(stft)

        stft = stft.permute(0, 3, 1, 2)

        features = []

        x = self.input_conv(stft)
        x = self.input_activation(x)
        features.append(x)

        for residual_unit in self.residual_units:
            x = residual_unit(x)
            features.append(x)

        logits = self.logits_conv(x)

        return logits, features
