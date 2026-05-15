import torch.nn as nn

from src.model.discriminator.stft_discriminator import STFTDiscriminator
from src.model.discriminator.waves_discriminator import WAVEDiscriminator


class SoundStreamDiscriminator(nn.Module):
    def __init__(self, negative_slope, waves, stft):
        super().__init__()

        self.stft_discriminator = STFTDiscriminator(
            window_length=stft.window_length,
            hop_length=stft.hop_length,
            frequency_bins=stft.frequency_bins,
            hidden_channels=stft.hidden_channels,
            negative_slope=negative_slope,
            input_kernel_size=stft.input_kernel_size,
            residual_kernel_size=stft.residual_kernel_size,
            channel_multipliers=stft.channel_multipliers,
            strides=stft.strides,
        )

        self.waves_discriminator1 = WAVEDiscriminator(
            downsample_factor=1,
            negative_slope=negative_slope,
            input_kernel_size=waves.input_kernel_size,
            input_out_channels=waves.input_out_channels,
            stack_kernel_size=waves.stack_kernel_size,
            stack_stride=waves.stack_stride,
            stack_out_channels=waves.stack_out_channels,
            stack_groups=waves.stack_groups,
            head_kernel_sizes=waves.head_kernel_sizes,
            head_out_channels=waves.head_out_channels,
        )
        self.waves_discriminator2 = WAVEDiscriminator(
            downsample_factor=2,
            negative_slope=negative_slope,
            input_kernel_size=waves.input_kernel_size,
            input_out_channels=waves.input_out_channels,
            stack_kernel_size=waves.stack_kernel_size,
            stack_stride=waves.stack_stride,
            stack_out_channels=waves.stack_out_channels,
            stack_groups=waves.stack_groups,
            head_kernel_sizes=waves.head_kernel_sizes,
            head_out_channels=waves.head_out_channels,
        )
        self.waves_discriminator4 = WAVEDiscriminator(
            downsample_factor=4,
            negative_slope=negative_slope,
            input_kernel_size=waves.input_kernel_size,
            input_out_channels=waves.input_out_channels,
            stack_kernel_size=waves.stack_kernel_size,
            stack_stride=waves.stack_stride,
            stack_out_channels=waves.stack_out_channels,
            stack_groups=waves.stack_groups,
            head_kernel_sizes=waves.head_kernel_sizes,
            head_out_channels=waves.head_out_channels,
        )

    def forward(self, x):
        logits_waves1, features_waves1 = self.waves_discriminator1(x)
        logits_waves2, features_waves2 = self.waves_discriminator2(x)
        logits_waves4, features_waves4 = self.waves_discriminator4(x)

        logits_stft, features_stft = self.stft_discriminator(x)

        return {
            "logits": [logits_waves1, logits_waves2, logits_waves4, logits_stft],
            "features": [
                features_waves1,
                features_waves2,
                features_waves4,
                features_stft,
            ],
        }
