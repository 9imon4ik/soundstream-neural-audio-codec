import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class WAVEDiscriminator(nn.Module):
    def __init__(
        self,
        downsample_factor,
        negative_slope,
        input_kernel_size,
        input_out_channels,
        stack_kernel_size,
        stack_stride,
        stack_out_channels,
        stack_groups,
        head_kernel_sizes,
        head_out_channels,
    ):
        super().__init__()

        self.downsample_factor = downsample_factor

        self.input_conv = weight_norm(
            nn.Conv1d(
                in_channels=1,
                out_channels=input_out_channels,
                kernel_size=input_kernel_size,
                stride=1,
                padding=input_kernel_size // 2,
            )
        )
        self.input_activation = nn.LeakyReLU(negative_slope=negative_slope)

        channels = input_out_channels
        stack_convs = []
        stack_activations = []
        for i in range(len(stack_out_channels)):
            stack_convs.append(
                weight_norm(
                    nn.Conv1d(
                        in_channels=channels,
                        out_channels=stack_out_channels[i],
                        kernel_size=stack_kernel_size,
                        stride=stack_stride,
                        padding=stack_kernel_size // 2,
                        groups=stack_groups[i],
                    )
                )
            )
            stack_activations.append(nn.LeakyReLU(negative_slope=negative_slope))
            channels = stack_out_channels[i]

        self.stack_convs = nn.ModuleList(stack_convs)
        self.stack_activations = nn.ModuleList(stack_activations)

        self.head_conv = weight_norm(
            nn.Conv1d(
                in_channels=stack_out_channels[3],
                out_channels=head_out_channels[0],
                kernel_size=head_kernel_sizes[0],
                stride=1,
                padding=head_kernel_sizes[0] // 2,
            )
        )
        self.head_activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.logits_conv = weight_norm(
            nn.Conv1d(
                in_channels=head_out_channels[0],
                out_channels=head_out_channels[1],
                kernel_size=head_kernel_sizes[1],
                stride=1,
                padding=head_kernel_sizes[1] // 2,
            )
        )

    def forward(self, x):
        for _ in range(self.downsample_factor.bit_length() - 1):
            x = F.avg_pool1d(
                x,
                kernel_size=4,
                stride=2,
                padding=1,
            )

        features = []

        x = self.input_conv(x)
        x = self.input_activation(x)
        features.append(x)

        for stack_conv, stack_activation in zip(
            self.stack_convs, self.stack_activations
        ):
            x = stack_conv(x)
            x = stack_activation(x)
            features.append(x)

        x = self.head_conv(x)
        x = self.head_activation(x)
        features.append(x)

        logits = self.logits_conv(x)

        return logits, features
