import torch.nn as nn


class ResidualUnit(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size,
        channel_multiplier,
        stride,
        negative_slope,
    ):
        super().__init__()

        self.unit = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            ),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels * channel_multiplier,
                kernel_size=(stride[0] + 2, stride[1] + 2),
                stride=stride,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=negative_slope),
        )

        self.skip = nn.Conv2d(
            in_channels=channels,
            out_channels=channels * channel_multiplier,
            kernel_size=(stride[0] + 2, stride[1] + 2),
            stride=stride,
            padding=1,
        )

    def forward(self, x):
        return self.skip(x) + self.unit(x)
