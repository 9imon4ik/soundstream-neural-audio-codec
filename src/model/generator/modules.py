import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, stride):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
        )

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

    def forward(self, x):
        target_len = x.shape[-1] * self.stride
        x = self.conv(x)
        right_crop = x.shape[-1] - target_len
        if right_crop > 0:
            x = x[..., :-right_crop]
        return x


class ResidualUnit(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()

        self.unit = nn.Sequential(
            CausalConv1d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                stride=1,
            ),
            nn.ELU(),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1),
            nn.ELU(),
        )

    def forward(self, x):
        return x + self.unit(x)
