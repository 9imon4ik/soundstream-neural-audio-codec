import torch.nn as nn

from src.model.modules import CausalConv1d, ResidualUnit


class EncoderBlock(nn.Module):
    def __init__(self, channels, stride, residual_dilations, residual_kernel_size):
        super().__init__()

        self.encoder = nn.Sequential(
            ResidualUnit(
                channels // 2,
                kernel_size=residual_kernel_size,
                dilation=residual_dilations[0],
            ),
            ResidualUnit(
                channels // 2,
                kernel_size=residual_kernel_size,
                dilation=residual_dilations[1],
            ),
            ResidualUnit(
                channels // 2,
                kernel_size=residual_kernel_size,
                dilation=residual_dilations[2],
            ),
            CausalConv1d(
                in_channels=channels // 2,
                out_channels=channels,
                kernel_size=stride * 2,
                stride=stride,
            ),
            nn.ELU(),
        )

    def forward(self, x):
        return self.encoder(x)


class SoundStreamEncoder(nn.Module):
    def __init__(
        self,
        hidden_channels,
        embedding_dim,
        strides,
        residual_dilations,
        input_kernel_size,
        residual_kernel_size,
        output_kernel_size,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            CausalConv1d(
                in_channels=1,
                out_channels=hidden_channels,
                kernel_size=input_kernel_size,
            ),
            nn.ELU(),
            EncoderBlock(
                channels=hidden_channels * 2,
                stride=strides[0],
                residual_dilations=residual_dilations,
                residual_kernel_size=residual_kernel_size,
            ),
            EncoderBlock(
                channels=hidden_channels * 4,
                stride=strides[1],
                residual_dilations=residual_dilations,
                residual_kernel_size=residual_kernel_size,
            ),
            EncoderBlock(
                channels=hidden_channels * 8,
                stride=strides[2],
                residual_dilations=residual_dilations,
                residual_kernel_size=residual_kernel_size,
            ),
            EncoderBlock(
                channels=hidden_channels * 16,
                stride=strides[3],
                residual_dilations=residual_dilations,
                residual_kernel_size=residual_kernel_size,
            ),
            CausalConv1d(
                in_channels=hidden_channels * 16,
                out_channels=embedding_dim,
                kernel_size=output_kernel_size,
            ),
        )

    def forward(self, x):
        return self.encoder(x)
