import torch.nn as nn

from src.model.modules import CausalConv1d, CausalConvTranspose1d, ResidualUnit


class DecoderBlock(nn.Module):
    def __init__(self, channels, stride, residual_dilations, residual_kernel_size):
        super().__init__()

        self.decoder = nn.Sequential(
            CausalConvTranspose1d(
                in_channels=channels,
                out_channels=channels // 2,
                kernel_size=stride * 2,
                stride=stride,
            ),
            nn.ELU(),
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
        )

    def forward(self, x):
        return self.decoder(x)


class SoundStreamDecoder(nn.Module):
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

        self.decoder = nn.Sequential(
            CausalConv1d(
                in_channels=embedding_dim,
                out_channels=16 * hidden_channels,
                kernel_size=input_kernel_size,
            ),
            nn.ELU(),
            DecoderBlock(
                channels=hidden_channels * 16,
                stride=strides[3],
                residual_dilations=residual_dilations,
                residual_kernel_size=residual_kernel_size,
            ),
            DecoderBlock(
                channels=hidden_channels * 8,
                stride=strides[2],
                residual_dilations=residual_dilations,
                residual_kernel_size=residual_kernel_size,
            ),
            DecoderBlock(
                channels=hidden_channels * 4,
                stride=strides[1],
                residual_dilations=residual_dilations,
                residual_kernel_size=residual_kernel_size,
            ),
            DecoderBlock(
                channels=hidden_channels * 2,
                stride=strides[0],
                residual_dilations=residual_dilations,
                residual_kernel_size=residual_kernel_size,
            ),
            CausalConv1d(
                in_channels=hidden_channels * 1,
                out_channels=1,
                kernel_size=output_kernel_size,
            ),
        )

    def forward(self, x):
        return self.decoder(x)
