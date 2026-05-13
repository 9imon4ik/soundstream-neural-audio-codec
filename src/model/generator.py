import torch.nn as nn

from src.model.decoder import SoundStreamDecoder
from src.model.encoder import SoundStreamEncoder
from src.model.quantizer import SoundStreamQuantizer


class SoundStreamGenerator(nn.Module):
    def __init__(
        self,
        strides,
        hidden_channels,
        embedding_dim,
        residual_dilations,
        encoder,
        decoder,
        quantizer,
    ):
        super().__init__()

        self.encoder = SoundStreamEncoder(
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            strides=strides,
            residual_dilations=residual_dilations,
            input_kernel_size=encoder.input_kernel_size,
            residual_kernel_size=encoder.residual_kernel_size,
            output_kernel_size=encoder.output_kernel_size,
        )

        self.quantizer = SoundStreamQuantizer(
            num_quantizers=quantizer.num_quantizers,
            embedding_dim=embedding_dim,
            codebook_size=quantizer.codebook_size,
            kmeans_iters=quantizer.kmeans_iters,
            decay=quantizer.decay,
            dead_code_threshold=quantizer.dead_code_threshold,
        )

        self.decoder = SoundStreamDecoder(
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            strides=strides,
            residual_dilations=residual_dilations,
            input_kernel_size=decoder.input_kernel_size,
            residual_kernel_size=decoder.residual_kernel_size,
            output_kernel_size=decoder.output_kernel_size,
        )

    def forward(self, x):
        encoded = self.encoder(x)
        quantized = self.quantizer(encoded)
        decoded = self.decoder(quantized["quantized"])

        return {
            "reconstructed_audio": decoded,
            "codebook_indices": quantized["codebook_indices"],
            "commitment_loss": quantized["commitment_loss"],
        }
