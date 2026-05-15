import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, codebook_size, decay, dead_code_threshold):
        super().__init__()

        self.decay = decay
        self.dead_code_threshold = dead_code_threshold
        self.register_buffer("codebook", torch.randn(codebook_size, embedding_dim))
        self.register_buffer("ema_counts", torch.zeros(codebook_size))
        self.register_buffer("ema_sums", torch.zeros(codebook_size, embedding_dim))

    @torch.no_grad()
    def init_codebook(self, x, kmeans_iters):
        B, D, T = x.shape
        x_flat = x.permute(0, 2, 1).reshape(B * T, D)

        indices = torch.randint(
            0, x_flat.shape[0], (self.codebook.shape[0],), device=x.device
        )
        centers = x_flat[indices]

        for _ in range(kmeans_iters):
            distances = (
                (x_flat**2).sum(1, keepdim=True)
                + (centers**2).sum(1)
                - 2 * x_flat @ centers.T
            )
            nearest = distances.argmin(1)

            for i in range(centers.shape[0]):
                mask = nearest == i

                if mask.any():
                    centers[i] = x_flat[mask].mean(0)

        self.codebook.copy_(centers)
        self.ema_sums.copy_(centers * self.dead_code_threshold)
        self.ema_counts.fill_(self.dead_code_threshold)

    @torch.no_grad()
    def update_codebook(self, x_flat, indices):
        counts = torch.bincount(indices, minlength=self.codebook.shape[0]).float()
        sums = torch.zeros_like(self.codebook)
        sums.index_add_(0, indices, x_flat)

        self.ema_counts.mul_(self.decay).add_(counts * (1 - self.decay))
        self.ema_sums.mul_(self.decay).add_(sums * (1 - self.decay))

        self.codebook.copy_(
            self.ema_sums / self.ema_counts.clamp(min=1e-5).unsqueeze(1)
        )

        dead = self.ema_counts < self.dead_code_threshold
        if dead.any():
            random_indices = torch.randint(
                0, x_flat.shape[0], (dead.sum(),), device=x_flat.device
            )
            self.codebook[dead] = x_flat[random_indices]
            self.ema_sums[dead] = x_flat[random_indices] * self.dead_code_threshold
            self.ema_counts[dead] = self.dead_code_threshold

    def forward(self, x, update_codebook=True):
        B, D, T = x.shape

        x_flat = x.permute(0, 2, 1).reshape(B * T, D)

        distances = (
            (x_flat**2).sum(1, keepdim=True)
            + (self.codebook**2).sum(1)
            - 2 * x_flat @ self.codebook.T
        )

        indices = distances.argmin(1)
        quantized = self.codebook[indices]

        if self.training and update_codebook:
            self.update_codebook(x_flat, indices)

        indices = indices.view(B, T)
        quantized = quantized.view(B, T, D).permute(0, 2, 1)

        return {
            "quantized": quantized,
            "indices": indices,
        }


class SoundStreamQuantizer(nn.Module):
    def __init__(
        self,
        num_quantizers,
        embedding_dim,
        codebook_size,
        kmeans_iters,
        decay,
        dead_code_threshold,
    ):
        super().__init__()

        self.kmeans_iters = kmeans_iters
        self.register_buffer("initialized", torch.tensor(False))

        self.quantizers = nn.ModuleList(
            [
                VectorQuantizer(
                    embedding_dim,
                    codebook_size,
                    decay,
                    dead_code_threshold,
                )
                for _ in range(num_quantizers)
            ]
        )

    @torch.no_grad()
    def init_codebooks(self, x):
        quantized_sum = torch.zeros_like(x)

        for quantizer in self.quantizers:
            residual = x - quantized_sum
            quantizer.init_codebook(residual, self.kmeans_iters)
            quantized_sum += quantizer(residual, update_codebook=False)["quantized"]

        self.initialized.fill_(True)

    def forward(self, x):
        if self.training and not self.initialized.item():
            self.init_codebooks(x)

        quantized_sum = torch.zeros_like(x)
        codebook_indices = []
        for quantizer in self.quantizers:
            residual = x - quantized_sum
            output = quantizer(residual)
            quantized = output["quantized"]
            indices = output["indices"]
            quantized_sum += quantized
            codebook_indices.append(indices)

        commitment_loss = F.mse_loss(x, quantized_sum.detach())

        codebook_indices = torch.stack(codebook_indices, dim=-1)

        return {
            "quantized": x + (quantized_sum - x).detach(),
            "codebook_indices": codebook_indices,
            "commitment_loss": commitment_loss,
        }
