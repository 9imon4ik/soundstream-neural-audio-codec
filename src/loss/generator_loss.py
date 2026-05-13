from torch import nn


class SoundStreamGeneratorLoss(nn.Module):
    def __init__(self, lambda_adv, lambda_feat, lambda_rec, lambda_com):
        super().__init__()

        self.lambda_adv = lambda_adv
        self.lambda_feat = lambda_feat
        self.lambda_rec = lambda_rec
        self.lambda_com = lambda_com

    def forward(
        self,
        real_audio,
        reconstructed_audio,
        real_outputs,
        fake_outputs,
        commitment_loss,
    ):
        loss_adv = max(0, 1.0 - fake_outputs).mean()

        loss_feat = 0
        loss_rec = 0

        loss = (
            self.lambda_adv * loss_adv
            + self.lambda_feat * loss_feat
            + self.lambda_rec * loss_rec
            + self.lambda_com * commitment_loss
        )

        return loss
