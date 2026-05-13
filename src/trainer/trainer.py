import torch

from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics):
        batch = self.move_batch_to_device(batch)

        if self.is_train:
            discriminator_losses = self.train_discriminator(batch)
            generator_outputs, generator_losses = self.train_generator(batch)

            batch.update(discriminator_losses)
            batch.update(generator_outputs)
            batch.update(generator_losses)
            metric_funcs = self.metrics["train"]
        else:
            batch = self.evaluate_batch(batch)
            metric_funcs = self.metrics["inference"]

        for loss_name in self.config.writer.loss_names:
            if loss_name not in batch:
                continue
            value = batch[loss_name]
            if isinstance(value, torch.Tensor):
                value = value.item()
            metrics.update(loss_name, value)

        for metric in metric_funcs:
            metrics.update(metric.name, metric(**batch))

        return batch

    def train_discriminator(self, batch):
        audio = batch["audio"]

        with torch.no_grad():
            generator_outputs = self.generator(audio)
            reconstructed_audio = generator_outputs["reconstructed_audio"]

        real_outputs = self.discriminator(audio)
        fake_outputs = self.discriminator(reconstructed_audio.detach())
        losses = self.discriminator_criterion(
            real_outputs=real_outputs,
            fake_outputs=fake_outputs,
        )

        self.discriminator_optimizer.zero_grad()
        losses["loss_discriminator"].backward()
        self.discriminator_optimizer.step()

        return self._detach_values(losses)

    def train_generator(self, batch):
        audio = batch["audio"]

        generator_outputs = self.generator(audio)
        reconstructed_audio = generator_outputs["reconstructed_audio"]

        fake_outputs = self.discriminator(reconstructed_audio)
        with torch.no_grad():
            real_outputs = self.discriminator(audio)

        losses = self.generator_criterion(
            real_audio=audio,
            reconstructed_audio=reconstructed_audio,
            real_outputs=real_outputs,
            fake_outputs=fake_outputs,
            commitment_loss=generator_outputs["commitment_loss"],
        )

        self.generator_optimizer.zero_grad()
        losses["loss_generator"].backward()
        self.generator_optimizer.step()

        return generator_outputs, self._detach_values(losses)

    def evaluate_batch(self, batch):
        audio = batch["audio"]
        generator_outputs = self.generator(audio)
        reconstructed_audio = generator_outputs["reconstructed_audio"]

        real_outputs = self.discriminator(audio)
        fake_outputs = self.discriminator(reconstructed_audio)

        discriminator_losses = self.discriminator_criterion(
            real_outputs=real_outputs,
            fake_outputs=fake_outputs,
        )
        generator_losses = self.generator_criterion(
            real_audio=audio,
            reconstructed_audio=reconstructed_audio,
            real_outputs=real_outputs,
            fake_outputs=fake_outputs,
            commitment_loss=generator_outputs["commitment_loss"],
        )

        batch.update(generator_outputs)
        batch.update(self._detach_values(discriminator_losses))
        batch.update(self._detach_values(generator_losses))
        return batch

    @staticmethod
    def _detach_values(values):
        return {
            key: value.detach() if hasattr(value, "detach") else value
            for key, value in values.items()
        }

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            # Log Stuff
            pass
        else:
            # Log Stuff
            pass
