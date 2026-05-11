import warnings

import hydra
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer.trainer import Trainer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()


@hydra.main(version_base=None, config_path="src/configs", config_name="soundstream")
def main(config):
    """
    Main script for training. Instantiates the model, optimizer, scheduler,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.trainer.seed)

    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    if config.trainer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.trainer.device

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    generator = instantiate(config.generator).to(device)
    logger.info(generator)
    discriminator = instantiate(config.discriminator).to(device)
    logger.info(discriminator)

    # build optimizer, learning rate scheduler
    generator_params = filter(lambda p: p.requires_grad, generator.parameters())
    generator_optimizer = instantiate(
        config.generator_optimizer, params=generator_params
    )
    discriminator_params = filter(lambda p: p.requires_grad, discriminator.parameters())
    discriminator_optimizer = instantiate(
        config.discriminator_optimizer, params=discriminator_params
    )

    # get function handles of loss and metrics
    generator_loss_function = instantiate(config.generator_loss_function).to(device)
    discriminator_loss_function = instantiate(config.discriminator_loss_function).to(
        device
    )
    metrics = instantiate(config.metrics)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        generator_criterion=generator_loss_function,
        discriminator_criterion=discriminator_loss_function,
        metrics=metrics,
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        config=config,
        device=device,
        dataloaders=dataloaders,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
