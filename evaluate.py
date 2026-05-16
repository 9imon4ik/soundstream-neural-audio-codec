import comet_ml
import hydra
import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from torchmetrics.functional.audio.nisqa import non_intrusive_speech_quality_assessment
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from tqdm.auto import tqdm

from src.datasets.data_utils import get_dataloaders
from src.utils.io_utils import ROOT_PATH

load_dotenv()


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_generator(config, device):
    generator = instantiate(config.generator).to(device).eval()
    ckpt = ROOT_PATH / (
        config.inferencer.checkpoint
        if "inferencer" in config
        else "saved/soundstream-final/checkpoint-epoch100.pth"
    )
    generator.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=False)["generator_state_dict"]
    )
    return generator


def reconstruct_batch(generator, audio, device):
    with torch.no_grad():
        return generator(audio.to(device))["reconstructed_audio"]


@hydra.main(version_base=None, config_path="src/configs", config_name="evaluate")
def main(config):
    device = get_device()
    fs = config.audio.sample_rate

    dataloaders = get_dataloaders(config)
    generator = load_generator(config, device)

    stoi_scores, nisqa_scores = [], []

    for batch in tqdm(dataloaders["test"]):
        batch["audio"] = batch["audio"].to(device)
        recon = reconstruct_batch(generator, batch["audio"], device)
        preds = recon.squeeze(1).cpu()
        target = batch["audio"].squeeze(1).cpu()
        stoi_scores.append(
            short_time_objective_intelligibility(preds, target, fs).mean().item()
        )
        nisqa_scores.append(non_intrusive_speech_quality_assessment(preds[0], fs)[0].item())

    stoi = sum(stoi_scores) / len(stoi_scores)
    nisqa = sum(nisqa_scores) / len(nisqa_scores)

    print(f"    stoi           : {stoi}")
    print(f"    nisqa_mos      : {nisqa}")

    comet_ml.login()
    exp = comet_ml.ExistingExperiment(experiment_key=config.writer.run_id)
    exp.log_metrics({"stoi": stoi, "nisqa_mos": nisqa})


if __name__ == "__main__":
    main()
