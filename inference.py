import io

import hydra
import requests
import torchaudio

from evaluate import get_device, load_generator, reconstruct_batch
from src.utils.io_utils import ROOT_PATH


def load_audio(path, sample_rate):
    path = str(path)
    if path.startswith("http"):
        wav, sr = torchaudio.load(io.BytesIO(requests.get(path, timeout=60).content))
    else:
        wav, sr = torchaudio.load(path)
    audio = torchaudio.functional.resample(wav.mean(0, keepdim=True), sr, sample_rate)
    return audio.unsqueeze(0) if audio.dim() == 2 else audio


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    device = get_device()
    fs = config.audio.sample_rate

    generator = load_generator(config, device)
    audio = load_audio(config.inferencer.input, fs)

    recon = reconstruct_batch(generator, audio, device)

    out = ROOT_PATH / (config.inferencer.output or "data/saved/inference/reconstructed.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(out), recon.squeeze(0).cpu(), fs)

    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
