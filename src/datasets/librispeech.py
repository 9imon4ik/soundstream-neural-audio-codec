from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, sample_rate, is_train, crop_seconds):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.is_train = is_train
        self.crop_seconds = crop_seconds
        self.files = sorted(Path(root_dir).rglob("*.flac"))
        if len(self.files) == 0:
            raise RuntimeError(f"No .flac files found in {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        audio_path = self.files[index]
        audio, original_sample_rate = torchaudio.load(audio_path)

        if original_sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(
                audio,
                orig_freq=original_sample_rate,
                new_freq=self.sample_rate,
            )

        if self.is_train:
            audio_len = audio.shape[1]
            crop_size = int(self.crop_seconds * self.sample_rate)

            if audio_len < crop_size:
                audio = F.pad(audio, (0, crop_size - audio_len), mode="replicate")
                audio_len = audio.shape[1]

            start = torch.randint(0, audio_len - crop_size + 1, (1,)).item()
            audio = audio[:, start : start + crop_size]

        return {"audio": audio, "audio_path": str(audio_path)}
