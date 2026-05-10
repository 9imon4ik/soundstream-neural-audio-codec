import tarfile
from pathlib import Path
from urllib.request import urlretrieve

data_path = Path("data/LibriSpeech")
data_path.mkdir(parents=True, exist_ok=True)

for split in ["train-clean-100", "test-clean"]:
    archive_path = data_path / f"{split}.tar.gz"
    urlretrieve(f"http://www.openslr.org/resources/12/{split}.tar.gz", archive_path)
    with tarfile.open(archive_path) as f:
        f.extractall(data_path)
    archive_path.unlink()
