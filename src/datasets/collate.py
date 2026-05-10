import torch
import torch.nn.functional as F


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    max_audio_len = max(item["audio"].shape[-1] for item in dataset_items)
    result_batch["audio"] = torch.stack(
        [
            F.pad(item["audio"], (0, max_audio_len - item["audio"].shape[-1]))
            for item in dataset_items
        ]
    )
    result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]

    return result_batch
