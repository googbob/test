import os
import logging

import torch
from PIL import Image
from torch.utils.data import get_worker_info


def _quiet_http_worker(worker_id):
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpx2').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)


class HuggingFaceImageStream(torch.utils.data.IterableDataset):

    def __init__(
        self,
        dataset_name,
        split,
        image_key,
        transform,
        steps_per_epoch,
        batch_size,
        world_size=1,
        rank=0,
        config_name=None,
        token_env='HF_TOKEN',
        shuffle_buffer_size=10000,
    ):
        super().__init__()
        _quiet_http_worker(0)
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                'Install the optional Hugging Face dependency with '
                '`pip install datasets` to use streaming.'
            ) from exc

        token = os.environ.get(token_env)
        stream = load_dataset(
            dataset_name,
            name=config_name,
            split=split,
            streaming=True,
            token=token,
        )
        stream = stream.shuffle(buffer_size=shuffle_buffer_size, seed=0)

        self.stream = stream
        self.image_key = image_key
        self.transform = transform
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank

    def __iter__(self):
        worker_info = get_worker_info()
        worker_count = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        sample_count = self.steps_per_epoch * self.batch_size
        samples_per_worker, remainder = divmod(sample_count, worker_count)
        worker_sample_count = samples_per_worker + (worker_id < remainder)

        stream = self.stream
        if worker_count > 1 or self.world_size > 1:
            stream = stream.shard(
                num_shards=self.world_size * worker_count,
                index=self.rank * worker_count + worker_id,
            )

        for sample_index, example in enumerate(stream):
            if sample_index >= worker_sample_count:
                break
            image = example[self.image_key]
            if not isinstance(image, Image.Image):
                raise TypeError(
                    f"Expected '{self.image_key}' to decode as a PIL image, "
                    f'got {type(image).__name__}.'
                )
            yield self.transform(image.convert('RGB')), 0

    def __len__(self):
        return self.steps_per_epoch * self.batch_size

    def set_epoch(self, epoch):
        self.stream.set_epoch(epoch)


def make_huggingface_stream(
    transform,
    batch_size,
    collator,
    pin_mem,
    num_workers,
    world_size,
    rank,
    dataset_name,
    split,
    image_key,
    steps_per_epoch,
    config_name=None,
    token_env='HF_TOKEN',
    shuffle_buffer_size=10000,
):
    dataset = HuggingFaceImageStream(
        dataset_name=dataset_name,
        split=split,
        image_key=image_key,
        transform=transform,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        config_name=config_name,
        token_env=token_env,
        shuffle_buffer_size=shuffle_buffer_size,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
        worker_init_fn=_quiet_http_worker,
    )
    return dataset, data_loader, dataset