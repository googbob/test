import gc
import io
import logging
import os
import random
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import get_worker_info

logger = logging.getLogger(__name__)

_GLOBAL_SEED = 0
_DEFAULT_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (compatible; IJEPA-stream/1.0; '
        '+https://github.com/facebookresearch/ijepa)'
    ),
}


def _quiet_http_worker(worker_id):
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpx2').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)


def _require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            'Install the optional Hugging Face dependency with '
            '`pip install datasets` to use streaming.'
        ) from exc
    return load_dataset


def _to_pil(value: Any) -> Optional[Image.Image]:
    if value is None:
        return None
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        if value.get('bytes') is not None:
            try:
                return Image.open(io.BytesIO(value['bytes']))
            except Exception:
                return None
        path = value.get('path')
        if path:
            try:
                return Image.open(path)
            except Exception:
                return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return Image.open(io.BytesIO(bytes(value)))
        except Exception:
            return None
    return None


def _fetch_url_image(url: str, timeout: float) -> Optional[Image.Image]:
    if not url or not isinstance(url, str):
        return None
    url = url.strip()
    if not url.startswith(('http://', 'https://')):
        return None
    try:
        request = urllib.request.Request(url, headers=_DEFAULT_HEADERS)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
        if not payload:
            return None
        image = Image.open(io.BytesIO(payload))
        image.load()
        return image
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        return None
    except Exception:
        return None


def _is_valid_rgb(image: Optional[Image.Image], min_size: int) -> Optional[Image.Image]:
    if image is None:
        return None
    try:
        image = image.convert('RGB')
    except Exception:
        return None
    width, height = image.size
    if width < min_size or height < min_size:
        return None
    return image


def _extract_image(
    example: Dict[str, Any],
    image_mode: str,
    image_key: str,
    fallback_keys: Sequence[str],
    url_timeout: float,
    min_size: int,
    status_key: Optional[str],
    status_allow: Optional[Sequence[str]],
) -> Optional[Image.Image]:
    if status_key and status_allow is not None:
        status = example.get(status_key)
        if status not in status_allow:
            return None

    keys = (image_key,) + tuple(fallback_keys or ())
    if image_mode in ('pil', 'wds_jpg'):
        for key in keys:
            if key not in example:
                continue
            image = _is_valid_rgb(_to_pil(example[key]), min_size)
            if image is not None:
                return image
        return None

    if image_mode == 'pil_list':
        for key in keys:
            values = example.get(key)
            if not values:
                continue
            if not isinstance(values, (list, tuple)):
                values = [values]
            for value in values:
                image = _is_valid_rgb(_to_pil(value), min_size)
                if image is not None:
                    return image
        return None

    if image_mode == 'url':
        for key in keys:
            if key not in example:
                continue
            image = _is_valid_rgb(
                _fetch_url_image(example[key], timeout=url_timeout),
                min_size,
            )
            if image is not None:
                return image
        return None

    raise ValueError(f'Unsupported image_mode: {image_mode}')


def _build_hf_stream(
    dataset_name: str,
    split: str,
    token: Optional[str],
    config_name: Optional[str] = None,
    all_configs: bool = False,
    columns: Optional[Sequence[str]] = None,
    shuffle_buffer_size: int = 1000,
    seed: int = _GLOBAL_SEED,
):
    load_dataset = _require_datasets()

    if all_configs or config_name == 'all':
        from datasets import get_dataset_config_names, interleave_datasets

        config_names = get_dataset_config_names(dataset_name, token=token)
        if not config_names:
            raise ValueError(f'No configs found for dataset {dataset_name}')
        parts = [
            load_dataset(
                dataset_name,
                name=name,
                split=split,
                streaming=True,
                token=token,
            )
            for name in config_names
        ]
        stream = interleave_datasets(parts, seed=seed)
    else:
        stream = load_dataset(
            dataset_name,
            name=config_name,
            split=split,
            streaming=True,
            token=token,
        )

    if columns:
        try:
            available = set(getattr(stream, 'column_names', None) or [])
            keep = [column for column in columns if not available or column in available]
            if keep:
                stream = stream.select_columns(list(keep))
        except Exception as exc:
            logging.getLogger(__name__).warning(
                'Could not select columns %s for %s: %s',
                columns,
                dataset_name,
                exc,
            )

    if shuffle_buffer_size and shuffle_buffer_size > 0:
        stream = stream.shuffle(buffer_size=shuffle_buffer_size, seed=seed)
    return stream


class _SourceState:
    __slots__ = (
        'name',
        'spec',
        'token',
        'default_shuffle',
        'stream',
        'image_mode',
        'image_key',
        'fallback_keys',
        'weight',
        'url_timeout',
        'min_size',
        'status_key',
        'status_allow',
    )

    def __init__(self, spec: Dict[str, Any], token: Optional[str], default_shuffle: int):
        self.name = spec.get('name') or spec['dataset_name']
        self.spec = spec
        self.token = token
        self.default_shuffle = default_shuffle
        self.image_mode = spec.get('image_mode', 'pil')
        self.image_key = spec.get('image_key', 'image')
        self.fallback_keys = list(spec.get('fallback_keys') or [])
        self.weight = float(spec.get('weight', 1.0))
        self.url_timeout = float(spec.get('url_timeout', 5.0))
        self.min_size = int(spec.get('min_size', 48))
        self.status_key = spec.get('status_key')
        allow = spec.get('status_allow')
        self.status_allow = list(allow) if allow is not None else None
        self.stream = None

    def ensure_stream(self, seed: Optional[int] = None):
        if self.stream is not None:
            return self.stream
        spec = self.spec
        self.stream = _build_hf_stream(
            dataset_name=spec['dataset_name'],
            split=spec.get('split', 'train'),
            token=self.token,
            config_name=spec.get('config_name'),
            all_configs=bool(spec.get('all_configs', False)),
            columns=spec.get('columns'),
            shuffle_buffer_size=int(
                spec.get('shuffle_buffer_size', self.default_shuffle)
            ),
            seed=int(spec.get('seed', seed if seed is not None else _GLOBAL_SEED)),
        )
        return self.stream

    def drop_stream(self):
        self.stream = None


class HuggingFaceImageStream(torch.utils.data.IterableDataset):
    """Single-dataset HF stream (kept for backward compatibility)."""

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
        image_mode='pil',
        fallback_keys=None,
        url_timeout=5.0,
        min_size=48,
        columns=None,
        status_key=None,
        status_allow=None,
        all_configs=False,
    ):
        super().__init__()
        _quiet_http_worker(0)
        token = os.environ.get(token_env)
        self.transform = transform
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.image_mode = image_mode
        self.image_key = image_key
        self.fallback_keys = list(fallback_keys or [])
        self.url_timeout = url_timeout
        self.min_size = min_size
        self.status_key = status_key
        self.status_allow = list(status_allow) if status_allow is not None else None
        self.stream = _build_hf_stream(
            dataset_name=dataset_name,
            split=split,
            token=token,
            config_name=config_name,
            all_configs=all_configs,
            columns=columns,
            shuffle_buffer_size=shuffle_buffer_size,
        )

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

        yielded = 0
        for example in stream:
            if yielded >= worker_sample_count:
                break
            image = _extract_image(
                example,
                image_mode=self.image_mode,
                image_key=self.image_key,
                fallback_keys=self.fallback_keys,
                url_timeout=self.url_timeout,
                min_size=self.min_size,
                status_key=self.status_key,
                status_allow=self.status_allow,
            )
            if image is None:
                continue
            yield self.transform(image), 0
            yielded += 1

    def __len__(self):
        return self.steps_per_epoch * self.batch_size

    def set_epoch(self, epoch):
        if hasattr(self.stream, 'set_epoch'):
            self.stream.set_epoch(epoch)


def _allocate_quotas(total: int, weights: Sequence[float]) -> List[int]:
    """Split `total` samples across sources proportional to weights."""
    weight_sum = sum(weights)
    if weight_sum <= 0 or total <= 0:
        return [0] * len(weights)
    raw = [total * (w / weight_sum) for w in weights]
    quotas = [int(x) for x in raw]
    leftover = total - sum(quotas)
    # Give remaining samples to largest fractional remainders.
    order = sorted(
        range(len(weights)),
        key=lambda i: (raw[i] - quotas[i], weights[i]),
        reverse=True,
    )
    for i in order[:leftover]:
        quotas[i] += 1
    return quotas


class HuggingFaceMultiImageStream(torch.utils.data.IterableDataset):
    """Multi-source HF stream with defective URL skipping.

    schedule:
      - 'interleave': weighted random mix (all streams open; higher RAM)
      - 'sequential': one source at a time; order shuffled each epoch (low RAM)
    """

    def __init__(
        self,
        sources: List[Dict[str, Any]],
        transform,
        steps_per_epoch,
        batch_size,
        world_size=1,
        rank=0,
        token_env='HF_TOKEN',
        shuffle_buffer_size=1000,
        max_skip_factor=50,
        schedule='interleave',
    ):
        super().__init__()
        _quiet_http_worker(0)
        if not sources:
            raise ValueError('sources must be a non-empty list')
        schedule = str(schedule).lower()
        if schedule not in ('interleave', 'sequential'):
            raise ValueError(
                f"schedule must be 'interleave' or 'sequential', got {schedule!r}"
            )
        token = os.environ.get(token_env)
        self.transform = transform
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.max_skip_factor = max_skip_factor
        self.schedule = schedule
        self.epoch = 0
        self.sources = [
            _SourceState(spec, token=token, default_shuffle=shuffle_buffer_size)
            for spec in sources
        ]
        self.weights = [max(src.weight, 0.0) for src in self.sources]
        if sum(self.weights) <= 0:
            raise ValueError('At least one stream weight must be > 0')
        # Interleave needs every stream open; sequential builds lazily per source.
        if self.schedule == 'interleave':
            for source in self.sources:
                source.ensure_stream()

    def _worker_plan(self):
        worker_info = get_worker_info()
        worker_count = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0
        sample_count = self.steps_per_epoch * self.batch_size
        samples_per_worker, remainder = divmod(sample_count, worker_count)
        worker_sample_count = samples_per_worker + (worker_id < remainder)
        shard_count = self.world_size * worker_count
        shard_index = self.rank * worker_count + worker_id
        return worker_count, worker_id, worker_sample_count, shard_count, shard_index

    def _sharded_iter(self, source: _SourceState, shard_count: int, shard_index: int):
        stream = source.ensure_stream(seed=_GLOBAL_SEED + self.epoch)
        if hasattr(stream, 'set_epoch'):
            stream.set_epoch(self.epoch)
        if shard_count > 1:
            stream = stream.shard(num_shards=shard_count, index=shard_index)
        return iter(stream)

    def _extract_from_source(self, source: _SourceState, example: Dict[str, Any]):
        return _extract_image(
            example,
            image_mode=source.image_mode,
            image_key=source.image_key,
            fallback_keys=source.fallback_keys,
            url_timeout=source.url_timeout,
            min_size=source.min_size,
            status_key=source.status_key,
            status_allow=source.status_allow,
        )

    def _iter_interleave(self):
        _, worker_id, worker_sample_count, shard_count, shard_index = self._worker_plan()
        rng = random.Random(
            _GLOBAL_SEED + 1009 * self.rank + 17 * worker_id + 31 * self.epoch
        )

        iterators = []
        for source in self.sources:
            iterators.append(self._sharded_iter(source, shard_count, shard_index))

        yielded = 0
        attempts = 0
        max_attempts = max(worker_sample_count * self.max_skip_factor, worker_sample_count)
        while yielded < worker_sample_count and attempts < max_attempts:
            attempts += 1
            source_index = rng.choices(
                range(len(self.sources)), weights=self.weights, k=1
            )[0]
            source = self.sources[source_index]
            try:
                example = next(iterators[source_index])
            except StopIteration:
                iterators[source_index] = self._sharded_iter(
                    source, shard_count, shard_index
                )
                continue

            image = self._extract_from_source(source, example)
            if image is None:
                continue
            yield self.transform(image), 0
            yielded += 1

    def _iter_sequential(self):
        """One HF source at a time; drop previous stream to keep RAM low."""
        _, worker_id, worker_sample_count, shard_count, shard_index = self._worker_plan()
        quotas = _allocate_quotas(worker_sample_count, self.weights)

        order = list(range(len(self.sources)))
        rng = random.Random(_GLOBAL_SEED + 7919 * self.epoch + 104729 * self.rank)
        rng.shuffle(order)

        if worker_id == 0 and self.rank == 0:
            names = [self.sources[i].name for i in order if quotas[i] > 0]
            logger.info(
                'Sequential stream order for epoch %d: %s',
                self.epoch,
                ' -> '.join(names),
            )

        yielded = 0
        for source_index in order:
            quota = quotas[source_index]
            if quota <= 0:
                continue
            source = self.sources[source_index]
            # Drop every other stream so only the active source stays resident.
            for other in self.sources:
                if other is not source:
                    other.drop_stream()
            gc.collect()

            if worker_id == 0 and self.rank == 0:
                logger.info(
                    'Streaming source %s (%d samples this worker)',
                    source.name,
                    quota,
                )

            iterator = self._sharded_iter(source, shard_count, shard_index)
            taken = 0
            attempts = 0
            max_attempts = max(quota * self.max_skip_factor, quota)
            while taken < quota and attempts < max_attempts:
                attempts += 1
                try:
                    example = next(iterator)
                except StopIteration:
                    iterator = self._sharded_iter(source, shard_count, shard_index)
                    continue
                image = self._extract_from_source(source, example)
                if image is None:
                    continue
                yield self.transform(image), 0
                taken += 1
                yielded += 1

            source.drop_stream()
            gc.collect()

        if yielded < worker_sample_count:
            logger.warning(
                'Sequential stream ended early: got %d / %d samples '
                '(rank=%d worker=%d epoch=%d)',
                yielded,
                worker_sample_count,
                self.rank,
                worker_id,
                self.epoch,
            )

    def __iter__(self):
        if self.schedule == 'sequential':
            yield from self._iter_sequential()
        else:
            yield from self._iter_interleave()

    def __len__(self):
        return self.steps_per_epoch * self.batch_size

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        for source in self.sources:
            if source.stream is not None and hasattr(source.stream, 'set_epoch'):
                source.stream.set_epoch(self.epoch)
            # Sequential rebuilds streams per source; clear so next iter is fresh.
            if self.schedule == 'sequential':
                source.drop_stream()


def _make_loader(dataset, collator, batch_size, pin_mem, num_workers):
    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=True,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False,
        prefetch_factor=2 if num_workers and num_workers > 0 else None,
        worker_init_fn=_quiet_http_worker,
    )


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
    image_mode='pil',
    fallback_keys=None,
    url_timeout=5.0,
    min_size=48,
    columns=None,
    status_key=None,
    status_allow=None,
    all_configs=False,
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
        image_mode=image_mode,
        fallback_keys=fallback_keys,
        url_timeout=url_timeout,
        min_size=min_size,
        columns=columns,
        status_key=status_key,
        status_allow=status_allow,
        all_configs=all_configs,
    )
    data_loader = _make_loader(
        dataset, collator, batch_size, pin_mem, num_workers
    )
    return dataset, data_loader, dataset


def make_huggingface_multi_stream(
    transform,
    batch_size,
    collator,
    pin_mem,
    num_workers,
    world_size,
    rank,
    sources,
    steps_per_epoch,
    token_env='HF_TOKEN',
    shuffle_buffer_size=1000,
    max_skip_factor=50,
    schedule='interleave',
):
    dataset = HuggingFaceMultiImageStream(
        sources=sources,
        transform=transform,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        token_env=token_env,
        shuffle_buffer_size=shuffle_buffer_size,
        max_skip_factor=max_skip_factor,
        schedule=schedule,
    )
    data_loader = _make_loader(
        dataset, collator, batch_size, pin_mem, num_workers
    )
    return dataset, data_loader, dataset
