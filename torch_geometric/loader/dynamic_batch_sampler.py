from typing import Iterator, List, Optional

import torch

from torch_geometric.data import Dataset


class DynamicBatchSampler(torch.utils.data.sampler.Sampler[List[int]]):
    r"""Dynamically adds samples to a mini-batch up to a maximum size (either
    based on number of nodes or number of edges). When data samples have a
    wide range in sizes, specifying a mini-batch size in terms of number of
    samples is not ideal and can cause CUDA OOM errors.

    Within the :class:`DynamicBatchSampler`, the number of steps per epoch is
    ambiguous, depending on the order of the samples. By default the
    :meth:`__len__` will be undefined. This is fine for most cases but
    progress bars will be infinite. Alternatively, :obj:`num_steps` can be
    supplied to cap the number of mini-batches produced by the sampler.

    **Usage:**

    .. code-block:: python

        from torch_geometric.loader import DataLoader, DynamicBatchSampler

        batch_sampler = DynamicBatchSampler(dataset, max_num=10000)
        loader = DataLoader(dataset, batch_sampler=sampler, ...)

    Args:
        dataset (Dataset): Dataset to sample from.
        max_num (int): size of mini-batch to aim for in number of nodes or
            edges.
        mode (str, optional): :obj:`node` or :obj:`edge` to measure batch
            size. (default: :obj:`node`)
        shuffle (bool, optional): set to :obj:`True` to have the data
            reshuffled at every epoch (default: :obj:`False`).
        skip_too_big (bool, optional): set to :obj:`True` to skip samples
            which can't fit in a batch by itself. (default: :obj:`False`).
        num_steps (int, optional): The number of mini-batches to draw for a
            single epoch. If set to :obj:`None`, will iterate through all the
            underlying data, but :meth:`__len__` will be :obj:`None` since it
            will be ambiguous. (default: :obj:`None`)
    """
    def __init__(self, dataset: Dataset, max_num: int, mode: str = 'node',
                 shuffle: bool = False, skip_too_big: bool = False,
                 num_steps: Optional[int] = None):
        if not isinstance(max_num, int) or max_num <= 0:
            raise ValueError("`max_num` should be a positive integer value "
                             "(got max_num={max_num}).")
        if mode not in ['node', 'edge']:
            raise ValueError("`mode` choice should be either "
                             f"`node` or `edge` (got {mode}).")

        self.max_num = max_num
        self.dataset = dataset
        self.shuffle = shuffle
        self.skip_too_big = skip_too_big
        if num_steps is None:
            num_steps = len(dataset)
        self.num_steps = num_steps

        self.mode = mode

    def __iter__(self) -> Iterator[List[int]]:
        batch = []
        batch_n = 0
        num_steps = 0
        num_processed = 0

        if self.shuffle:
            indices = torch.randperm(len(self.dataset), dtype=torch.long)
        else:
            indices = torch.arange(len(self.dataset), dtype=torch.long)

        # Main iteration loop
        while (num_processed < len(self.dataset)
               and num_steps < self.num_steps):
            # Fill batch
            for idx in indices[num_processed:]:
                # Size of sample
                if self.mode == 'node':
                    n = self.dataset[idx].num_nodes
                else:
                    n = self.dataset[idx].num_edges

                if batch_n + n > self.max_num:
                    if batch_n == 0:
                        if self.skip_too_big:
                            continue
                        else:
                            raise RuntimeError(
                                f"Size of a single data example ({n} @ index "
                                "{idx}) is larger than max_num"
                                "({self.max_num})")

                    # Mini-batch filled
                    break

                # Add sample to current batch
                batch.append(idx.item())
                num_processed += 1
                batch_n += n

            yield batch
            batch = []
            batch_n = 0
            num_steps += 1

    def __len__(self) -> int:
        return self.num_steps
