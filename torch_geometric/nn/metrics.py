from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import torch
from torch import Tensor

from torch_geometric.utils import scatter

try:
    import torchmetrics  # noqa
    WITH_TORCHMETRICS = True
    BaseMetric = torchmetrics.Metric
except Exception:
    WITH_TORCHMETRICS = False
    BaseMetric = torch.nn.Module


class LinkPredMetric(BaseMetric, ABC):
    r"""An abstract class for computing link prediction retrieval metrics.

    Args:
        top_k (int): The number of top-:math:`k` predictions to evaluate
            against.
    """
    is_differentiable: Optional[bool] = None
    higher_is_better: Optional[bool] = None
    full_state_update: Optional[bool] = None

    def __init__(self, top_k: int):
        super().__init__()

        if top_k <= 0:
            raise ValueError(f"'top_k' needs to be a positive integer in "
                             f"'{self.__class__.__name__}' (got {top_k})")

        self.top_k = top_k

        if WITH_TORCHMETRICS:
            self.add_state('accum', torch.tensor(0.), dist_reduce_fx='sum')
            self.add_state('total', torch.tensor(0), dist_reduce_fx='sum')
        else:
            self.register_buffer('accum', torch.tensor(0.))
            self.register_buffer('total', torch.tensor(0))

    def update(
        self,
        pred_index_mat: Tensor,
        edge_label_index: Union[Tensor, Tuple[Tensor, Tensor]],
    ):
        r"""Updates the state variables based on the current mini-batch
        prediction.

        Args:
            pred_index_mat (torch.Tensor): The top-:math:`k` predictions of
                every example in the mini-batch with shape
                :obj:`[batch_size, top_k]`.
            edge_label_index (torch.Tensor): The ground-truth indices for every
                example in the mini-batch, given in COO format of shape
                :obj:`[2, num_ground_truth_indices]`.
        """
        if pred_index_mat.size(1) != self.top_k:
            raise ValueError(f"Expected 'pred_index_mat' to hold {self.top_k} "
                             f"many indices for every entry "
                             f"(got {pred_index_mat.size(1)})")

        # Compute a boolean matrix indicating if the k-th prediction is part of
        # the ground-truth:
        max_index = max(
            pred_index_mat.max() if pred_index_mat.numel() > 0 else 0,
            edge_label_index[1].max()
            if edge_label_index[1].numel() > 0 else 0,
        )
        arange = torch.arange(
            start=0,
            end=max_index * pred_index_mat.size(0),
            step=max_index,
            device=pred_index_mat.device,
        ).view(-1, 1)
        flat_pred_index = (pred_index_mat + arange).view(-1)
        flat_y_index = max_index * edge_label_index[0] + edge_label_index[1]

        pred_isin_mat = torch.isin(flat_pred_index, flat_y_index)
        pred_isin_mat = pred_isin_mat.view(pred_index_mat.size())

        # Compute the number of targets per example:
        y_count = scatter(
            torch.ones_like(edge_label_index[0]),
            edge_label_index[0],
            dim=0,
            dim_size=pred_index_mat.size(0),
            reduce='sum',
        )

        metric = self._compute(pred_isin_mat, y_count)

        self.accum += metric.sum()
        self.total += (y_count > 0).sum()

    def compute(self) -> Tensor:
        r"""Computes the final metric value."""
        if self.total == 0:
            return torch.zeros_like(self.accum)
        return self.accum / self.total

    def reset(self) -> 'LinkPredMetric':
        r"""Reset metric state variables to their default value."""
        if WITH_TORCHMETRICS:
            super().reset()
        else:
            self.accum.zero_()
            self.total.zero_()

        return self

    @abstractmethod
    def _compute(self, pred_isin_mat: Tensor, y_count: Tensor) -> Tensor:
        r"""Compute the specific metric.
        To be implemented separately for each metric class.

        Args:
            pred_isin_mat (torch.Tensor): A boolean matrix whose :obj:`(i,k)`
                element indicates if the :obj:`k`-th prediction for the
                :obj:`i`-th example is correct or not.
            y_count (torch.Tensor): A vector indicating the number of
                ground-truth labels for each example.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.top_k})'


class LinkPredPrecision(LinkPredMetric):
    r"""A link prediction metric to compute Precision@:math`k`.

    Args:
        top_k (int): The number of top-:math:`k` predictions to evaluate
            against.
    """
    def _compute(self, pred_isin_mat: Tensor, y_count: Tensor) -> Tensor:
        return pred_isin_mat.sum(dim=-1) / self.top_k
