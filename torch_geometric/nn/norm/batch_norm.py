import torch
from torch import Tensor


class BatchNorm(torch.nn.Module):
    r"""Applies batch normalization over a batch of node features as described
    in the `"Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift" <https://arxiv.org/abs/1502.03167>`_
    paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{\mathbf{x} -
        \textrm{E}[\mathbf{x}]}{\sqrt{\textrm{Var}[\mathbf{x}] + \epsilon}}
        \odot \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over all nodes
    inside the mini-batch.

    Args:
        in_channels (int): Size of each input sample.
        eps (float, optional): A value added to the denominator for numerical
            stability. (default: :obj:`1e-5`)
        momentum (float, optional): The value used for the running mean and
            running variance computation. (default: :obj:`0.1`)
        affine (bool, optional): If set to :obj:`True`, this module has
            learnable affine parameters :math:`\gamma` and :math:`\beta`.
            (default: :obj:`True`)
        track_running_stats (bool, optional): If set to :obj:`True`, this
            module tracks the running mean and variance, and when set to
            :obj:`False`, this module does not track such statistics and always
            uses batch statistics in both training and eval modes.
            (default: :obj:`True`)
        allow_single_element (bool, optional): If set to :obj:`True`, batches
            with only a single element will work as though in training mode.
            That is the running mean and variance will be used.
            (default: :obj:`False`)
    """
    def __init__(self, in_channels: int, eps: float = 1e-5,
                 momentum: float = 0.1, affine: bool = True,
                 track_running_stats: bool = True,
                 allow_single_element: bool = False):
        super().__init__()
        self.module = torch.nn.BatchNorm1d(in_channels, eps, momentum, affine,
                                           track_running_stats)
        self.in_channels = in_channels
        self.allow_single_element = allow_single_element

    def reset_parameters(self):
        self.module.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """"""
        if self.allow_single_element and x.size(0) <= 1:
            training = self.module.training
            running_mean = self.module.running_mean
            running_var = self.module.running_var
            if running_mean is None or running_var is None:
                self.module.running_var = torch.ones(self.in_channels,
                                                     device=x.device)
                self.module.running_mean = torch.zeros(self.in_channels,
                                                       device=x.device)
            self.module.eval()
            out = self.module(x)
            self.module.training = training
            self.module.running_mean = running_mean
            self.module.running_var = running_var
            return out
        return self.module(x)

    def __repr__(self):
        return f'{self.__class__.__name__}({self.module.num_features})'
