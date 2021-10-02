from typing import Union
from torch_geometric.typing import PairOptTensor, PairTensor, Adj

from torch import Tensor
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

from torch.nn import Linear
from torch_geometric.utils import softmax

from ..inits import reset


class PointTransformerConv(MessagePassing):
    r"""The PointTransformer layer from the `"Point Transformer"
    <https://arxiv.org/abs/2012.09164>` paper

    .. math::
        \mathbf{x}^{\prime}_i =  \sum_{j \in
        \mathcal{N}(i) \cup \{ i \}} \rho \left( \gamma \left( \phi
        \left( \mathbf{x}_i \right) - \psi \left( \mathbf{x}_j \right) \right)
        + \delta  \right) \odot \left( \alpha \left(  \mathbf{x}_j \right)
        + \delta \right) ,

    where :math:`\gamma` and :math:`\delta = \theta \left( \mathbf{p}_j -
    \mathbf{p}_i \right)` denote neural networks, *.i.e.* MLPs, that
    respectively map transformed node features and relative spatial
    coordinates. :math:`\mathbf{P} \in \mathbb{R}^{N \times D}`
    defines the position of each point.

    Args:
        in_channels (int or tuple): Size of each input sample features
        out_channels (int) : size of each output sample features
        local_nn : (torch.nn.Module, optional): A neural network
            :math:`\delta = \theta \left( \mathbf{p}_j -
            \mathbf{p}_i \right)` which maps relative spatial coordinates
            :obj:`pos_j - pos_i` of shape
            :obj:`[-1, 3]` to shape
            :obj:`[-1, out_channels]`, *e.g.*, defined by
            :class:`torch.nn.Sequential`. Will be defaulted to a linear
            mapping if not specified. (default: :obj:`None`)
        attn_nn : (torch.nn.Module, optional): A neural network
            :math:`\gamma` which maps transformed node features of shape
            :obj:`[-1, out_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
            (default: :obj:`None`)
        add_self_loops (bool, optional) : If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels, out_channels, local_nn=None, attn_nn=None,
                 add_self_loops=True):

        super(PointTransformerConv,
              self).__init__(aggr='add')  # "Add" aggregation

        self.in_channels = in_channels
        self.out_channels = out_channels

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if local_nn is None:
            local_nn = Linear(3, out_channels)

        self.local_nn = local_nn
        self.attn_nn = attn_nn
        self.lin = Linear(in_channels[0], out_channels)
        self.lin_src = Linear(in_channels[0], out_channels)
        self.lin_dst = Linear(in_channels[1], out_channels)

        self.add_self_loops = add_self_loops

    def reset_parameters(self):
        reset(self.local_nn)
        reset(self.attn_nn)
        reset(self.lin)
        reset(self.lin_src)
        reset(self.lin_dst)

    def forward(self, x: Union[Tensor, PairTensor],
                pos: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        """"""

        if not isinstance(x, tuple):
            x: PairOptTensor = (x, x)

        if isinstance(pos, Tensor):
            pos: PairTensor = (pos, pos)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index,
                                               num_nodes=pos[1].size(0))
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        return self.propagate(edge_index, x=x, pos=pos)

    def message(self, x_i, x_j, pos_i, pos_j, index):

        # compute position encoding
        out_delta = self.local_nn((pos_j - pos_i))
        alpha = self.lin_dst(x_i) - self.lin_src(x_j) + out_delta
        if self.attn_nn is not None:
            alpha = self.attn_nn(alpha)
        alpha = softmax(alpha, index)

        msg = alpha * (self.lin(x_j) + out_delta)

        return msg

    def __repr__(self):
        return '{}({}, {}, local_nn={}, attn_nn={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.local_nn, self.attn_nn)
