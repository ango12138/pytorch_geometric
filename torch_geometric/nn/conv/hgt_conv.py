import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, ones, reset
from torch_geometric.nn.module_dict import ModuleDict
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import EdgeType, Metadata, NodeType, pyg_lib
from torch_geometric.utils import softmax


def pad_list(xs: List[Tensor], dims: Tensor) -> List[Tensor]:
    max_size = dims.max()
    for i, x in enumerate(xs):
        if x.shape[1] < max_size:
            xs[i] = torch.concat(
                (x, torch.zeros(x.shape[0], max_size - x.shape[1])), dim=1)
    return xs


def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    elif aggr is None:
        return torch.stack(xs, dim=1)
    elif len(xs) == 1:
        return xs[0]
    else:
        out = torch.stack(xs, dim=0)
        out = getattr(torch, aggr)(out, dim=0)
        out = out[0] if isinstance(out, tuple) else out
        return out


class HGTConv(MessagePassing):
    r"""The Heterogeneous Graph Transformer (HGT) operator from the
    `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
    paper.

    .. note::

        For an example of using HGT, see `examples/hetero/hgt_dblp.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        hetero/hgt_dblp.py>`_.

    Args:
        in_channels (int or Dict[str, int]): Size of each input sample of every
            node type, or :obj:`-1` to derive the size from the first input(s)
            to the forward method.
        out_channels (int): Size of each output sample.
        metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
            of the heterogeneous graph, *i.e.* its node and edge types given
            by a list of strings and a list of string triplets, respectively.
            See :meth:`torch_geometric.data.HeteroData.metadata` for more
            information.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        group (string, optional): The aggregation scheme to use for grouping
            node embeddings generated by different relations.
            (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
            (default: :obj:`"sum"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        group: str = "sum",
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(f"'out_channels' (got {out_channels}) must be "
                             f"divisible by the number of heads (got {heads})")

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}
        # can only use grouped matmul if torch >= 1.14
        major_vers, minor_vers = str(torch.__version__).split('.')[:2]
        self.use_gmm = int(major_vers) >= 2 or int(minor_vers) >= 14
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.group = group

        self.k_lin = ModuleDict()
        self.q_lin = ModuleDict()
        self.v_lin = ModuleDict()
        self.a_lin = ModuleDict()
        self.skip = ParameterDict()
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.dims = torch.tensor(list(self.in_channels.values()))
        self.max_channels = self.dims.max()
        self.infer_shapes = self.max_channels == -1
        if self.use_gmm:
            # grouped gemm allows us not to have to pad
            for node_type, in_channels in self.in_channels.items():
                self.k_lin[node_type] = Linear(in_channels, out_channels)
                self.q_lin[node_type] = Linear(in_channels, out_channels)
                self.v_lin[node_type] = Linear(in_channels, out_channels)
                self.a_lin[node_type] = Linear(out_channels, out_channels)
                self.skip[node_type] = Parameter(torch.Tensor(1))
        else:
            # need to pad xs to concatenate them
            self.no_pad = (self.dims == self.max_channels).all()
            for node_type in self.node_types:
                self.k_lin[node_type] = Linear(self.max_channels, out_channels)
                self.q_lin[node_type] = Linear(self.max_channels, out_channels)
                self.v_lin[node_type] = Linear(self.max_channels, out_channels)
                self.a_lin[node_type] = Linear(self.max_channels, out_channels)
                self.skip[node_type] = Parameter(torch.Tensor(1))

        self.a_rel = ParameterDict()
        self.m_rel = ParameterDict()
        self.p_rel = ParameterDict()
        dim = out_channels // heads
        for edge_type in metadata[1]:
            edge_type = '__'.join(edge_type)
            self.a_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.m_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
            self.p_rel[edge_type] = Parameter(torch.Tensor(heads))

        self.reset_parameters()
        major_vers, minor_vers = str(torch.__version__).split('.')[:2]

    def reset_parameters(self):
        reset(self.k_lin)
        reset(self.q_lin)
        reset(self.v_lin)
        reset(self.a_lin)
        ones(self.skip)
        ones(self.p_rel)
        glorot(self.a_rel)
        glorot(self.m_rel)

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Union[Dict[EdgeType, Tensor],
                               Dict[EdgeType, SparseTensor]]  # Support both.
    ) -> Dict[NodeType, Optional[Tensor]]:
        r"""
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding input node
                features  for each individual node type.
            edge_index_dict (Dict[str, Union[Tensor, SparseTensor]]): A
                dictionary holding graph connectivity information for each
                individual edge type, either as a :obj:`torch.LongTensor` of
                shape :obj:`[2, num_edges]` or a
                :obj:`torch_sparse.SparseTensor`.

        :rtype: :obj:`Dict[str, Optional[Tensor]]` - The output node embeddings
            for each node type.
            In case a node type does not receive any message, its output will
            be set to :obj:`None`.
        """
        xs = list(x_dict.values())


        if not self.use_gmm:
            if self.no_pad and not self.infer_shapes:
                x = torch.cat(xs)
            else:
                if self.infer_shapes:
                    self.dims = torch.tensor([x.shape[-1] for x in xs])
                    #initialize lazy params
                    max_channels = self.dims.max()
                    for node_type, u_k_lin in self.k_lin.items():
                        self.k_lin[node_type] = Linear(max_channels, self.out_channels)
                    reset(self.k_lin)
                    for node_type, u_q_lin in self.q_lin.items():
                        self.q_lin[node_type] = Linear(max_channels, self.out_channels)
                    reset(self.q_lin)
                    for node_type, u_v_lin in self.v_lin.items():
                        self.v_lin[node_type] = Linear(max_channels, self.out_channels)
                    reset(self.v_lin)
                    self.infer_shapes = False
                x = torch.cat(pad_list(xs, self.dims))
        elif self.infer_shapes:
            self.dims = {node_type:x.shape[-1] for node_type, x in x_dict.items()}
            #initialize lazy params
            for node_type, dim in self.dims.items():
                self.k_lin[node_type] = Linear(dim, self.out_channels)
                self.q_lin[node_type] = Linear(dim, self.out_channels)
                self.v_lin[node_type] = Linear(dim, self.out_channels)
            reset(self.k_lin)
            reset(self.q_lin)
            reset(self.v_lin)
            self.infer_shapes = False



        H, D = self.heads, self.out_channels // self.heads

        k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}
        # parralelize over node-types
        k_wts = [
            self.k_lin[node_type].weight.T for node_type in self.node_types
        ]
        k_biases = [
            self.k_lin[node_type].bias for node_type in self.node_types
        ]
        q_wts = [
            self.q_lin[node_type].weight.T for node_type in self.node_types
        ]
        q_biases = [
            self.q_lin[node_type].bias for node_type in self.node_types
        ]
        v_wts = [
            self.v_lin[node_type].weight.T for node_type in self.node_types
        ]
        v_biases = [
            self.v_lin[node_type].bias for node_type in self.node_types
        ]
        out_dict = {node_type: [] for node_type in self.node_types}

        if not self.use_gmm:
            ptr = [0]
            count = 0
            for x_type_i in xs:
                count += x_type_i.size(0)
                ptr.append(count)
            ptr = torch.tensor(ptr).to(x.device)
            k_wt = torch.stack(k_wts)
            k_bias = torch.stack(k_biases)
            q_wt = torch.stack(q_wts)
            q_bias = torch.stack(q_biases)
            v_wt = torch.stack(v_wts)
            v_bias = torch.stack(v_biases)

        # compute K, Q, V over node-types
        if self.use_gmm:
            # compute K
            k_list = pyg_lib.ops.grouped_matmul(inputs=xs, others=k_wts,
                                                biases=k_biases)
            k_dict = {
                node_type: k_list[i].view(-1, H, D)
                for i, node_type in enumerate(self.node_types)
            }
            # compute Q
            q_list = pyg_lib.ops.grouped_matmul(inputs=xs, others=q_wts,
                                                biases=q_biases)
            q_dict = {
                node_type: q_list[i].view(-1, H, D)
                for i, node_type in enumerate(self.node_types)
            }
            # compute V
            v_list = pyg_lib.ops.grouped_matmul(inputs=xs, others=v_wts,
                                                biases=v_biases)
            v_dict = {
                node_type: v_list[i].view(-1, H, D)
                for i, node_type in enumerate(self.node_types)
            }
        else:
            k = pyg_lib.ops.segment_matmul(inputs=x, ptr=ptr, other=k_wt,
                                           bias=k_bias)
            k_dict = {
                node_type: k[ptr[i]:ptr[i + 1]].view(-1, H, D)
                for i, node_type in enumerate(self.node_types)
            }
            q = pyg_lib.ops.segment_matmul(inputs=x, ptr=ptr, other=q_wt,
                                           bias=q_bias)
            q_dict = {
                node_type: q[ptr[i]:ptr[i + 1]].view(-1, H, D)
                for i, node_type in enumerate(self.node_types)
            }
            v = pyg_lib.ops.segment_matmul(inputs=x, ptr=ptr, other=v_wt,
                                           bias=vp_bias)
            v_dict = {
                node_type: v[ptr[i]:ptr[i + 1]].view(-1, H, D)
                for i, node_type in enumerate(self.node_types)
            }

        # parallelize over edge-types
        src_types = [edge_type[0] for edge_type in self.edge_types]
        a_rels = [
            self.a_rel['__'.join(edge_type)] for edge_type in self.edge_types
        ]
        m_rels = [
            self.m_rel['__'.join(edge_type)] for edge_type in self.edge_types
        ]

        if self.use_gmm:
            k_ins = [
                k_dict[src_type].transpose(0, 1) for src_type in src_types
            ]
            v_ins = [
                v_dict[src_type].transpose(0, 1) for src_type in src_types
            ]
            print([k.shape for k in k_ins])
            print([a.shape for a in a_rels])
            k_outs = [
                k_o_i.transpose(1, 0)
                for k_o_i in pyg_lib.ops.grouped_matmul(k_ins, a_rels)
            ]
            v_outs = [
                v_o_i.transpose(1, 0)
                for v_o_i in pyg_lib.ops.grouped_matmul(v_ins, m_rels)
            ]
            increment_dict = {
                src_type: k_outs[i].shape[0]
                for i, src_type in enumerate(src_types)
            }
            k_out = torch.cat(k_outs)
            v_out = torch.cat(v_outs)
        else:
            k_ins = [k_dict[src_type] for src_type in src_types]
            v_ins = [v_dict[src_type] for src_type in src_types]
            a_rel, m_rel = torch.cat(a_rels), torch.cat(m_rels)
            trans_ptr = [0]
            count = 0
            for k_i in k_ins:
                count += k_i.size(0)
                trans_ptr.append(count)
            trans_ptr = torch.tensor(trans_ptr)
            k_out = pyg_lib.ops.segment_matmul(
                torch.cat(k_ins).transpose(0, 1), trans_ptr,
                a_rel).transpose(1, 0)
            v_out = pyg_lib.ops.segment_matmul(
                torch.cat(v_ins).transpose(0, 1), trans_ptr,
                m_rel).transpose(1, 0)
            increment_dict = {
                src_type: k_out[trans_ptr[i]:trans_ptr[i + 1]].shape[0]
                for i, src_type in enumerate(src_types)
            }

        q_list = []
        p_rels = []
        for e_type in self.edge_types:
            src_type, dst_type = e_type[0], e_type[-1]
            if torch.numel(edge_index_dict[e_type]) != 0:
                edge_index_dict[e_type][0, :] = edge_index_dict[e_type][
                    0, :] + increment_dict[src_type]
                edge_index_dict[e_type][1, :] = edge_index_dict[e_type][
                    1, :] + increment_dict[dst_type]
                q_list.append(q_dict[dst_type])
                p_rels.append(self.p_rel[edge_type])
        q = torch.cat(q_list)
        p = torch.cat(p_rels)
        e_idx = torch.cat(list(edge_index_dict.values()), dim=1)
        out = self.propagate(e_idx, k=k_out, q=q, v=v_out, rel=p, size=None)
        for e_type in enumerate(self.edge_types):
            dst_type = e_type[-1]
            dst_n_ids = edge_index_dict[edge_type][1, :]
            out_dict[dst_type].append(out[dst_n_ids])

        # Iterate over node-types:
        for node_type, outs in out_dict.items():
            out = group(outs, self.group)

            if out is None:
                out_dict[node_type] = None
                continue

            out = self.a_lin[node_type](F.gelu(out))
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, rel: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:

        alpha = (q_i * k_j).sum(dim=-1) * rel
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
                f'heads={self.heads})')
