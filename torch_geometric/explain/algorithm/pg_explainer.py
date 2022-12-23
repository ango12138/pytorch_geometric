import logging
from typing import Optional

import torch
from torch import Tensor, nn
from torch_scatter import scatter_mean
from tqdm import tqdm

from torch_geometric.explain import Explanation
from torch_geometric.explain.algorithm.utils import (
    clear_masks,
    drop_isolated_nodes,
    set_masks,
)
from torch_geometric.explain.config import MaskType, ModelMode, ModelTaskLevel
from torch_geometric.utils import (
    get_message_passing_embeddings,
    k_hop_subgraph,
)

from .base import ExplainerAlgorithm


# TODO add example.
class PGExplainer(ExplainerAlgorithm):
    r"""The PGExplainer model from the `"Parameterized Explainer for Graph
    Neural Network" <https://arxiv.org/abs/2011.04573>`_ paper. It
    uses a neural network :obj:`explainer_model` , to predict which
    edges are crucial to a GNNs node or graph prediction.

    Args:
        epochs (int, optional): The number of epochs to train.
            (default: :obj:`30`).
        lr (float, optional): The learning rate to apply.
            (default: :obj:`0.003`).
        log (bool, optional): If set to :obj:`False`, will not log any learning
            progress. (default: :obj:`True`)
        **kwargs (optional): Additional hyper-parameters to override default
            settings in
            :attr:`~torch_geometric.explain.algorithm.PGExplainer.coeffs`.
    """

    coeffs = {
        'edge_size': 0.05,
        'edge_ent': 1.0,
        'temp': [5.0, 2.0],
        'bias': 0
    }

    def __init__(self, epochs: int = 30, lr: float = 0.003, log: bool = True,
                 **kwargs):
        super().__init__(training_needed=True)
        self.epochs = epochs
        self.lr = lr
        self.log = log
        self.coeffs.update(kwargs)

    def supports(self) -> bool:
        task_level = self.model_config.task_level
        if task_level not in [ModelTaskLevel.node, ModelTaskLevel.graph]:
            logging.error(f"Task level '{task_level.value}' not supported")
            return False

        edge_mask_type = self.explainer_config.edge_mask_type
        if edge_mask_type not in [MaskType.object]:
            logging.error(f"Edge mask type '{edge_mask_type}' not "
                          f"supported")
            return False

        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type is not None:
            logging.error(
                "Node mask not supported. Set 'node_mask_type' to 'None'")
            return False

        return True

    def subgraph(self, node_idx: int, x: Tensor, edge_index: Tensor, model,
                 **kwargs):
        r"""Returns the subgraph of the given node.
        Args:
            node_idx (int): The node to explain.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.
        :rtype: (Tensor, Tensor, LongTensor, LongTensor, LongTensor, dict)
        """
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, ExplainerAlgorithm._num_hops(model), edge_index,
            relabel_nodes=True, num_nodes=num_nodes,
            flow=ExplainerAlgorithm._flow(model))

        x = x[subset]
        kwargs_new = {}
        for key, value in kwargs.items():
            if torch.is_tensor(value) and value.size(0) == num_nodes:
                kwargs_new[key] = value[subset]
            elif torch.is_tensor(value) and value.size(0) == num_edges:
                kwargs_new[key] = value[edge_mask]
            else:
                kwargs_new[key] = value
        return x, edge_index, mapping, edge_mask, subset, kwargs_new

    def train_explainer(self, model: torch.nn.Module, x: Tensor,
                        edge_index: Tensor, target: Tensor = None,
                        index: Tensor = None, **kwargs):
        # Initialize trainable parameters.
        z = get_message_passing_embeddings(model, x=x, edge_index=edge_index,
                                           **kwargs)[-1]
        out_channels = z.shape[-1]
        task_level = self.model_config.task_level
        self.exp_in_channels = 2 * out_channels if (
            task_level == ModelTaskLevel.graph) else 3 * out_channels
        self.explainer_model = nn.Sequential(
            nn.Linear(self.exp_in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        assert x.shape[0] == z.shape[0]

        model_state = model.training
        model.eval()
        clear_masks(model)
        self.to(x.device)
        optimizer = torch.optim.Adam(self.explainer_model.parameters(),
                                     lr=self.lr)
        self.explainer_model.train()

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description('Training PG Explainer')

        # train explainer_model
        bias = self.coeffs['bias']
        task_level = self.model_config.task_level
        if task_level == ModelTaskLevel.graph:
            batch = kwargs.get('batch', None)
            if batch is None:
                # Assume all nodes belong to the same graph.
                batch = torch.zeros(x.shape[0], dtype=torch.long,
                                    device=x.device)
            assert x.shape[0] == batch.shape[0]
            assert batch.unique().shape[0] == target.shape[0]
            batch = batch.squeeze()
            explainer_in = self._create_explainer_input(edge_index, z).detach()

            for e in range(0, self.epochs):
                optimizer.zero_grad()
                t = self._get_temp(e)
                self.edge_mask = self._compute_edge_mask(
                    self.explainer_model(explainer_in), t, bias=bias)
                set_masks(model, self.edge_mask, edge_index)
                out = model(x=x, edge_index=edge_index, **kwargs)
                self._loss(out, target.squeeze(), batch=batch,
                           edge_index=edge_index).backward()
                optimizer.step()
                if self.log:  # pragma: no cover
                    pbar.update(1)

        else:
            index = index if index is not None else torch.arange(x.shape[0])
            assert index.unique().shape[0] == index.shape[0]
            index = drop_isolated_nodes(index, edge_index)
            for e in range(0, self.epochs):
                loss = torch.tensor([0.0], device=x.device).detach()
                t = self._get_temp(e)
                optimizer.zero_grad()

                for n in index:
                    n = int(n)
                    kwargs['z'] = z
                    (x_n, edge_index_n, mapping, _, _,
                     kwargs_n) = self.subgraph(n, x, edge_index, model,
                                               **kwargs)
                    z_n = kwargs_n.pop('z')
                    explainer_in = self._create_explainer_input(
                        edge_index_n, z_n, mapping).detach()
                    self.edge_mask = self._compute_edge_mask(
                        self.explainer_model(explainer_in), t, bias=bias)

                    set_masks(model, self.edge_mask, edge_index_n)
                    out = model(x=x_n, edge_index=edge_index_n, **kwargs_n)
                    loss += self._loss(out[mapping], target[n])
                    clear_masks(model)

                assert not torch.isnan(loss)
                loss.backward()
                optimizer.step()

                if self.log:  # pragma: no cover
                    pbar.update(1)

        if self.log:
            pbar.close()
        clear_masks(model)
        model.train(model_state)
        self.training_needed = False

    def _get_temp(self, e: int) -> float:
        temp = self.coeffs['temp']
        return temp[0] * ((temp[1] / temp[0])**(e / self.epochs))

    def _create_explainer_input(self, edge_index, z, node_idx=None) -> Tensor:
        rows, cols = edge_index
        z_j, z_i = z[rows], z[cols]
        task_level = self.model_config.task_level
        if task_level == ModelTaskLevel.node:
            z_node = z[node_idx].repeat(rows.size(0), 1)
            return torch.cat([z_i, z_j, z_node], 1)
        else:
            return torch.cat([z_i, z_j], 1)

    def _compute_edge_mask(self, edge_weight, temperature=1.0, bias=0.0,
                           training=True):

        if training:  # noise is added to edge_weight.
            bias += 0.0001
            eps = (2 * bias - 1) * torch.rand(edge_weight.size()) + (1 - bias)
            eps = eps.to(edge_weight.device)
            return (eps.log() -
                    (1 - eps).log() + edge_weight).squeeze(dim=1) / temperature

        else:
            return edge_weight.squeeze(dim=1)

    def _loss(self, y_hat, y, batch=None, edge_index=None):
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)

        mask = self.edge_mask.sigmoid().squeeze()

        # Regularization losses
        size_loss = mask.sum() * self.coeffs['edge_size']
        mask_ent_reg = -mask * mask.log() - (1 - mask) * (1 - mask).log()
        mask_ent_loss = mask_ent_reg.mean() if batch is None else scatter_mean(
            mask_ent_reg, batch[edge_index[0]]).sum()
        mask_ent_loss *= self.coeffs['edge_ent']

        return loss + size_loss + mask_ent_loss

    def forward(self, model: torch.nn.Module, x: Tensor, edge_index: Tensor, *,
                target=None, index: Optional[int] = None,
                **kwargs) -> Explanation:
        r"""Returns an :obj:`edge_mask` that explains :obj:`model` prediction.

        Args:
            model(torch.nn.Module):
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            index (Optional, int): The node id to explain.
                Only required if :obj:`task` is :obj:`"node"`.
            **kwargs (optional): Additional arguments passed to the GNN module.
        :rtype: :class:`Tensor`
        """
        assert self.training_needed is False
        z = get_message_passing_embeddings(model, x=x, edge_index=edge_index,
                                           **kwargs)[-1]
        if self.model_config.task_level == ModelTaskLevel.graph:
            explainer_in = self._create_explainer_input(edge_index, z)
            edge_mask = self._compute_edge_mask(
                self.explainer_model(explainer_in), training=False)
            edge_mask = edge_mask.sigmoid()

        else:
            if not isinstance(index, int):
                raise ValueError('Only one node can be explained at a time,'
                                 f'got(index = {index})')
            # We need to compute hard masks to properly clean up edges and
            # nodes attributions not involved during message passing:
            _, hard_edge_mask = self._get_hard_masks(model, index, edge_index,
                                                     num_nodes=x.size(0))
            explainer_in = self._create_explainer_input(edge_index, z, index)
            edge_mask = self._compute_edge_mask(
                self.explainer_model(explainer_in), training=False)
            edge_mask = self._post_process_mask(edge_mask, hard_edge_mask,
                                                apply_sigmoid=True)

        return Explanation(edge_mask=edge_mask, x=x, edge_index=edge_index)

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'epochs={self.epochs}, lr={self.lr}')
