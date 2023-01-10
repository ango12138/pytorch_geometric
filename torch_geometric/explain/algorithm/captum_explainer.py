import logging
from typing import Any, Dict, Optional, Union

import torch
from torch import Tensor

from torch_geometric.explain import Explanation, HeteroExplanation
from torch_geometric.explain.algorithm import ExplainerAlgorithm
from torch_geometric.explain.algorithm.captum import (
    CaptumHeteroModel,
    CaptumModel,
    convert_captum_output,
    to_captum_input,
)
from torch_geometric.explain.config import MaskType
from torch_geometric.typing import EdgeType, NodeType


class CaptumExplainer(ExplainerAlgorithm):
    """A Captum-based explainer for identifying compact subgraph structures and
    node features that play a crucial role in the predictions made by a GNN.

    This explainer algorithm uses `Captum <https://captum.ai/>`_ to compute
    attributions.

    Args:
        attribution_method (Attribution or str): The Captum attribution method
            to use. Can be a string or a :class:`captum.attr` method.
        **kwargs: Additional arguments for the Captum attribution method.
    """
    def __init__(
        self,
        attribution_method: Union[str, Any],
        **kwargs,
    ):
        super().__init__()

        self.attribution_method = attribution_method
        self.kwargs = kwargs

    def _get_mask_type(self):
        "Based on the explainer config, return the mask type."
        node_mask_type = self.explainer_config.node_mask_type
        edge_mask_type = self.explainer_config.edge_mask_type
        if node_mask_type is not None and edge_mask_type is not None:
            mask_type = 'node_and_edge'
        elif node_mask_type is not None:
            mask_type = 'node'
        elif edge_mask_type is not None:
            mask_type = 'edge'
        return mask_type

    def forward(
        self,
        model: torch.nn.Module,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        *,
        target: Optional[Tensor] = None,
        index: Optional[Tensor] = None,
        **kwargs,
    ) -> Explanation:
        import captum.attr  # noqa

        mask_type = self._get_mask_type()

        inputs, add_forward_args = to_captum_input(
            x,
            edge_index,
            mask_type,
            *kwargs.values(),
        )

        if hasattr(model, 'metadata'):
            metadata = model.metadata
            captum_model = CaptumHeteroModel(
                model,
                mask_type,
                index,
                model.metadata,
            )
        else:
            metadata = None
            captum_model = CaptumModel(model, mask_type, index)

        if isinstance(self.attribution_method, str):
            attribution_method = getattr(
                captum.attr,
                self.attribution_method,
            )(captum_model)
        else:
            attribution_method = self.attribution_method(captum_model)

        attributions = attribution_method.attribute(
            inputs=inputs,
            target=target[index],
            additional_forward_args=add_forward_args,
            **self.kwargs,
        )

        node_mask, edge_mask = convert_captum_output(
            attributions,
            mask_type,
            metadata,
        )

        if hasattr(model, 'metadata'):
            return HeteroExplanation(
                node_mask_dict=node_mask,
                edge_mask_dict=edge_mask,
            )
        else:
            return Explanation(node_mask=node_mask, edge_mask=edge_mask)

    def supports(self) -> bool:
        node_mask_type = self.explainer_config.node_mask_type
        if node_mask_type not in [None, MaskType.attributes]:
            logging.error(f"'{self.__class__.__name__}' only supports "
                          f"'node_mask_type' None or 'attributes' "
                          f"(got '{node_mask_type.value}')")
            return False

        return True
