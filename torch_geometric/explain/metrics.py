from typing import Tuple

import torch
from torch import Tensor, tensor
from torchmetrics import AUROC, ROC

from torch_geometric.explain import Explanation


def get_groundtruth_metrics(
        explanation: Explanation, groundtruth: Explanation
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """accuracy_scores: Compute accuracy scores when
    groundtruth is available

    Args:
        explanation (Explanation): Explanation output generated by an explainer
        groundtruth (Explanation): Groundtruth for explanation
    """

    ex_masks = explanation.masks
    ex_mask_tensor = torch.cat(
        list(map(lambda x: x.view(-1), ex_masks.values())))
    gt_masks = groundtruth.masks
    gt_mask_tensor = torch.cat(
        list(map(lambda x: x.view(-1), gt_masks.values())))
    threshold = 0
    gt_mask_tensor[gt_mask_tensor > threshold] = 1.0
    roc = ROC(task="binary")
    fpr, tpr, thresholds = roc(ex_mask_tensor, gt_mask_tensor)
    auroc = AUROC(task="binary")
    auc = auroc(fpr, tpr)
    ex_mask_tensor[ex_mask_tensor > threshold] = 1.0
    correct_preds = gt_mask_tensor == ex_mask_tensor
    incorrect_preds = gt_mask_tensor != ex_mask_tensor

    tp = torch.sum(gt_mask_tensor[correct_preds])
    tn = torch.sum(correct_preds) - tp
    fp = torch.sum(ex_mask_tensor[incorrect_preds])
    fn = torch.sum(gt_mask_tensor[incorrect_preds])

    if (tp + fp) == 0:
        precision = tensor(0.0)
    else:
        precision = tp / (tp + fp)

    if (tp + fn) == 0:
        recall = tensor(0.0)
    else:
        recall = tp / (tp + fn)

    if precision == 0.0 and recall == 0.0:
        f1_score = tensor(0.0)
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    accuracy = (tp + tn) / (tp + fp + tn + fn)
    return accuracy, recall, precision, f1_score, auc
