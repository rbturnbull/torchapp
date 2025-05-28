import torch
from torchmetrics import Metric
from sklearn.metrics import f1_score


class AvgSmoothLoss(Metric):
    def __init__(self, beta=0.98, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.beta = beta
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("val", default=torch.tensor(0.), dist_reduce_fx="sum")

    def reset(self):
        self.count = torch.tensor(0)
        self.val = torch.tensor(0.)

    def update(self, loss):
        # Ensure loss is detached to avoid graph-related issues
        loss = loss.detach().float()
        self.count += 1
        self.val = torch.lerp(loss.mean(), self.val, self.beta)

    def compute(self):
        # Return the smoothed loss value
        return self.val / (1 - self.beta**self.count)


def logit_accuracy(predictions, target):
    """
    Gives the accuracy when the output of the network is in logits and the target is binary.

    For example, this can be used with BCEWithLogitsLoss.
    """
    return ((predictions > 0.0) == (target > 0.5)).float().mean()


def logit_f1(logits, target):
    """
    Gives the f1 score when the output of the network is in logits and the target is binary.

    For example, this can be used with BCEWithLogitsLoss.
    """
    predictions = logits > 0.0
    target_binary = target > 0.5
    return f1_score(target_binary.cpu(), predictions.cpu())


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the accuracy of predictions against targets.
    
    Args:
        predictions (torch.Tensor): The predicted values.
        targets (torch.Tensor): The ground truth values.
    
    Returns:
        torch.Tensor: The accuracy as a tensor.
    """
    if predictions.shape != targets.shape:
        predictions = predictions.argmax(dim=-1) if predictions.ndim > 1 else predictions
    
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")
    
    return (predictions == targets).float().mean()
