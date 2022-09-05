from sklearn.metrics import f1_score


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
