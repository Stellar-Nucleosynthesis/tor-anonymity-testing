import numpy as np
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    confusion_matrix
)
from typing import Dict


def compute_basic_metrics(y_true: np.ndarray,
                          y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute basic classification metrics.

    Args:
        y_true: True labels (1 = match, 0 = no match)
        y_pred: Predicted labels

    Returns:
        Dictionary of metrics
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    metrics = {
        'true_positive_rate': tpr,
        'false_positive_rate': fpr,
        'precision': precision,
        'recall': tpr,
        'specificity': specificity,
        'f1_score': f1,
        'accuracy': accuracy,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }

    return metrics


def compute_roc_metrics(y_true: np.ndarray,
                        y_scores: np.ndarray) -> Dict[str, any]:
    """
    Compute ROC curve and AUC.

    Args:
        y_true: True labels
        y_scores: Prediction scores (not binary)

    Returns:
        Dictionary with ROC data and AUC
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    return {
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }


def compute_precision_recall(y_true: np.ndarray,
                             y_scores: np.ndarray) -> Dict[str, any]:
    """
    Compute precision-recall curve.

    Args:
        y_true: True labels
        y_scores: Prediction scores

    Returns:
        Dictionary with PR curve data
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    return {
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall,
        'thresholds': thresholds
    }