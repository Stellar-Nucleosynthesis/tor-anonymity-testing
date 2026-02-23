import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional
from pathlib import Path

def plot_roc_curve(
        fpr: np.ndarray,
        tpr: np.ndarray,
        auc_score: float,
        scenario_name: str = "Attack Scenario",
        save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot ROC curve.

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        scenario_name: Name for the plot
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(fpr, tpr, 'b-', linewidth=2,
            label=f'{scenario_name} (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {scenario_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_multiple_roc_curves(
        scenario_results: Dict[str, Dict],
        save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot ROC curves for multiple scenarios.

    Args:
        scenario_results: Dict mapping scenario names to their metrics
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 9))

    cmap = plt.colormaps.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, len(scenario_results)))

    for (scenario_name, metrics), color in zip(scenario_results.items(), colors):
        if 'fpr' in metrics and 'tpr' in metrics:
            ax.plot(metrics['fpr'], metrics['tpr'],
                    linewidth=2, color=color,
                    label=f"{scenario_name} (AUC={metrics.get('roc_auc', 0):.3f})")

    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig