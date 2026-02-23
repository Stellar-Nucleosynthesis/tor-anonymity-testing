import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from pathlib import Path

def plot_distribution(
        values: np.ndarray,
        title: str = "Distribution",
        xlabel: str = "Value",
        bins: int = 50,
        save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot histogram with KDE.

    Args:
        values: Values to plot
        title: Plot title
        xlabel: X-axis label
        bins: Number of bins
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(values, bins=bins, alpha=0.7, density=True, label='Histogram')

    from scipy.stats import gaussian_kde
    if len(values) > 1:
        kde = gaussian_kde(values)
        x_range = np.linspace(values.min(), values.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig