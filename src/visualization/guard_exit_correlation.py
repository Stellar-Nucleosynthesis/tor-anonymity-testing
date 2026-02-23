import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from pathlib import Path
import plotly.graph_objects as go

def plot_correlation_matrix(
        correlation_matrix: np.ndarray,
        guard_ids: Optional[List[str]] = None,
        exit_ids: Optional[List[str]] = None,
        save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot correlation matrix heatmap.

    Args:
        correlation_matrix: Matrix of correlation scores
        guard_ids: IDs for guard observations (rows)
        exit_ids: IDs for exit observations (columns)
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(correlation_matrix, cmap='RdYlGn', aspect='auto',
                   vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Correlation Score', fontsize=12)

    if guard_ids and len(guard_ids) <= 50:
        ax.set_yticks(range(len(guard_ids)))
        ax.set_yticklabels(guard_ids, fontsize=8)
    else:
        ax.set_ylabel('Guard Observations', fontsize=12)

    if exit_ids and len(exit_ids) <= 50:
        ax.set_xticks(range(len(exit_ids)))
        ax.set_xticklabels(exit_ids, rotation=45, ha='right', fontsize=8)
    else:
        ax.set_xlabel('Exit Observations', fontsize=12)

    ax.set_title('Traffic Correlation Matrix', fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_interactive_correlation_matrix(
        correlation_matrix: np.ndarray,
        guard_ids: List[str],
        exit_ids: List[str],
        save_path: Optional[Path] = None
) -> go.Figure:
    """
    Create interactive correlation matrix using Plotly.

    Args:
        correlation_matrix: Matrix of correlations
        guard_ids: Guard observation IDs
        exit_ids: Exit observation IDs
        save_path: Path to save HTML

    Returns:
        Plotly figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=exit_ids,
        y=guard_ids,
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        colorbar=dict(title="Correlation Score")
    ))

    fig.update_layout(
        title='Interactive Traffic Correlation Matrix',
        xaxis_title='Exit Observations',
        yaxis_title='Guard Observations',
        width=1200,
        height=1000
    )

    if save_path:
        fig.write_html(save_path)

    return fig