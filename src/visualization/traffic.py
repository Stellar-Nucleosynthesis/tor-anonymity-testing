import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
from pathlib import Path

def plot_timeline(
        timestamps: np.ndarray,
        packet_sizes: np.ndarray,
        title: str = "Traffic Timeline",
        save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot traffic timeline.

    Args:
        timestamps: Packet timestamps
        packet_sizes: Packet sizes
        title: Plot title
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.scatter(timestamps, packet_sizes, alpha=0.5, s=10)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Packet Size (bytes)', fontsize=12)
    ax1.set_title(f'{title} - Packet Sizes', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    cumulative_bytes = np.cumsum(packet_sizes)
    ax2.plot(timestamps, cumulative_bytes, 'b-', linewidth=2)
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Cumulative Bytes', fontsize=12)
    ax2.set_title(f'{title} - Cumulative Traffic', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig