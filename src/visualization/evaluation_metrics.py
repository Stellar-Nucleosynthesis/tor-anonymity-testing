import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from pathlib import Path

def plot_metrics_comparison(scenario_results: Dict[str, Dict],
                            metrics: List[str] = None,
                            save_path: Optional[Path] = None) -> plt.Figure:
    """
    Plot bar chart comparing metrics across scenarios.

    Args:
        scenario_results: Dict of scenario results
        metrics: List of metrics to compare
        save_path: Path to save plot

    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ['roc_auc', 'f1_score', 'precision', 'recall', 'accuracy']

    data = []
    for scenario, results in scenario_results.items():
        for metric in metrics:
            if metric in results:
                data.append({
                    'Scenario': scenario,
                    'Metric': metric,
                    'Value': results[metric]
                })

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(14, 8))

    scenarios = df['Scenario'].unique()
    x = np.arange(len(metrics))
    width = 0.8 / len(scenarios)

    for i, scenario in enumerate(scenarios):
        scenario_data = df[df['Scenario'] == scenario]
        values = [scenario_data[scenario_data['Metric'] == m]['Value'].values[0]
                  if len(scenario_data[scenario_data['Metric'] == m]) > 0 else 0
                  for m in metrics]

        ax.bar(x + i * width, values, width, label=scenario)

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Attack Scenario Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig