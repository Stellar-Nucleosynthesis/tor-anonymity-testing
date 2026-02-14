import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from pathlib import Path

from src.visualization.evaluation_metrics import plot_metrics_comparison
from src.visualization.roc import plot_multiple_roc_curves, plot_roc_curve


def create_comprehensive_report(scenario_results: Dict[str, Dict],
                                save_dir: Path,
                                style: str = 'seaborn-v0_8-darkgrid'):
    """
    Create comprehensive visual report.

    Args:
        scenario_results: Results from all scenarios
        save_dir: Directory to save report plots
        style: Color style of the report
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use(style if style in plt.style.available else 'default')
    sns.set_palette("husl")

    plot_multiple_roc_curves(
        scenario_results,
        save_path=save_dir / 'roc_comparison.png'
    )

    plot_metrics_comparison(
        scenario_results,
        save_path=save_dir / 'metrics_comparison.png'
    )

    for scenario_name, results in scenario_results.items():
        scenario_dir = save_dir / scenario_name
        scenario_dir.mkdir(exist_ok=True)

        if 'fpr' in results and 'tpr' in results:
            plot_roc_curve(
                results['fpr'],
                results['tpr'],
                results.get('roc_auc', 0),
                scenario_name,
                save_path=scenario_dir / 'roc_curve.png'
            )
