"""
KERAAL Results Visualization
=============================
Visualize LOSO training results with confusion matrices,
performance comparisons, and detailed analysis.

Author: PhysioKorea MLOps Team
Date: 2026-01-13
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# IJCNN 2024 Baseline results for comparison
IJCNN_BASELINE = {
    'RTK': {'accuracy': 64.4, 'name': 'Torso Rotation'},
    'CTK': {'accuracy': 56.2, 'name': 'Hiding Face'},
    'ELK': {'accuracy': 43.0, 'name': 'Flank Stretch'}
}


def load_results(results_dir: Path = None):
    """Load latest results from results directory"""
    if results_dir is None:
        # Auto-detect
        candidates = [
            Path.home() / "keraal" / "results",
            Path("D:/keraal/results"),
            Path("./results")
        ]
        for path in candidates:
            if path.exists():
                results_dir = path
                break

    # Find latest metrics file
    metrics_files = list(results_dir.glob("metrics_*.json"))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics files found in {results_dir}")

    latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)
    timestamp = latest_metrics.stem.replace("metrics_", "")

    print(f"Loading results from timestamp: {timestamp}")

    # Load metrics
    with open(latest_metrics) as f:
        metrics = json.load(f)

    # Load predictions CSV
    predictions_file = results_dir / f"predictions_{timestamp}.csv"
    predictions = None
    if predictions_file.exists():
        predictions = pd.read_csv(predictions_file)
        print(f"Loaded {len(predictions)} predictions")

    # Load PKL for full data
    pkl_file = results_dir / "loso_results.pkl"
    full_results = None
    if pkl_file.exists():
        with open(pkl_file, 'rb') as f:
            full_results = pickle.load(f)

    return metrics, predictions, full_results, timestamp


def plot_confusion_matrix(cm, classes, title='Confusion Matrix',
                          normalize=True, save_path=None, ax=None):
    """Plot confusion matrix with annotations"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_norm
    else:
        cm_display = cm

    # Plot heatmap
    sns.heatmap(cm_display, annot=False, fmt='.2%' if normalize else 'd',
                cmap='Blues', xticklabels=classes, yticklabels=classes,
                ax=ax, cbar=True)

    # Add annotations with both percentage and count
    for i in range(len(classes)):
        for j in range(len(classes)):
            if normalize:
                text = f'{cm_norm[i, j]:.1%}\n({cm[i, j]})'
            else:
                text = f'{cm[i, j]}'
            ax.text(j + 0.5, i + 0.5, text,
                   ha='center', va='center',
                   fontsize=12, fontweight='bold',
                   color='white' if cm_display[i, j] > 0.5 else 'black')

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return ax


def plot_performance_comparison(metrics, save_path=None):
    """Plot our results vs IJCNN baseline"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    exercises = ['RTK', 'CTK', 'ELK']
    our_results = []
    baseline_results = []

    for ex in exercises:
        if ex in metrics['per_exercise_metrics']:
            our_results.append(metrics['per_exercise_metrics'][ex]['accuracy']['mean'] * 100)
        else:
            our_results.append(0)
        baseline_results.append(IJCNN_BASELINE[ex]['accuracy'])

    # Bar chart comparison
    x = np.arange(len(exercises))
    width = 0.35

    bars1 = axes[0].bar(x - width/2, baseline_results, width, label='IJCNN 2024 Baseline',
                        color='#ff7f0e', alpha=0.8)
    bars2 = axes[0].bar(x + width/2, our_results, width, label='Our LOSO Result',
                        color='#1f77b4', alpha=0.8)

    # Add value labels
    for bar, val in zip(bars1, baseline_results):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, our_results):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_xlabel('Exercise', fontsize=12)
    axes[0].set_title('Performance Comparison: Our Method vs IJCNN 2024 Baseline',
                      fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{ex}\n({IJCNN_BASELINE[ex]["name"]})' for ex in exercises])
    axes[0].legend(loc='upper left')
    axes[0].set_ylim(0, 100)
    axes[0].axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Random')

    # Improvement chart
    improvements = [(our - base) / base * 100 for our, base in zip(our_results, baseline_results)]
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]

    bars3 = axes[1].bar(exercises, improvements, color=colors, alpha=0.8)

    for bar, imp in zip(bars3, improvements):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'+{imp:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    axes[1].set_ylabel('Improvement over Baseline (%)', fontsize=12)
    axes[1].set_xlabel('Exercise', fontsize=12)
    axes[1].set_title('Relative Improvement over IJCNN 2024 Baseline',
                      fontsize=14, fontweight='bold')
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_per_exercise_confusion_matrices(metrics, save_path=None):
    """Plot confusion matrices for each exercise"""
    exercises = ['RTK', 'CTK', 'ELK']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    classes = ['Correct', 'Error']

    for idx, ex in enumerate(exercises):
        if ex in metrics['per_exercise_metrics']:
            cm = np.array(metrics['per_exercise_metrics'][ex]['confusion_matrix'])
            acc = metrics['per_exercise_metrics'][ex]['accuracy']['mean'] * 100
            bal_acc = metrics['per_exercise_metrics'][ex]['balanced_accuracy']['mean'] * 100

            plot_confusion_matrix(cm, classes,
                                 title=f'{ex} ({IJCNN_BASELINE[ex]["name"]})\nAcc: {acc:.1f}%, Bal.Acc: {bal_acc:.1f}%',
                                 ax=axes[idx])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_error_detection_analysis(metrics, save_path=None):
    """Analyze error detection performance"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall confusion matrix
    cm = np.array(metrics['all_metrics']['confusion_matrix'])

    # Calculate metrics
    tn, fp = cm[0]
    fn, tp = cm[1]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # Error detection rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Correct classification rate

    # Plot confusion matrix
    classes = ['Correct', 'Error']
    plot_confusion_matrix(cm, classes, title='Overall Confusion Matrix', ax=axes[0])

    # Metrics bar chart
    metrics_names = ['Error Detection\n(Recall)', 'Precision', 'Specificity', 'Accuracy']
    metrics_values = [recall * 100, precision * 100, specificity * 100,
                      metrics['all_metrics']['accuracy']['mean'] * 100]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

    bars = axes[1].bar(metrics_names, metrics_values, color=colors, alpha=0.8)

    for bar, val in zip(bars, metrics_values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    axes[1].set_ylabel('Score (%)', fontsize=12)
    axes[1].set_title('Error Detection Performance Metrics', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 110)
    axes[1].axhline(y=100, color='gray', linestyle='--', alpha=0.3)

    # Add annotation
    axes[1].annotate(f'False Negatives: {fn}\n(Errors missed)',
                    xy=(0, recall * 100), xytext=(0.5, 60),
                    fontsize=10, ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray'))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def plot_prediction_distribution(predictions, save_path=None):
    """Plot prediction confidence distribution"""
    if predictions is None:
        print("No predictions data available")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Confidence distribution by correctness
    correct_preds = predictions[predictions['correct'] == True]['confidence']
    incorrect_preds = predictions[predictions['correct'] == False]['confidence']

    axes[0, 0].hist(correct_preds, bins=50, alpha=0.7, label='Correct Predictions', color='#2ecc71')
    axes[0, 0].hist(incorrect_preds, bins=50, alpha=0.7, label='Incorrect Predictions', color='#e74c3c')
    axes[0, 0].set_xlabel('Confidence', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].legend()

    # 2. Confidence by exercise
    for ex in predictions['exercise'].unique():
        ex_data = predictions[predictions['exercise'] == ex]
        axes[0, 1].hist(ex_data['confidence'], bins=30, alpha=0.5, label=ex)
    axes[0, 1].set_xlabel('Confidence', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_title('Confidence Distribution by Exercise', fontsize=14, fontweight='bold')
    axes[0, 1].legend()

    # 3. Accuracy by exercise
    exercise_acc = predictions.groupby('exercise')['correct'].mean() * 100
    bars = axes[1, 0].bar(exercise_acc.index, exercise_acc.values, color='#3498db', alpha=0.8)
    for bar, val in zip(bars, exercise_acc.values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 0].set_title('Accuracy by Exercise', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylim(0, 100)

    # 4. True vs Predicted distribution
    true_counts = predictions['true_label'].value_counts()
    pred_counts = predictions['pred_label'].value_counts()

    x = np.arange(2)
    width = 0.35
    axes[1, 1].bar(x - width/2, [true_counts.get('Correct', 0), true_counts.get('Error', 0)],
                   width, label='Ground Truth', color='#2ecc71', alpha=0.8)
    axes[1, 1].bar(x + width/2, [pred_counts.get('Correct', 0), pred_counts.get('Error', 0)],
                   width, label='Predicted', color='#3498db', alpha=0.8)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(['Correct', 'Error'])
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_title('Label Distribution: Ground Truth vs Predicted', fontsize=14, fontweight='bold')
    axes[1, 1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig


def generate_summary_table(metrics):
    """Generate a summary table of results"""
    print("\n" + "="*70)
    print(" KERAAL LOSO TRAINING RESULTS SUMMARY")
    print("="*70)

    # Overall metrics
    print("\n📊 OVERALL PERFORMANCE")
    print("-"*50)
    print(f"  Accuracy:          {metrics['all_metrics']['accuracy']['mean']*100:.2f}% ± {metrics['all_metrics']['accuracy']['std']*100:.2f}%")
    print(f"  Balanced Accuracy: {metrics['all_metrics']['balanced_accuracy']['mean']*100:.2f}% ± {metrics['all_metrics']['balanced_accuracy']['std']*100:.2f}%")
    print(f"  F1 Score:          {metrics['all_metrics']['f1']['mean']*100:.2f}% ± {metrics['all_metrics']['f1']['std']*100:.2f}%")

    # Per-exercise metrics
    print("\n📈 PER-EXERCISE PERFORMANCE")
    print("-"*50)
    print(f"{'Exercise':<10} {'Accuracy':<15} {'IJCNN Baseline':<15} {'Improvement':<15}")
    print("-"*50)

    for ex in ['RTK', 'CTK', 'ELK']:
        if ex in metrics['per_exercise_metrics']:
            acc = metrics['per_exercise_metrics'][ex]['accuracy']['mean'] * 100
            baseline = IJCNN_BASELINE[ex]['accuracy']
            improvement = (acc - baseline) / baseline * 100
            print(f"{ex:<10} {acc:.1f}%{'':<10} {baseline:.1f}%{'':<10} +{improvement:.1f}%")

    # Confusion matrix analysis
    cm = np.array(metrics['all_metrics']['confusion_matrix'])
    tn, fp = cm[0]
    fn, tp = cm[1]

    print("\n🔍 ERROR DETECTION ANALYSIS")
    print("-"*50)
    print(f"  True Positives (Errors detected):    {tp}")
    print(f"  True Negatives (Correct classified): {tn}")
    print(f"  False Positives (False alarms):      {fp}")
    print(f"  False Negatives (Errors missed):     {fn}")
    print(f"  Error Detection Rate (Recall):       {tp/(tp+fn)*100:.1f}%")
    print(f"  Precision:                           {tp/(tp+fp)*100:.1f}%")

    print("\n" + "="*70)


def main():
    """Main visualization pipeline"""
    print("KERAAL Results Visualization")
    print("="*50)

    # Load results
    try:
        metrics, predictions, full_results, timestamp = load_results()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run training first or specify results directory.")
        return

    # Create output directory
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    # Generate summary
    generate_summary_table(metrics)

    # Generate visualizations
    print("\n📊 Generating visualizations...")

    # 1. Performance comparison
    plot_performance_comparison(metrics, save_path=output_dir / f'performance_comparison_{timestamp}.png')

    # 2. Per-exercise confusion matrices
    plot_per_exercise_confusion_matrices(metrics, save_path=output_dir / f'confusion_matrices_{timestamp}.png')

    # 3. Error detection analysis
    plot_error_detection_analysis(metrics, save_path=output_dir / f'error_detection_{timestamp}.png')

    # 4. Prediction distribution (if available)
    if predictions is not None:
        plot_prediction_distribution(predictions, save_path=output_dir / f'prediction_distribution_{timestamp}.png')

    print(f"\n✅ All visualizations saved to {output_dir}/")
    print("\nGenerated files:")
    for f in output_dir.glob(f"*_{timestamp}.png"):
        print(f"  - {f.name}")

    # Show all plots
    plt.show()


if __name__ == '__main__':
    main()
