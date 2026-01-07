"""
KERAAL LOSO Training Pipeline
==============================
Leave-One-Subject-Out Cross-Validation for Rehabilitation Assessment

Based on IJCNN 2024 baseline methodology:
- LOSO evaluation (most rigorous)
- Per-exercise evaluation (RTK, CTK, ELK)
- Balanced Accuracy for class imbalance

Author: PhysioKorea MLOps Team
Date: 2026-01-07
"""

import os
import sys
import numpy as np
import pickle
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

import warnings
warnings.filterwarnings('ignore')

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent))
from keraal_converter import KeraalSample


# ============================================================
# Configuration
# ============================================================
def get_project_root() -> Path:
    """Auto-detect project root based on environment"""
    # Check common locations
    candidates = [
        Path.home() / "keraal",           # HPC: ~/keraal
        Path("D:/keraal"),                 # Windows
        Path(__file__).parent,             # Script location
    ]
    for path in candidates:
        if path.exists() and (path / "data").exists():
            return path
    return Path(__file__).parent


@dataclass
class Config:
    """Training configuration"""
    # Paths (auto-detected)
    PROJECT_ROOT: Path = None
    DATA_PATH: Path = None
    RAW_DATA_DIR: Path = None
    RESULT_DIR: Path = None

    def __post_init__(self):
        if self.PROJECT_ROOT is None:
            self.PROJECT_ROOT = get_project_root()
        if self.DATA_PATH is None:
            self.DATA_PATH = self.PROJECT_ROOT / "data" / "processed" / "keraal_loso.pkl"
        if self.RAW_DATA_DIR is None:
            self.RAW_DATA_DIR = self.PROJECT_ROOT / "data" / "raw"
        if self.RESULT_DIR is None:
            self.RESULT_DIR = self.PROJECT_ROOT / "results"

    # Model
    HIDDEN_SIZE: int = 128
    NUM_LAYERS: int = 2
    DROPOUT: float = 0.3

    # Training (IJCNN 2024: 1000 epochs, batch 32, lr 0.01)
    BATCH_SIZE: int = 32
    EPOCHS: int = 50  # Reduced for sample data (increase to 200+ for full data)
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 0.01

    # Data
    MAX_SEQ_LEN: int = 300
    SKELETON_TYPE: str = 'kinect'  # 'kinect' or 'blazepose'

    # Evaluation
    NUM_RUNS: int = 5  # IJCNN: 10 runs average
    TASK: str = 'binary'  # 'binary' or 'multiclass'


# ============================================================
# Raw Data Loading
# ============================================================
def load_raw_data(config: Config) -> List:
    """Load data from raw kinect/annotator files"""
    import xml.etree.ElementTree as ET

    raw_dir = config.RAW_DATA_DIR
    kinect_dir = raw_dir / "kinect"
    annotator_dir = raw_dir / "annotatorA"

    if not kinect_dir.exists():
        print(f"Kinect directory not found: {kinect_dir}")
        return []

    samples = []
    kinect_files = list(kinect_dir.glob("*.txt"))
    print(f"Found {len(kinect_files)} kinect files")

    for kinect_file in kinect_files:
        # Parse filename: G1A-Kinect-CTK-R1-Brest-022.txt
        name = kinect_file.stem
        parts = name.replace("Kinect-", "").split("-")

        # Extract info
        if len(parts) >= 5:
            group = "group1A" if name.startswith("G1A") else "group2A" if name.startswith("G2A") else "group3"
            exercise = parts[1] if len(parts) > 1 else "Unknown"

            # Load skeleton data
            try:
                skeleton = np.loadtxt(kinect_file)
                if skeleton.ndim == 1:
                    skeleton = skeleton.reshape(1, -1)
            except Exception as e:
                print(f"Error loading {kinect_file}: {e}")
                continue

            # Check for annotation
            anvil_name = name.replace("Kinect-", "").replace("G1A-", "G1A-").replace("G2A-", "G2A-") + ".anvil"
            anvil_file = annotator_dir / anvil_name

            is_correct = True
            error_type = None

            if anvil_file.exists():
                try:
                    tree = ET.parse(anvil_file)
                    root = tree.getroot()
                    # Look for error annotations
                    for el in root.iter():
                        if 'error' in el.tag.lower() or 'label' in el.tag.lower():
                            if el.text and el.text.strip():
                                is_correct = False
                                error_type = el.text.strip()
                                break
                except Exception:
                    pass

            # Create sample
            sample = KeraalSample(
                id=name,
                group=group,
                exercise=exercise,
                skeleton_kinect=skeleton,
                is_correct=is_correct,
                error_type=error_type
            )
            samples.append(sample)

    print(f"Loaded {len(samples)} samples from raw data")
    return samples


def load_or_create_data(config: Config) -> List:
    """Load from pkl or create from raw data"""
    # Try loading pkl first
    if config.DATA_PATH.exists():
        print(f"Loading from {config.DATA_PATH}")
        with open(config.DATA_PATH, 'rb') as f:
            data = pickle.load(f)
        return data.get('samples', data) if isinstance(data, dict) else data

    # Load from raw data
    print(f"PKL not found, loading from raw data...")
    samples = load_raw_data(config)

    if samples:
        # Save for future use
        config.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(config.DATA_PATH, 'wb') as f:
            pickle.dump({'samples': samples}, f)
        print(f"Saved to {config.DATA_PATH}")

    return samples


# ============================================================
# Subject Extraction
# ============================================================
def extract_subject_id(sample_id: str, group: str) -> str:
    """
    Extract subject ID from sample ID.

    Examples:
        G1A-CTK-R1-Brest-022 → Brest-022
        G3-Kinect-CTK-P1T1-Unknown-E1B1-0 → P1T1
    """
    parts = sample_id.split('-')

    if group in ['group1A', 'group2A']:
        # Format: G1A-CTK-R1-Brest-022 → last 2 parts
        if len(parts) >= 5:
            return f"{parts[-2]}-{parts[-1]}"

    elif group == 'group3':
        # Format: G3-Kinect-CTK-P1T1-Unknown-E1B1-0 → P1T1
        if len(parts) >= 4:
            return parts[3]  # P1T1

    return sample_id  # Fallback


def extract_exercise_type(sample_id: str) -> str:
    """
    Extract exercise type from sample ID.

    Examples:
        G1A-CTK-R1-Brest-022 → CTK
        G3-Kinect-RTK-P1T1-Unknown-E1B1-0 → RTK
    """
    for ex in ['CTK', 'RTK', 'ELK']:
        if ex in sample_id.upper():
            return ex
    return 'Unknown'


# ============================================================
# Dataset
# ============================================================
class KeraalLOSODataset(Dataset):
    """PyTorch Dataset with subject info for LOSO."""

    def __init__(self, samples: List[KeraalSample], config: Config, task: str = 'binary'):
        self.samples = samples
        self.config = config
        self.task = task

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get skeleton data
        if self.config.SKELETON_TYPE == 'kinect' and sample.kinect is not None:
            skeleton = sample.kinect.copy()
        elif sample.blazepose is not None:
            skeleton = sample.blazepose.copy()
        else:
            skeleton = sample.kinect.copy() if sample.kinect is not None else np.zeros((100, 25, 3))

        # Normalize
        skeleton = self._normalize(skeleton)

        # Extract features
        features = self._extract_features(skeleton)

        # Pad/truncate
        features = self._pad_truncate(features)

        # Label
        if self.task == 'binary':
            label = 0 if sample.is_correct else 1
        else:
            # Multiclass: Correct=0, Error1=1, Error2=2, Error3=3
            label_map = {'none': 0, 'Error1': 1, 'Error2': 2, 'Error3': 3}
            label = label_map.get(sample.error_type, 0) if not sample.is_correct else 0

        # Subject ID for LOSO
        subject_id = extract_subject_id(sample.sample_id, sample.group)
        exercise = sample.exercise

        return {
            'features': torch.FloatTensor(features),
            'label': torch.LongTensor([label]),
            'subject_id': subject_id,
            'exercise': exercise,
            'sample_id': sample.sample_id
        }

    def _normalize(self, skeleton: np.ndarray) -> np.ndarray:
        """Normalize skeleton: center + scale."""
        # Center on first joint (SpineBase/Nose)
        center = skeleton[:, 0:1, :]
        skeleton = skeleton - center

        # Scale to unit box
        min_val = skeleton.min()
        max_val = skeleton.max()
        if max_val - min_val > 1e-6:
            skeleton = (skeleton - min_val) / (max_val - min_val)

        return skeleton

    def _extract_features(self, skeleton: np.ndarray) -> np.ndarray:
        """Extract position + velocity + acceleration features."""
        T, J, D = skeleton.shape

        # Flatten: (T, J, D) → (T, J*D)
        pos = skeleton.reshape(T, -1)

        # Velocity
        vel = np.diff(skeleton, axis=0, prepend=skeleton[0:1])
        vel = vel.reshape(T, -1)

        # Acceleration
        acc = np.diff(vel, axis=0, prepend=vel[0:1])

        # Concatenate
        features = np.concatenate([pos, vel, acc], axis=1)

        return features

    def _pad_truncate(self, features: np.ndarray) -> np.ndarray:
        """Pad or truncate to fixed length."""
        T = features.shape[0]

        if T < self.config.MAX_SEQ_LEN:
            pad_len = self.config.MAX_SEQ_LEN - T
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
        else:
            features = features[:self.config.MAX_SEQ_LEN]

        return features


# ============================================================
# Model
# ============================================================
class LSTMClassifier(nn.Module):
    """Bidirectional LSTM with Attention."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        logits = self.classifier(context)
        return logits


# ============================================================
# LOSO Cross-Validation
# ============================================================
class LOSOEvaluator:
    """Leave-One-Subject-Out Cross-Validation Evaluator."""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self, samples: List[KeraalSample], exercise: str = None) -> Dict:
        """
        Run LOSO evaluation.

        Args:
            samples: List of KeraalSample
            exercise: If specified, filter by exercise type (CTK, RTK, ELK)

        Returns:
            Dict with evaluation metrics
        """
        # Filter by exercise if specified
        if exercise:
            samples = [s for s in samples if s.exercise == exercise]

        if len(samples) == 0:
            print(f"No samples for exercise: {exercise}")
            return {}

        # Group samples by subject
        subject_samples = defaultdict(list)
        for sample in samples:
            subject_id = extract_subject_id(sample.sample_id, sample.group)
            subject_samples[subject_id].append(sample)

        subjects = list(subject_samples.keys())
        n_subjects = len(subjects)

        print(f"\n{'='*60}")
        print(f"LOSO Evaluation: {exercise or 'All exercises'}")
        print(f"{'='*60}")
        print(f"Total samples: {len(samples)}")
        print(f"Subjects: {n_subjects}")
        print(f"Samples per subject: {[len(subject_samples[s]) for s in subjects]}")

        # Run multiple times
        all_results = []

        for run in range(self.config.NUM_RUNS):
            run_results = self._run_loso(samples, subject_samples, subjects, run)
            all_results.append(run_results)

        # Aggregate results
        metrics = self._aggregate_results(all_results)

        return metrics

    def _run_loso(self, samples, subject_samples, subjects, run_idx) -> List[Dict]:
        """Run one LOSO iteration."""
        results = []

        for fold_idx, test_subject in enumerate(subjects):
            # Split by subject
            train_samples = []
            test_samples = []

            for subject, s_list in subject_samples.items():
                if subject == test_subject:
                    test_samples.extend(s_list)
                else:
                    train_samples.extend(s_list)

            if len(train_samples) == 0 or len(test_samples) == 0:
                continue

            # Create datasets
            train_dataset = KeraalLOSODataset(train_samples, self.config, self.config.TASK)
            test_dataset = KeraalLOSODataset(test_samples, self.config, self.config.TASK)

            train_loader = DataLoader(
                train_dataset,
                batch_size=min(self.config.BATCH_SIZE, len(train_samples)),
                shuffle=True
            )

            # Get input size
            sample_features = train_dataset[0]['features']
            input_size = sample_features.shape[1]
            num_classes = 2 if self.config.TASK == 'binary' else 4

            # Create model
            model = LSTMClassifier(
                input_size=input_size,
                hidden_size=self.config.HIDDEN_SIZE,
                num_layers=self.config.NUM_LAYERS,
                num_classes=num_classes,
                dropout=self.config.DROPOUT
            ).to(self.device)

            # Train
            self._train_model(model, train_loader)

            # Evaluate
            fold_results = self._evaluate_model(model, test_dataset, test_subject)
            results.extend(fold_results)

            # Progress
            if run_idx == 0:
                n_correct = sum(1 for r in fold_results if r['correct'])
                print(f"  Fold {fold_idx+1}/{len(subjects)}: Subject {test_subject} | "
                      f"{n_correct}/{len(fold_results)} correct")

        return results

    def _train_model(self, model, train_loader):
        """Train model."""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )

        # Class weights for imbalance
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(self.config.EPOCHS):
            for batch in train_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].squeeze(-1).to(self.device)

                optimizer.zero_grad()
                logits = model(features)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

    def _evaluate_model(self, model, test_dataset, test_subject) -> List[Dict]:
        """Evaluate model on test set."""
        results = []
        model.eval()

        with torch.no_grad():
            for i in range(len(test_dataset)):
                sample = test_dataset[i]
                features = sample['features'].unsqueeze(0).to(self.device)
                true_label = sample['label'].item()

                logits = model(features)
                pred_label = logits.argmax(dim=1).item()

                results.append({
                    'sample_id': sample['sample_id'],
                    'subject_id': test_subject,
                    'exercise': sample['exercise'],
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'correct': pred_label == true_label
                })

        return results

    def _aggregate_results(self, all_results: List[List[Dict]]) -> Dict:
        """Aggregate results across runs."""
        # Flatten all results
        all_true = []
        all_pred = []

        for run_results in all_results:
            for r in run_results:
                all_true.append(r['true_label'])
                all_pred.append(r['pred_label'])

        # Per-run metrics
        run_metrics = []
        for run_results in all_results:
            y_true = [r['true_label'] for r in run_results]
            y_pred = [r['pred_label'] for r in run_results]

            run_metrics.append({
                'accuracy': accuracy_score(y_true, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'precision': precision_recall_fscore_support(y_true, y_pred, average='macro')[0],
                'recall': precision_recall_fscore_support(y_true, y_pred, average='macro')[1],
                'f1': precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
            })

        # Average and std
        metrics = {
            'accuracy': {
                'mean': np.mean([m['accuracy'] for m in run_metrics]),
                'std': np.std([m['accuracy'] for m in run_metrics])
            },
            'balanced_accuracy': {
                'mean': np.mean([m['balanced_accuracy'] for m in run_metrics]),
                'std': np.std([m['balanced_accuracy'] for m in run_metrics])
            },
            'f1': {
                'mean': np.mean([m['f1'] for m in run_metrics]),
                'std': np.std([m['f1'] for m in run_metrics])
            },
            'confusion_matrix': confusion_matrix(all_true, all_pred),
            'n_samples': len(all_results[0]),
            'n_runs': len(all_results)
        }

        return metrics


# ============================================================
# Main
# ============================================================
def print_results(metrics: Dict, title: str):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {title}")
    print(f"{'='*60}")

    print(f"Accuracy:          {metrics['accuracy']['mean']:.1%} ± {metrics['accuracy']['std']:.1%}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']['mean']:.1%} ± {metrics['balanced_accuracy']['std']:.1%}")
    print(f"F1 Score:          {metrics['f1']['mean']:.1%} ± {metrics['f1']['std']:.1%}")

    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    if cm.shape == (2, 2):
        print(f"              Pred Correct  Pred Error")
        print(f"True Correct      {cm[0,0]:5d}       {cm[0,1]:5d}")
        print(f"True Error        {cm[1,0]:5d}       {cm[1,1]:5d}")
    else:
        print(cm)


def main():
    print("="*60)
    print("KERAAL LOSO Training Pipeline")
    print("="*60)

    # Setup
    config = Config()
    print(f"\nDevice: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Task: {config.TASK}")
    print(f"Runs: {config.NUM_RUNS}")
    print(f"Epochs: {config.EPOCHS}")

    # Load data
    print(f"\nProject root: {config.PROJECT_ROOT}")
    print(f"Raw data dir: {config.RAW_DATA_DIR}")

    samples = load_or_create_data(config)
    if not samples:
        print("No data found. Check data directory.")
        return
    print(f"Loaded {len(samples)} samples")

    # Show distribution
    print("\nLabel distribution:")
    label_counts = defaultdict(int)
    for s in samples:
        label = 'Correct' if s.is_correct else (s.error_type or 'Error')
        label_counts[label] += 1
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print("\nExercise distribution:")
    ex_counts = defaultdict(int)
    for s in samples:
        ex_counts[s.exercise] += 1
    for ex, count in sorted(ex_counts.items()):
        print(f"  {ex}: {count}")

    # Create evaluator
    evaluator = LOSOEvaluator(config)

    # Run LOSO for all exercises combined
    all_metrics = evaluator.run(samples, exercise=None)
    if all_metrics:
        print_results(all_metrics, "ALL EXERCISES")

    # Run LOSO per exercise (IJCNN style)
    per_exercise_metrics = {}
    for ex in ['RTK', 'CTK', 'ELK']:
        ex_samples = [s for s in samples if s.exercise == ex]
        if len(ex_samples) >= 3:
            ex_metrics = evaluator.run(samples, exercise=ex)
            if ex_metrics:
                per_exercise_metrics[ex] = ex_metrics
                print_results(ex_metrics, ex)

    # Summary comparison
    print(f"\n{'='*60}")
    print("SUMMARY: Per-Exercise Performance")
    print(f"{'='*60}")
    print(f"{'Exercise':<10} {'Accuracy':<15} {'Balanced Acc':<15} {'F1 Score':<15}")
    print("-"*55)

    for ex in ['RTK', 'CTK', 'ELK']:
        if ex in per_exercise_metrics:
            m = per_exercise_metrics[ex]
            print(f"{ex:<10} "
                  f"{m['accuracy']['mean']:.1%} ± {m['accuracy']['std']:.1%}  "
                  f"{m['balanced_accuracy']['mean']:.1%} ± {m['balanced_accuracy']['std']:.1%}  "
                  f"{m['f1']['mean']:.1%} ± {m['f1']['std']:.1%}")

    # Compare with IJCNN 2024 baseline
    print(f"\n{'='*60}")
    print("COMPARISON: vs IJCNN 2024 Baseline (LSTM)")
    print(f"{'='*60}")
    print(f"{'Exercise':<10} {'IJCNN Best':<12} {'IJCNN Avg':<15} {'Ours':<15}")
    print("-"*55)

    ijcnn_results = {
        'RTK': {'best': 64.4, 'avg': 53.9},
        'CTK': {'best': 56.2, 'avg': 49.1},
        'ELK': {'best': 43.0, 'avg': 31.6}
    }

    for ex in ['RTK', 'CTK', 'ELK']:
        if ex in per_exercise_metrics and ex in ijcnn_results:
            ours = per_exercise_metrics[ex]['accuracy']['mean'] * 100
            ijcnn = ijcnn_results[ex]
            print(f"{ex:<10} {ijcnn['best']:.1f}%       {ijcnn['avg']:.1f}%          {ours:.1f}%")

    # Save results
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = config.RESULT_DIR / 'loso_results.pkl'

    with open(result_path, 'wb') as f:
        pickle.dump({
            'all_metrics': all_metrics,
            'per_exercise_metrics': per_exercise_metrics,
            'config': {
                'epochs': config.EPOCHS,
                'batch_size': config.BATCH_SIZE,
                'hidden_size': config.HIDDEN_SIZE,
                'num_runs': config.NUM_RUNS,
                'task': config.TASK
            }
        }, f)

    print(f"\nResults saved to {result_path}")


if __name__ == '__main__':
    main()
