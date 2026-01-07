"""
Keraal Rehabilitation Assessment Classifier

Task: Binary Classification (Correct vs Error)
Method: LSTM + Leave-One-Out Cross-Validation
Based on Hawkeye CORAL approach but adapted for binary classification

Sample Distribution (8 samples):
- Correct: 4 samples (CTK, RTK, ELK)
- Error: 4 samples (Error1: 2, Error2: 1, Error3: 1)
"""
import os
import sys
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent))
from keraal_converter import KeraalSample

# ============================================================
# Configuration
# ============================================================
class Config:
    DATA_PATH = Path("D:/keraal/processed/keraal_hawkeye_format.pkl")
    RESULT_DIR = Path("D:/keraal/results")

    # Model
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.3

    # Training
    BATCH_SIZE = 2
    EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01

    # Data
    MAX_SEQ_LEN = 300  # Pad/truncate to this length
    SKELETON_TYPE = 'kinect'  # 'kinect' (25 joints) or 'blazepose' (33 joints)

    # Label mapping
    LABEL_MAP = {
        'Correct': 0,
        'Error1': 1,
        'Error2': 1,
        'Error3': 1
    }  # Binary: Correct vs Any Error


# ============================================================
# Feature Engineering
# ============================================================
class FeatureEngineer:
    """Feature extraction from skeleton sequences."""

    @staticmethod
    def add_velocity(x: np.ndarray) -> np.ndarray:
        """Compute velocity (first derivative)."""
        velocity = np.diff(x, axis=0, prepend=x[0:1])
        return velocity

    @staticmethod
    def add_acceleration(x: np.ndarray) -> np.ndarray:
        """Compute acceleration (second derivative)."""
        velocity = np.diff(x, axis=0, prepend=x[0:1])
        acceleration = np.diff(velocity, axis=0, prepend=velocity[0:1])
        return acceleration

    @staticmethod
    def extract_joint_angles(skeleton: np.ndarray, joint_triplets: List[Tuple[int, int, int]]) -> np.ndarray:
        """
        Extract joint angles from skeleton.

        Args:
            skeleton: (T, J, 3) skeleton sequence
            joint_triplets: List of (joint1, center, joint2) for angle calculation

        Returns:
            (T, num_angles) angle sequence
        """
        T = skeleton.shape[0]
        angles = []

        for j1, jc, j2 in joint_triplets:
            v1 = skeleton[:, j1, :] - skeleton[:, jc, :]
            v2 = skeleton[:, j2, :] - skeleton[:, jc, :]

            # Compute angle using dot product
            cos_angle = np.sum(v1 * v2, axis=1) / (
                np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
            )
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)

        return np.stack(angles, axis=1)

    @staticmethod
    def normalize_skeleton(skeleton: np.ndarray, center_joint: int = 0) -> np.ndarray:
        """
        Normalize skeleton relative to center joint.

        Args:
            skeleton: (T, J, 3) skeleton sequence
            center_joint: Index of center joint (0 = SpineBase for Kinect)

        Returns:
            Normalized skeleton
        """
        center = skeleton[:, center_joint:center_joint+1, :]
        normalized = skeleton - center

        # Scale to unit bounding box
        min_val = normalized.min(axis=(0, 1), keepdims=True)
        max_val = normalized.max(axis=(0, 1), keepdims=True)
        scale = max_val - min_val + 1e-8
        normalized = (normalized - min_val) / scale

        return normalized


# ============================================================
# Dataset
# ============================================================
class KeraalDataset(Dataset):
    """PyTorch Dataset for Keraal skeleton sequences."""

    # Kinect joint triplets for angle calculation
    KINECT_ANGLE_TRIPLETS = [
        (4, 5, 6),    # Left elbow angle
        (8, 9, 10),   # Right elbow angle
        (12, 13, 14), # Left knee angle
        (16, 17, 18), # Right knee angle
        (0, 1, 20),   # Spine angle
    ]

    def __init__(self, samples: List, indices: List[int], config: Config):
        self.samples = [samples[i] for i in indices]
        self.config = config
        self.feature_engineer = FeatureEngineer()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get skeleton data
        if self.config.SKELETON_TYPE == 'kinect' and sample.kinect is not None:
            skeleton = sample.kinect  # (T, 25, 3)
        elif sample.blazepose is not None:
            skeleton = sample.blazepose  # (T, 33, 3)
        else:
            # Fallback to any available
            skeleton = sample.kinect if sample.kinect is not None else sample.blazepose

        # Normalize skeleton
        skeleton = self.feature_engineer.normalize_skeleton(skeleton)

        # Compute velocity and acceleration
        velocity = self.feature_engineer.add_velocity(skeleton)
        acceleration = self.feature_engineer.add_acceleration(skeleton)

        # Flatten joints: (T, J, 3) -> (T, J*3)
        T, J, _ = skeleton.shape
        skeleton_flat = skeleton.reshape(T, -1)
        velocity_flat = velocity.reshape(T, -1)
        acceleration_flat = acceleration.reshape(T, -1)

        # Concatenate features
        features = np.concatenate([skeleton_flat, velocity_flat, acceleration_flat], axis=1)

        # Pad or truncate to max_seq_len
        if features.shape[0] < self.config.MAX_SEQ_LEN:
            pad_len = self.config.MAX_SEQ_LEN - features.shape[0]
            features = np.pad(features, ((0, pad_len), (0, 0)), mode='constant')
        else:
            features = features[:self.config.MAX_SEQ_LEN]

        # Get label (0 = Correct, 1 = Error)
        label = 0 if sample.is_correct else 1

        return {
            'features': torch.FloatTensor(features),
            'label': torch.LongTensor([label]),
            'sample_id': sample.sample_id
        }


# ============================================================
# Model: Bidirectional LSTM Classifier
# ============================================================
class LSTMClassifier(nn.Module):
    """Bidirectional LSTM for binary classification."""

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

        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        # LSTM
        lstm_out, _ = self.lstm(x)  # (B, T, hidden*2)

        # Attention pooling
        attn_weights = self.attention(lstm_out)  # (B, T, 1)
        context = torch.sum(attn_weights * lstm_out, dim=1)  # (B, hidden*2)

        # Classify
        logits = self.classifier(context)  # (B, num_classes)

        return logits


# ============================================================
# Training
# ============================================================
def train_one_fold(model, train_loader, val_sample, config, device):
    """Train model on one fold and evaluate on validation sample."""

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    best_val_correct = False

    model.train()
    for epoch in range(config.EPOCHS):
        total_loss = 0

        for batch in train_loader:
            features = batch['features'].to(device)
            labels = batch['label'].squeeze(-1).to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validate
        model.eval()
        with torch.no_grad():
            val_features = val_sample['features'].unsqueeze(0).to(device)
            val_label = val_sample['label'].item()

            logits = model(val_features)
            pred = logits.argmax(dim=1).item()

            if pred == val_label:
                best_val_correct = True
        model.train()

    return best_val_correct, pred, val_label


def leave_one_out_cv(samples, config, device):
    """Leave-One-Out Cross-Validation."""

    n_samples = len(samples)
    results = []

    print(f"\n{'='*60}")
    print(f"Leave-One-Out Cross-Validation ({n_samples} folds)")
    print(f"{'='*60}")

    for i in range(n_samples):
        # Split
        train_indices = [j for j in range(n_samples) if j != i]
        val_index = i

        # Create datasets
        train_dataset = KeraalDataset(samples, train_indices, config)
        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

        # Get validation sample
        val_dataset = KeraalDataset(samples, [val_index], config)
        val_sample = val_dataset[0]

        # Determine input size from features
        input_size = val_sample['features'].shape[1]

        # Create model
        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            num_classes=2,
            dropout=config.DROPOUT
        ).to(device)

        # Train
        correct, pred, true_label = train_one_fold(model, train_loader, val_sample, config, device)

        sample = samples[val_index]
        label_names = {0: 'Correct', 1: 'Error'}

        results.append({
            'sample_id': sample.sample_id,
            'true_label': true_label,
            'pred_label': pred,
            'correct': correct,
            'exercise': sample.exercise
        })

        status = "OK" if correct else "FAIL"
        print(f"Fold {i+1:2d}: {sample.sample_id[:30]:30s} | "
              f"True: {label_names[true_label]:7s} | "
              f"Pred: {label_names[pred]:7s} | {status}")

    return results


def evaluate_results(results):
    """Compute and display evaluation metrics."""

    y_true = [r['true_label'] for r in results]
    y_pred = [r['pred_label'] for r in results]

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.3f} ({sum(r['correct'] for r in results)}/{len(results)})")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"              Pred Correct  Pred Error")
    print(f"True Correct      {cm[0,0]:3d}          {cm[0,1]:3d}")
    print(f"True Error        {cm[1,0]:3d}          {cm[1,1]:3d}")

    # Per-exercise breakdown
    print(f"\n{'='*60}")
    print("PER-EXERCISE BREAKDOWN")
    print(f"{'='*60}")
    exercises = set(r['exercise'] for r in results)
    for ex in sorted(exercises):
        ex_results = [r for r in results if r['exercise'] == ex]
        ex_acc = sum(r['correct'] for r in ex_results) / len(ex_results)
        print(f"{ex}: {ex_acc:.1%} ({sum(r['correct'] for r in ex_results)}/{len(ex_results)})")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'results': results
    }


# ============================================================
# Main
# ============================================================
def main():
    print("="*60)
    print("Keraal Rehabilitation Assessment Classifier")
    print("="*60)

    # Setup
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load data
    print(f"\nLoading data from {config.DATA_PATH}...")
    with open(config.DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    print(f"Loaded {len(samples)} samples")

    # Display sample distribution
    label_counts = {}
    for s in samples:
        label = 'Correct' if s.is_correct else (s.error_type or 'Error')
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        binary_label = 'Correct' if label == 'Correct' else 'Error'
        print(f"  {label}: {count} ({binary_label})")

    # Run Leave-One-Out CV
    results = leave_one_out_cv(samples, config, device)

    # Evaluate
    metrics = evaluate_results(results)

    # Save results
    config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = config.RESULT_DIR / 'classification_results.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"\nResults saved to {result_path}")

    return metrics


if __name__ == '__main__':
    main()
