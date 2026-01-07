"""
Train with Class Balancing to improve Error detection
Problem: Error detection rate is only 45.5%
Solution: Apply class weights and oversampling
"""
import sys
import numpy as np
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from convert_full_dataset import KeraalSample


def extract_features(skeleton):
    T, J, C = skeleton.shape
    center = skeleton[:, 0:1, :]
    skeleton = skeleton - center
    scale = np.max(np.abs(skeleton)) + 1e-8
    skeleton = skeleton / scale

    flat = skeleton.reshape(T, -1)
    features = []
    features.extend(flat.mean(axis=0))
    features.extend(flat.std(axis=0))
    features.extend(flat.max(axis=0))
    features.extend(flat.min(axis=0))

    velocity = np.diff(flat, axis=0)
    if len(velocity) > 0:
        features.extend(velocity.mean(axis=0))
        features.extend(velocity.std(axis=0))

    return np.array(features, dtype=np.float32)


def evaluate_model(y_true, y_pred, model_name):
    """Evaluate and print metrics."""
    acc = accuracy_score(y_true, y_pred)

    # Per-class metrics
    cm = confusion_matrix(y_true, y_pred)

    correct_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() > 0 else 0
    error_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() > 0 else 0

    print(f"\n{model_name}:")
    print(f"  Overall Accuracy: {acc:.1%}")
    print(f"  Correct Recall:   {correct_recall:.1%} ({cm[0,0]}/{cm[0].sum()})")
    print(f"  Error Recall:     {error_recall:.1%} ({cm[1,1]}/{cm[1].sum()})")
    print(f"  Confusion Matrix:")
    print(f"                    Pred Correct  Pred Error")
    print(f"    True Correct        {cm[0,0]:>6}       {cm[0,1]:>6}")
    print(f"    True Error          {cm[1,0]:>6}       {cm[1,1]:>6}")

    return {
        'accuracy': acc,
        'correct_recall': correct_recall,
        'error_recall': error_recall,
        'confusion_matrix': cm
    }


def run_cv(X, y, model, model_name, n_folds=5):
    """Run cross-validation."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_preds = np.zeros_like(y)

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_train = np.nan_to_num(X_train)
        X_val = np.nan_to_num(X_val)

        model.fit(X_train, y_train)
        all_preds[val_idx] = model.predict(X_val)

    return evaluate_model(y, all_preds, model_name)


def main():
    print("="*60)
    print("Class-Balanced Training for Better Error Detection")
    print("="*60)

    # Load data
    with open('processed/keraal_group2a_full.pkl', 'rb') as f:
        data = pickle.load(f)
    samples = data['samples']

    # Prepare data
    X, y = [], []
    for s in samples:
        if s.kinect is None:
            continue
        X.append(extract_features(s.kinect))
        y.append(0 if s.is_correct else 1)

    X = np.array(X)
    y = np.array(y)

    print(f"\nDataset: {len(y)} samples")
    print(f"  Correct: {(y==0).sum()}")
    print(f"  Error: {(y==1).sum()}")
    print(f"  Imbalance ratio: {(y==0).sum() / (y==1).sum():.1f}:1")

    # 1. Baseline (no balancing)
    print("\n" + "="*60)
    print("[1] Baseline (No Balancing)")
    print("="*60)

    baseline_svm = SVC(kernel='linear', C=1.0, random_state=42)
    run_cv(X, y, baseline_svm, "SVM Linear (baseline)")

    baseline_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    run_cv(X, y, baseline_rf, "Random Forest (baseline)")

    # 2. Class Weight Balancing
    print("\n" + "="*60)
    print("[2] Class Weight Balancing")
    print("="*60)

    balanced_svm = SVC(kernel='linear', C=1.0, class_weight='balanced', random_state=42)
    run_cv(X, y, balanced_svm, "SVM Linear (balanced)")

    balanced_rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                          class_weight='balanced', random_state=42)
    run_cv(X, y, balanced_rf, "Random Forest (balanced)")

    # 3. Custom Class Weights (emphasize Error more)
    print("\n" + "="*60)
    print("[3] Custom Weights (Error x5)")
    print("="*60)

    custom_weights = {0: 1, 1: 5}  # Penalize Error misclassification more

    custom_svm = SVC(kernel='linear', C=1.0, class_weight=custom_weights, random_state=42)
    run_cv(X, y, custom_svm, "SVM Linear (Error x5)")

    custom_rf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                        class_weight=custom_weights, random_state=42)
    run_cv(X, y, custom_rf, "Random Forest (Error x5)")

    # 4. Summary
    print("\n" + "="*60)
    print("SUMMARY: Error Detection Rate Comparison")
    print("="*60)
    print(f"{'Method':<30} {'Error Recall':>15}")
    print("-"*50)


if __name__ == '__main__':
    main()
