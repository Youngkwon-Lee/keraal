"""
Train ML Models on Full Keraal Group2A Dataset (51 samples)

Models:
1. Traditional ML (SVM, Random Forest, etc.)
2. LSTM with proper train/val split
"""
import sys
import numpy as np
import pickle
from pathlib import Path
from typing import List
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from convert_full_dataset import KeraalSample


def extract_statistical_features(skeleton: np.ndarray) -> np.ndarray:
    """Extract statistical features from skeleton sequence."""
    T, J, C = skeleton.shape
    features = []

    flat = skeleton.reshape(T, -1)

    # Global statistics
    features.extend(flat.mean(axis=0))
    features.extend(flat.std(axis=0))
    features.extend(flat.max(axis=0))
    features.extend(flat.min(axis=0))
    features.extend(flat.max(axis=0) - flat.min(axis=0))

    # Velocity
    velocity = np.diff(flat, axis=0)
    if len(velocity) > 0:
        features.extend(velocity.mean(axis=0))
        features.extend(velocity.std(axis=0))
        features.extend(np.abs(velocity).max(axis=0))
    else:
        features.extend(np.zeros(flat.shape[1] * 3))

    # Acceleration
    if len(velocity) > 1:
        acceleration = np.diff(velocity, axis=0)
        features.extend(acceleration.mean(axis=0))
        features.extend(acceleration.std(axis=0))
    else:
        features.extend(np.zeros(flat.shape[1] * 2))

    features.append(T)

    # Joint distances (key pairs)
    if J >= 25:  # Kinect
        key_joints = [0, 3, 7, 11, 15, 19]
    else:
        key_joints = [0, 11, 12, 15, 16, 23, 24]

    for i, j1 in enumerate(key_joints):
        for j2 in key_joints[i+1:]:
            if j1 < J and j2 < J:
                dist = np.linalg.norm(skeleton[:, j1, :] - skeleton[:, j2, :], axis=1)
                features.extend([dist.mean(), dist.std(), dist.max(), dist.min()])

    return np.array(features, dtype=np.float32)


def normalize_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """Center and scale skeleton."""
    center = skeleton[:, 0:1, :]
    centered = skeleton - center
    flat = centered.reshape(-1, 3)
    scale = np.max(np.abs(flat)) + 1e-8
    return centered / scale


def prepare_data(samples, skeleton_type='kinect'):
    """Prepare feature matrix and labels."""
    X, y, sample_ids = [], [], []

    for sample in samples:
        if skeleton_type == 'kinect' and sample.kinect is not None:
            skeleton = sample.kinect
        elif skeleton_type == 'blazepose' and sample.blazepose is not None:
            skeleton = sample.blazepose
        else:
            skeleton = sample.kinect if sample.kinect is not None else sample.blazepose

        if skeleton is None:
            continue

        skeleton = normalize_skeleton(skeleton)
        features = extract_statistical_features(skeleton)

        X.append(features)
        y.append(0 if sample.is_correct else 1)
        sample_ids.append(sample.sample_id)

    return np.array(X), np.array(y), sample_ids


def run_cross_validation(X, y, n_folds=5):
    """Run stratified k-fold cross-validation for multiple models."""
    models = [
        ('SVM (Linear)', SVC(kernel='linear', C=1.0, random_state=42)),
        ('SVM (RBF)', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)),
        ('Random Forest', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
        ('Gradient Boosting', GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)),
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('KNN (k=5)', KNeighborsClassifier(n_neighbors=5)),
    ]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scaler = StandardScaler()

    results = []

    print(f"\n{'='*60}")
    print(f"Stratified {n_folds}-Fold Cross-Validation")
    print(f"{'='*60}")

    for name, model in models:
        fold_scores = []
        fold_predictions = []
        fold_true = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Scale
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Handle NaN/Inf
            X_train_scaled = np.nan_to_num(X_train_scaled)
            X_val_scaled = np.nan_to_num(X_val_scaled)

            # Train and predict
            model.fit(X_train_scaled, y_train)
            pred = model.predict(X_val_scaled)

            fold_scores.append(accuracy_score(y_val, pred))
            fold_predictions.extend(pred)
            fold_true.extend(y_val)

        mean_acc = np.mean(fold_scores)
        std_acc = np.std(fold_scores)

        results.append({
            'model': name,
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'fold_scores': fold_scores,
            'predictions': fold_predictions,
            'true_labels': fold_true
        })

        print(f"\n{name}:")
        print(f"  Accuracy: {mean_acc:.1%} (+/- {std_acc:.1%})")
        print(f"  Fold scores: {[f'{s:.1%}' for s in fold_scores]}")

    return results


def train_final_model(X, y, model_class, **kwargs):
    """Train final model on full dataset."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled)

    model = model_class(**kwargs)
    model.fit(X_scaled, y)

    return model, scaler


def main():
    print("="*60)
    print("Keraal Full Dataset Training (Group2A - 51 samples)")
    print("="*60)

    # Load data
    data_path = Path("D:/keraal/processed/keraal_group2a_full.pkl")
    print(f"\nLoading data from {data_path}...")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    print(f"Loaded {len(samples)} samples")
    print(f"  Correct: {data['stats']['correct']}")
    print(f"  Error: {data['stats']['error']}")

    # Prepare features for each skeleton type
    skeleton_types = ['kinect', 'blazepose']

    all_results = {}

    for skel_type in skeleton_types:
        print(f"\n{'='*60}")
        print(f"Training with {skel_type.upper()} skeletons")
        print(f"{'='*60}")

        X, y, sample_ids = prepare_data(samples, skeleton_type=skel_type)
        print(f"Feature matrix shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y)} (0=Correct, 1=Error)")

        # Run cross-validation
        results = run_cross_validation(X, y, n_folds=5)
        all_results[skel_type] = results

    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    print(f"\n{'Model':<25} {'Kinect':>12} {'BlazePose':>12}")
    print("-"*50)

    # Get unique model names
    model_names = [r['model'] for r in all_results['kinect']]

    for model_name in model_names:
        kinect_result = next(r for r in all_results['kinect'] if r['model'] == model_name)
        blazepose_result = next(r for r in all_results['blazepose'] if r['model'] == model_name)

        kinect_acc = f"{kinect_result['mean_accuracy']:.1%}"
        blazepose_acc = f"{blazepose_result['mean_accuracy']:.1%}"

        print(f"{model_name:<25} {kinect_acc:>12} {blazepose_acc:>12}")

    # Find best model
    best_model = None
    best_acc = 0
    best_skel = None

    for skel_type, results in all_results.items():
        for r in results:
            if r['mean_accuracy'] > best_acc:
                best_acc = r['mean_accuracy']
                best_model = r['model']
                best_skel = skel_type

    print(f"\nBest Model: {best_model} with {best_skel.upper()} ({best_acc:.1%})")

    # Baseline
    majority_class = 0 if sum(y == 0) >= sum(y == 1) else 1
    baseline_acc = sum(y == majority_class) / len(y)
    print(f"Baseline (majority): {baseline_acc:.1%}")

    # Save results
    result_dir = Path("D:/keraal/results")
    result_dir.mkdir(exist_ok=True)

    with open(result_dir / 'full_dataset_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\nResults saved to {result_dir / 'full_dataset_results.pkl'}")

    return all_results


if __name__ == '__main__':
    main()
