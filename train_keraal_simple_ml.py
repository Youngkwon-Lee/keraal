"""
Keraal Simple ML Classifier

For small datasets (8 samples), use traditional ML instead of deep learning:
- Random Forest
- SVM
- Logistic Regression
- KNN

With statistical features extracted from skeleton sequences.
"""
import sys
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from keraal_converter import KeraalSample


def extract_statistical_features(skeleton: np.ndarray) -> np.ndarray:
    """
    Extract statistical features from skeleton sequence.

    Args:
        skeleton: (T, J, 3) skeleton sequence

    Returns:
        1D feature vector
    """
    T, J, C = skeleton.shape
    features = []

    # Flatten to (T, J*C)
    flat = skeleton.reshape(T, -1)

    # 1. Global statistics per joint coordinate
    features.extend(flat.mean(axis=0))      # Mean position
    features.extend(flat.std(axis=0))       # Std position
    features.extend(flat.max(axis=0))       # Max position
    features.extend(flat.min(axis=0))       # Min position
    features.extend(flat.max(axis=0) - flat.min(axis=0))  # Range

    # 2. Velocity statistics
    velocity = np.diff(flat, axis=0)
    if len(velocity) > 0:
        features.extend(velocity.mean(axis=0))
        features.extend(velocity.std(axis=0))
        features.extend(np.abs(velocity).max(axis=0))
    else:
        features.extend(np.zeros(J*C * 3))

    # 3. Acceleration statistics
    if len(velocity) > 1:
        acceleration = np.diff(velocity, axis=0)
        features.extend(acceleration.mean(axis=0))
        features.extend(acceleration.std(axis=0))
    else:
        features.extend(np.zeros(J*C * 2))

    # 4. Temporal features
    features.append(T)  # Sequence length

    # 5. Joint distance features (selected pairs)
    # Kinect: 0=SpineBase, 3=Head, 7=HandLeft, 11=HandRight
    if J >= 25:  # Kinect
        key_joints = [0, 3, 7, 11, 15, 19]  # Spine, Head, Hands, Feet
    else:  # BlazePose
        key_joints = [0, 11, 12, 15, 16, 23, 24]  # Nose, Shoulders, Wrists, Hips

    for i, j1 in enumerate(key_joints):
        for j2 in key_joints[i+1:]:
            if j1 < J and j2 < J:
                dist = np.linalg.norm(skeleton[:, j1, :] - skeleton[:, j2, :], axis=1)
                features.append(dist.mean())
                features.append(dist.std())
                features.append(dist.max())
                features.append(dist.min())

    # 6. Body symmetry features (left vs right)
    if J >= 25:  # Kinect
        left_joints = [4, 5, 6, 7, 12, 13, 14, 15]   # Left arm, leg
        right_joints = [8, 9, 10, 11, 16, 17, 18, 19]  # Right arm, leg

        for lj, rj in zip(left_joints, right_joints):
            if lj < J and rj < J:
                left_pos = skeleton[:, lj, :]
                right_pos = skeleton[:, rj, :]
                # Mirror right side for comparison
                right_mirrored = right_pos.copy()
                right_mirrored[:, 0] *= -1  # Flip x-axis

                symmetry_diff = np.linalg.norm(left_pos - right_mirrored, axis=1)
                features.append(symmetry_diff.mean())
                features.append(symmetry_diff.std())

    return np.array(features, dtype=np.float32)


def normalize_skeleton(skeleton: np.ndarray) -> np.ndarray:
    """Center and scale skeleton."""
    # Center to spine base (joint 0)
    center = skeleton[:, 0:1, :]
    centered = skeleton - center

    # Scale to unit bounding box
    flat = centered.reshape(-1, 3)
    scale = np.max(np.abs(flat)) + 1e-8
    normalized = centered / scale

    return normalized


def prepare_data(samples: List[KeraalSample], skeleton_type: str = 'kinect'):
    """Prepare feature matrix and labels."""
    X = []
    y = []
    sample_ids = []

    for sample in samples:
        # Get skeleton
        if skeleton_type == 'kinect' and sample.kinect is not None:
            skeleton = sample.kinect
        elif sample.blazepose is not None:
            skeleton = sample.blazepose
        else:
            skeleton = sample.kinect if sample.kinect is not None else sample.blazepose

        if skeleton is None:
            continue

        # Normalize
        skeleton = normalize_skeleton(skeleton)

        # Extract features
        features = extract_statistical_features(skeleton)

        X.append(features)
        y.append(0 if sample.is_correct else 1)  # 0=Correct, 1=Error
        sample_ids.append(sample.sample_id)

    return np.array(X), np.array(y), sample_ids


def leave_one_out_cv(X, y, sample_ids, model_class, model_name, **model_kwargs):
    """Leave-One-Out Cross-Validation."""
    n = len(y)
    predictions = []

    for i in range(n):
        # Split
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        X_test = X[i:i+1]
        y_test = y[i]

        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Handle NaN/Inf
        X_train_scaled = np.nan_to_num(X_train_scaled, nan=0, posinf=0, neginf=0)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)

        # Train
        model = model_class(**model_kwargs)
        model.fit(X_train_scaled, y_train)

        # Predict
        pred = model.predict(X_test_scaled)[0]
        predictions.append(pred)

    # Evaluate
    accuracy = accuracy_score(y, predictions)
    cm = confusion_matrix(y, predictions)

    return {
        'model': model_name,
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': y.tolist(),
        'confusion_matrix': cm,
        'sample_ids': sample_ids
    }


def main():
    print("="*60)
    print("Keraal Simple ML Classifier")
    print("="*60)

    # Load data
    data_path = Path("D:/keraal/processed/keraal_hawkeye_format.pkl")
    print(f"\nLoading data from {data_path}...")

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    print(f"Loaded {len(samples)} samples")

    # Prepare features
    print("\nExtracting features...")
    X, y, sample_ids = prepare_data(samples, skeleton_type='kinect')
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels: {y} (0=Correct, 1=Error)")

    # Define models to test
    models = [
        ('Random Forest', RandomForestClassifier, {'n_estimators': 10, 'max_depth': 3, 'random_state': 42}),
        ('SVM (RBF)', SVC, {'kernel': 'rbf', 'C': 1.0, 'random_state': 42}),
        ('SVM (Linear)', SVC, {'kernel': 'linear', 'C': 1.0, 'random_state': 42}),
        ('Logistic Regression', LogisticRegression, {'max_iter': 1000, 'random_state': 42}),
        ('KNN (k=3)', KNeighborsClassifier, {'n_neighbors': 3}),
        ('Gradient Boosting', GradientBoostingClassifier, {'n_estimators': 10, 'max_depth': 2, 'random_state': 42}),
    ]

    # Run experiments
    print("\n" + "="*60)
    print("Leave-One-Out Cross-Validation Results")
    print("="*60)

    results = []
    best_accuracy = 0
    best_model = None

    for model_name, model_class, model_kwargs in models:
        result = leave_one_out_cv(X, y, sample_ids, model_class, model_name, **model_kwargs)
        results.append(result)

        acc = result['accuracy']
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model_name

        # Print per-fold results
        print(f"\n{model_name}:")
        print(f"  Accuracy: {acc:.1%} ({int(acc*len(y))}/{len(y)})")

        # Detailed predictions
        label_names = {0: 'Correct', 1: 'Error'}
        for sid, true, pred in zip(sample_ids, y, result['predictions']):
            status = "OK" if true == pred else "X"
            print(f"    {sid[:35]:35s} | True: {label_names[true]:7s} | Pred: {label_names[pred]:7s} | {status}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'Accuracy':>10}")
    print("-"*37)
    for r in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{r['model']:<25} {r['accuracy']:>9.1%}")

    print(f"\nBest Model: {best_model} ({best_accuracy:.1%})")

    # Baseline comparison
    majority_class = 0 if sum(y == 0) >= sum(y == 1) else 1
    baseline_acc = sum(y == majority_class) / len(y)
    print(f"Baseline (majority): {baseline_acc:.1%}")
    print(f"Random baseline: 50.0%")

    # Save results
    result_dir = Path("D:/keraal/results")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(result_dir / 'simple_ml_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {result_dir / 'simple_ml_results.pkl'}")

    return results


if __name__ == '__main__':
    main()
