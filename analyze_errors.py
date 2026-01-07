"""
Analyze error type distribution and per-error-type accuracy
"""
import sys
import pickle
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from convert_full_dataset import KeraalSample


def extract_features(skeleton):
    """Extract statistical features."""
    T, J, C = skeleton.shape
    features = []

    # Center and normalize
    center = skeleton[:, 0:1, :]
    skeleton = skeleton - center
    scale = np.max(np.abs(skeleton)) + 1e-8
    skeleton = skeleton / scale

    flat = skeleton.reshape(T, -1)

    features.extend(flat.mean(axis=0))
    features.extend(flat.std(axis=0))
    features.extend(flat.max(axis=0))
    features.extend(flat.min(axis=0))

    velocity = np.diff(flat, axis=0)
    if len(velocity) > 0:
        features.extend(velocity.mean(axis=0))
        features.extend(velocity.std(axis=0))

    return np.array(features, dtype=np.float32)


def main():
    print("="*60)
    print("오류 유형별 분석")
    print("="*60)

    # Load
    with open('processed/keraal_group2a_full.pkl', 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']

    # 1. Distribution
    print("\n[1] 오류 유형별 분포")
    print("-"*40)

    error_details = []
    for s in samples:
        if s.is_correct:
            error_details.append(('Correct', s.exercise))
        else:
            et = s.error_type if s.error_type else 'Unknown'
            error_details.append((et, s.exercise))

    error_types = [e[0] for e in error_details]
    print(f"{'유형':<15} {'개수':>6}")
    print("-"*25)
    for et, count in sorted(Counter(error_types).items()):
        print(f"{et:<15} {count:>6}개")

    # 2. Per-error-type in binary classification
    print("\n[2] 이진 분류에서 오류 유형별 검출률")
    print("-"*40)

    # Prepare data
    X, y_binary, y_detail = [], [], []
    for s in samples:
        if s.kinect is None:
            continue
        features = extract_features(s.kinect)
        X.append(features)
        y_binary.append(0 if s.is_correct else 1)
        if s.is_correct:
            y_detail.append('Correct')
        else:
            y_detail.append(s.error_type if s.error_type else 'Unknown')

    X = np.array(X)
    y_binary = np.array(y_binary)
    y_detail = np.array(y_detail)

    # Train binary classifier and check per-error-type detection
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = np.nan_to_num(X_scaled)

    # Leave-one-out for detailed analysis
    predictions = []
    for i in range(len(X)):
        X_train = np.delete(X_scaled, i, axis=0)
        y_train = np.delete(y_binary, i)
        X_test = X_scaled[i:i+1]

        model = SVC(kernel='linear', C=1.0)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)[0]
        predictions.append(pred)

    predictions = np.array(predictions)

    # Per error type detection rate
    print(f"{'오류 유형':<15} {'총 개수':>8} {'검출 성공':>10} {'검출률':>10}")
    print("-"*50)

    for et in sorted(set(y_detail)):
        mask = y_detail == et
        total = mask.sum()

        if et == 'Correct':
            # For correct, prediction should be 0
            correct_pred = (predictions[mask] == 0).sum()
        else:
            # For errors, prediction should be 1
            correct_pred = (predictions[mask] == 1).sum()

        rate = correct_pred / total * 100
        print(f"{et:<15} {total:>8}개 {correct_pred:>10}개 {rate:>9.1f}%")

    # 3. Overall
    print("\n[3] 전체 성능")
    print("-"*40)
    overall_acc = (predictions == y_binary).sum() / len(y_binary) * 100
    print(f"전체 정확도: {overall_acc:.1f}%")

    # Confusion matrix
    print("\n혼동 행렬:")
    cm = confusion_matrix(y_binary, predictions)
    print(f"              예측 Correct  예측 Error")
    print(f"실제 Correct      {cm[0,0]:>6}       {cm[0,1]:>6}")
    print(f"실제 Error        {cm[1,0]:>6}       {cm[1,1]:>6}")

    # 4. Exercise-wise breakdown
    print("\n[4] 운동별 성능")
    print("-"*40)

    exercises = [e[1] for e in error_details]
    exercises = np.array(exercises[:len(predictions)])

    for ex in sorted(set(exercises)):
        mask = exercises == ex
        ex_acc = (predictions[mask] == y_binary[mask]).sum() / mask.sum() * 100
        print(f"{ex}: {ex_acc:.1f}% ({(predictions[mask] == y_binary[mask]).sum()}/{mask.sum()})")


if __name__ == '__main__':
    main()
