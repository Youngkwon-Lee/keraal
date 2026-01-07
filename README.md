# KERAAL Rehabilitation Exercise Classifier

재활 운동 동작의 정확성을 자동으로 평가하는 포즈 추정 기반 분류 시스템

## 프로젝트 개요

KERAAL(Kinect Exercise Recognition and Assessment for Assisted Living)은 EU ECHORD++ 프로젝트의 일환으로, 저요통 환자의 재활 운동을 지도하는 로봇 코치 개발을 위한 데이터셋입니다.

본 프로젝트는 KERAAL 데이터셋을 활용하여 **재활 운동의 정확/오류 여부를 자동으로 분류**하는 머신러닝 시스템을 구현합니다.

### 참여 기관 (원본 KERAAL 프로젝트)
- IMT Atlantique
- Generation Robots
- CHRU Brest

## 디렉토리 구조

```
D:\keraal\
├── keraal_converter.py        # 데이터 변환 파이프라인
├── train_keraal_classifier.py # LSTM 분류기 (LOOCV)
├── train_keraal_simple_ml.py  # 전통 ML 모델
├── train_full_dataset.py      # 전체 데이터셋 학습
├── train_balanced.py          # 클래스 불균형 보정 학습
├── convert_full_dataset.py    # 데이터셋 변환
├── analyze_errors.py          # 오류 유형별 분석
├── visualize_skeleton.py      # 스켈레톤 시각화
│
├── keraal_sample_2022/        # 원본 샘플 데이터
│   └── keraal_sample_2022/
│       ├── group1A/           # Annotation 포함 그룹
│       ├── group2A/           # Annotation 포함 그룹
│       └── group3/            # 파일명에 레이블 포함
│
├── annotatorA/                # 어노테이션 세트 A
├── annotatorB/                # 어노테이션 세트 B
├── downloads/                 # 다운로드 데이터
├── processed/                 # 전처리된 데이터
│   └── keraal_hawkeye_format.pkl
│
├── blazepose/                 # BlazePose 포즈 추정 결과
├── openpose/                  # OpenPose 포즈 추정 결과
├── kinect/                    # Kinect 센서 데이터
│
├── results/                   # 학습 결과
└── visualizations/            # 시각화 결과
```

## 데이터셋 구조

### 운동 유형
| 코드 | 운동명 | 설명 |
|------|--------|------|
| CTK | Chair-To-Stand Kinect | 의자에서 일어서기 |
| ELK | Extended Leg Kinect | 다리 뻗기 |
| RTK | Return-To-Chair Kinect | 의자로 돌아가기 |

### 레이블 체계
- **Correct**: 정상 수행
- **Error1**: 오류 유형 1
- **Error2**: 오류 유형 2
- **Error3**: 오류 유형 3

### 포즈 추정 포맷

| 소스 | 관절 수 | 차원 | 형식 |
|------|---------|------|------|
| Kinect | 25 | (T, 25, 3) 또는 (T, 25, 7) | TXT |
| BlazePose | 33 | (T, 33, 3) | JSON |
| OpenPose | 14 | (T, 14, 2) | JSON |
| Vicon | 17 | (T, 17, 3) 또는 (T, 17, 7) | TXT |

## 설치 및 실행

### 요구사항
```bash
pip install numpy torch scikit-learn matplotlib
```

### 데이터 변환
```bash
python keraal_converter.py
```

### 모델 학습

```bash
# LSTM 분류기 (Leave-One-Out CV)
python train_keraal_classifier.py

# 전통 ML 모델 (SVM, Random Forest)
python train_keraal_simple_ml.py

# 클래스 균형 학습
python train_balanced.py
```

### 오류 분석
```bash
python analyze_errors.py
```

### 스켈레톤 시각화
```bash
python visualize_skeleton.py
```

## 모델 아키텍처

### LSTM 분류기
```
Input (T, 25, 3) → LSTM (128 hidden, 2 layers) → FC → Binary Output
```

- **Hidden Size**: 128
- **Num Layers**: 2
- **Dropout**: 0.3
- **Validation**: Leave-One-Out Cross-Validation

### 전통 ML 모델
- **SVM (Linear)**: class_weight='balanced'
- **Random Forest**: 100 trees, max_depth=5

### 특징 추출
```python
# 통계적 특징 (mean, std, max, min)
# 속도 특징 (velocity mean, velocity std)
# 정규화: 중심점 기준 + 스케일 정규화
```

## 핵심 클래스

### KeraalSample
```python
@dataclass
class KeraalSample:
    sample_id: str
    group: str
    exercise: str           # CTK, ELK, RTK
    kinect: np.ndarray      # (T, 25, 3)
    blazepose: np.ndarray   # (T, 33, 3)
    is_correct: bool
    error_type: str         # Error1, Error2, Error3, none
    error_severity: str     # SmallError, BigError
    body_part: str          # BothArms, LeftArm, etc.
```

### KeraalConverter
```python
converter = KeraalConverter("path/to/data")
samples = converter.load_all_samples()
X, y, ids = converter.to_hawkeye_format(use_blazepose=False)
stats = converter.get_statistics()
```

## 관절 매핑

### Kinect 25 Joints
```
SpineBase(0), SpineMid(1), Neck(2), Head(3),
ShoulderLeft(4), ElbowLeft(5), WristLeft(6), HandLeft(7),
ShoulderRight(8), ElbowRight(9), WristRight(10), HandRight(11),
HipLeft(12), KneeLeft(13), AnkleLeft(14), FootLeft(15),
HipRight(16), KneeRight(17), AnkleRight(18), FootRight(19),
SpineShoulder(20), HandTipLeft(21), ThumbLeft(22), HandTipRight(23), ThumbRight(24)
```

### BlazePose 33 Joints
```
Nose(0), Left_eye_inner(1), Left_eye(2), Left_eye_outer(3),
Right_eye_inner(4), Right_eye(5), Right_eye_outer(6),
Left_ear(7), Right_ear(8), Mouth_left(9), Mouth_right(10),
Left_shoulder(11), Right_shoulder(12), Left_elbow(13), Right_elbow(14),
Left_wrist(15), Right_wrist(16), Left_pinky(17), Right_pinky(18),
Left_index(19), Right_index(20), Left_thumb(21), Right_thumb(22),
Left_hip(23), Right_hip(24), Left_knee(25), Right_knee(26),
Left_ankle(27), Right_ankle(28), Left_heel(29), Right_heel(30),
Left_foot_index(31), Right_foot_index(32)
```

## 참조

- **KERAAL 프로젝트**: https://keraal.enstb.org
- **EU ECHORD++**: European Clearing House for Open Robotics Development
- **데이터셋 출처**: IMT Atlantique, CHRU Brest

## 라이선스

연구 및 교육 목적으로만 사용 가능합니다.
