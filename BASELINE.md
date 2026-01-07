# KERAAL 베이스라인 연구 및 확장 논문 정리

## 1. 공식 베이스라인 (IJCNN 2024)

### 논문 정보
- **제목**: A Medical Low-Back Pain Physical Rehabilitation Dataset for Human Body Movement Analysis
- **저자**: Sao Mai Nguyen, Maxime Devanne, Olivier Remy-Neris, Mathieu Lempereur, Andre Thepaut
- **학회**: IJCNN 2024 (International Joint Conference on Neural Networks), Yokohama, Japan
- **링크**: [arXiv](https://arxiv.org/html/2407.00521) | [HAL](https://inria.hal.science/hal-04629541v1)

### 4가지 Challenge 정의

| Challenge | 태스크 | 설명 |
|-----------|--------|------|
| **1. Motion Assessment** | 이진 분류 | 정상(Correct) vs 오류(Incorrect) |
| **2. Error Classification** | 다중 분류 | 오류 유형 분류 (Error1, Error2, Error3) |
| **3. Spatial Localization** | 위치 추정 | 오류 발생 신체 부위 식별 |
| **4. Temporal Localization** | 시간 추정 | 오류 발생 시점 탐지 |

### 베이스라인 모델

#### GMM (Gaussian Mixture Model)
- **방식**: 확률 기반, 정상 시연만으로 학습
- **장점**: 빠른 학습 (수초), 라벨 불필요
- **단점**: 오류 검출 성능 낮음

#### LSTM (Long Short-Term Memory)
- **방식**: 시계열 딥러닝
- **장점**: 시간적 패턴 학습
- **단점**: 많은 데이터 필요

### Challenge 2 결과 (오류 유형 분류)

| 운동 | GMM+SVM | LSTM (평균±std) | LSTM (최고) |
|------|---------|-----------------|-------------|
| **Torso Rotation (RTK)** | 27.78% | 53.89±4.82% | **64.44%** |
| **Hiding Face (CTK)** | 33.33% | 49.10±3.28% | **56.19%** |
| **Flank Stretch (ELK)** | 25.32% | 31.64±6.48% | **43.03%** |

### 주요 발견
1. **오류 검출 어려움**: 대부분의 오류가 "정상"으로 오분류됨
2. **센서 비교**: Kinect ≈ OpenPose ≈ BlazePose (깊이 센서 이점 미미)
3. **라벨 품질**: 의사 간 일치도 Cohen's κ = 0.62

### GMM (Gaussian Mixture Model) 상세 설명

#### GMM이란?
**가우시안 혼합 모델(GMM)**은 데이터가 여러 개의 가우시안(정규) 분포의 혼합으로 생성되었다고 가정하는 확률 모델입니다.

```
P(x) = Σ πₖ · N(x | μₖ, Σₖ)

πₖ: k번째 가우시안의 가중치 (mixing coefficient)
μₖ: k번째 가우시안의 평균 (mean)
Σₖ: k번째 가우시안의 공분산 (covariance)
```

#### KERAAL에서 GMM 적용 방식

```
[정상 운동 데이터] → [GMM 학습] → [확률 분포 모델링]
                                         ↓
[새로운 운동] → [Log-Likelihood 계산] → [임계값 비교] → [정상/오류 판정]
```

**1단계: 학습 (정상 데이터만)**
- 정상적인 재활 운동 시퀀스로 GMM 파라미터 학습
- EM(Expectation-Maximization) 알고리즘 사용
- Riemannian Manifold 상에서 학습 (회전 데이터 처리)

**2단계: 추론**
- 새로운 운동의 Log-Likelihood 계산
- 정상 분포에서 벗어난 정도 측정
- 임계값 이하 → 오류로 판정

**3단계: 분류 (GMM+SVM)**
- GMM 출력을 특징으로 사용
- SVM으로 오류 유형 분류

#### GMM의 장단점 (재활 운동 평가)

| 장점 | 단점 |
|------|------|
| ✅ 정상 데이터만 필요 (오류 라벨 불필요) | ❌ 오류 검출 성능 낮음 (27-33%) |
| ✅ 빠른 학습 (수초) | ❌ 시간적 패턴 학습 어려움 |
| ✅ 해석 가능 (확률 기반) | ❌ 복잡한 동작 변이 처리 한계 |
| ✅ 실시간 추론 가능 | ❌ 고차원 데이터에 취약 |

#### 왜 GMM이 재활 평가에 사용되나?

1. **One-Class Classification**: 오류 데이터 수집이 어려운 상황
2. **이상 탐지 관점**: 정상 분포 학습 → 이상치 검출
3. **빠른 프로토타이핑**: 라벨링 없이 빠르게 시스템 구축

#### GMM vs 딥러닝 비교

| 측면 | GMM | LSTM/STGCN |
|------|-----|------------|
| **학습 데이터** | 정상만 | 정상+오류 |
| **학습 시간** | 수초 | 20분~수시간 |
| **성능** | 27-33% | 54-64% |
| **해석성** | 높음 | 낮음 |
| **실시간** | 가능 | 모델에 따라 |

---

## 1.1 연구 방법론 비교

### 데이터 분할 방식

| 연구 | 훈련 데이터 | 테스트 데이터 | 프로토콜 |
|------|------------|--------------|----------|
| **IJCNN 2024** | Group 3 (건강인 540개) | Group 1A+2A (환자+건강인) | Group 기반 |
| **GMM vs STGCN** | 다양한 크기 실험 | 나머지 | Few-shot 시나리오 |
| **DL Benchmark** | 건강인 전체 | 환자 1명씩 | **Leave-One-Subject-Out** |

### 실험 설정 비교

| 연구 | 에폭 | 배치 | 옵티마이저 | 하드웨어 |
|------|------|------|-----------|----------|
| **IJCNN 2024** | 1,000 | 32 | Adam (lr=0.01) | - |
| **GMM vs STGCN** | 250 | - | - | CPU i9-9900KF |
| **DL Benchmark** | 1,500 | 64 | ReduceLROnPlateau | RTX 4090 |

### 평가 메트릭

| 연구 | Challenge 1 | Challenge 2 | 비고 |
|------|-------------|-------------|------|
| **IJCNN 2024** | F1 Score | Accuracy (10회 평균) | 임계값 그리드 탐색 |
| **GMM vs STGCN** | F1 Score | Accuracy | 센서별 비교 |
| **DL Benchmark** | Balanced Acc | Balanced Acc | 클래스 불균형 보정 |

### 핵심 인사이트

```
1. 임상 시나리오 모방
   훈련: 건강인 데이터 (Group 2A, 3)
   테스트: 환자 데이터 (Group 1A)
   → 실제 상황: 건강인으로 학습 → 환자에게 적용

2. 가장 엄격한 평가: Leave-One-Subject-Out (LOSO)
   → 환자 1명씩 제외하고 학습 → 해당 환자로 테스트
   → 과적합 방지, 일반화 성능 측정

3. 클래스 불균형 처리
   Correct >> Error (약 4:1)
   → Balanced Accuracy 사용 권장

4. 센서 선택
   Kinect ≈ OpenPose ≈ BlazePose
   → RGB 카메라만으로 충분 (깊이 센서 불필요)
```

---

## 2. 확장 연구 논문

### 2.1 STGCN vs GMM 비교 연구 (2024)

- **제목**: Analyzing Data Efficiency and Performance of Machine Learning Algorithms for Assessing Low Back Pain Physical Rehabilitation Exercises
- **링크**: [arXiv](https://arxiv.org/html/2408.02855)

#### 비교 결과

| 측면 | GMM | STGCN |
|------|-----|-------|
| **학습 시간** | 수초 | 20-70분 |
| **데이터 효율** | 적은 샘플에서 우수 | 많은 샘플에서 우수 |
| **라벨 요구** | 정상 시연만 필요 | 정상/비정상 모두 필요 |
| **Few-shot 성능** | KIMORE에서 우수 | KERAAL에서 우수 |

#### 센서별 성능
```
Kinect ≈ OpenPose ≈ BlazePose
→ RGB 카메라만으로 충분한 성능 달성 가능
→ 깊이 카메라의 추가 이점 제한적
```

#### 주요 결론
- 대부분 상황에서 **STGCN 권장**
- 실시간/빠른 학습 필요시 GMM 사용
- 오류 데이터 수집 어려울 때 GMM 유용

---

### 2.2 Skeleton-Based Transformer (ICORR 2025)

- **제목**: Skeleton-Based Transformer for Classification of Errors and Better Feedback in Low Back Pain Physical Rehabilitation Exercises
- **저자**: Marusic, A., Nguyen, S. M., Tapus, A.
- **학회**: ICORR 2025 (International Conference on Rehabilitation Robotics)
- **링크**: [HAL](https://hal.science/hal-05000534)

#### 제안 모델
- HyperFormer 기반 Transformer 아키텍처
- 재활 운동 오류 분류에 특화

#### 결과
- KERAAL에서 **SOTA(State-of-the-Art) 달성**
- 기존 방법 대비 유의미한 성능 향상

---

### 2.3 PhysioFormer (2025)

- **제목**: PhysioFormer: A Spatio-Temporal Transformer for Physical Rehabilitation Assessment
- **링크**: [Springer](https://link.springer.com/chapter/10.1007/978-981-96-3525-2_14) | [HAL](https://hal.science/hal-04857956v1)

#### 제안 모델
- SkateFormer 기반 Transformer
- Skeletal-Temporal Self-Attention 활용
- 관절 간 관계 기반 그룹화

#### 평가 데이터셋
- KIMORE
- UI-PRMD
- **KERAAL**

#### 결과
- 세 데이터셋 모두에서 SOTA 달성

---

### ~~2.4 D-STGCNT (2024)~~ - KERAAL 미사용

> **주의**: 이 논문은 KERAAL을 사용하지 않았습니다. KIMORE, UI-PRMD만 사용.

- **제목**: D-STGCNT: A Dense Spatio-Temporal Graph Conv-GRU Network based on Transformer
- **링크**: [arXiv](https://arxiv.org/html/2401.06150)
- **사용 데이터셋**: KIMORE, UI-PRMD (KERAAL ❌)

---

### 2.5 Deep Learning Benchmark (2025)

- **제목**: Deep Learning for Skeleton Based Human Motion Rehabilitation Assessment: A Benchmark
- **링크**: [arXiv](https://arxiv.org/html/2507.21018v1)
- **GitHub**: https://github.com/MSD-IRIMAS/DeepRehabPile

#### 실험 방법론

**핵심: 각 데이터셋 독립 학습 (Transfer Learning 없음)**

```
60개 데이터셋 (분류 39개 + 회귀 21개)
    ↓
각 데이터셋마다 9개 모델 × 5회 반복 학습
    ↓
각 모델 from scratch 학습 (pre-training 없음)
    ↓
Multi-Comparison Matrix로 순위 통합
```

**학습 프로토콜**:
1. 데이터셋별 독립 실험 (한번에 학습 ❌)
2. 각 모델 5회 반복 (다른 random seed)
3. 1,500 epoch, batch 64 통일
4. Leave-One-Subject-Out 또는 5-fold CV
5. Wilcoxon Signed-Rank Test로 통계 검증

#### 비교 모델 (9개) - 성능 순위

| 순위 | 모델 | Accuracy | 구조 |
|------|------|----------|------|
| **1** | **LITEMV** | **77.37%** | Transformer (경량) |
| **2** | VanTran | ~77.22% | Vanilla Transformer |
| **3** | ConvTran | - | Conv + Transformer |
| **4** | MotionGRU | - | GRU 기반 |
| **5** | H-Inception | - | Inception 구조 |
| **6** | DisjointCNN | - | CNN |
| **7** | ConvLSTM | - | CNN + LSTM |
| **8** | FCN | - | Fully Conv |
| **9** | STGCN | - | Graph Conv (최하위) |

> ⚠️ STGCN이 최하위: "chance 수준보다 약간 나은 정도"

#### KERAAL 특징
- 이진 분류 + 다중 분류 = 6개 태스크 (3운동 × 2분류)
- LOSO 프로토콜 적용 (가장 엄격)
- 개별 수치 미공개 (39개 통합 결과만)

#### Rehab-Pile 데이터셋 목록 (60개)

**분류 데이터셋 (39개)**:
| 데이터셋 | 피험자 | 운동 | 샘플 | 라벨 | 센서 |
|----------|--------|------|------|------|------|
| **KERAAL** | 31명 | 3 | 2,622 | 오류유형 | Kinect v2 |
| **KIMORE** | 78명 | 5 | 355 | 점수→이진 | Kinect |
| **IRDS** | 29명 | 9 | 2,316 | 이진 | Kinect One |
| **UCDHE** | - | 2 | 4,289 | 이진/다중 | OpenPose |
| **KINECAL** | 90명 | 4 | 294 | 낙상위험 | Kinect |
| **SPHERE** | 6명 | 1 | 48 | 이상탐지 | - |

**회귀 데이터셋 (21개)**:
| 데이터셋 | 피험자 | 운동 | 샘플 | 라벨 | 센서 |
|----------|--------|------|------|------|------|
| **KIMORE** | 78명 | 5 | 355 | 0-100점 | Kinect |
| **UI-PRMD** | 10명 | 10 | 1,268 | 0-1점 | Vicon+Kinect |
| **EHE** | - | 6 | 869 | 알츠하이머 | Kinect |

#### KERAAL vs 주요 데이터셋

| 특징 | KERAAL | KIMORE | UI-PRMD | IRDS |
|------|--------|--------|---------|------|
| 환자 데이터 | ✅ 12명 | ✅ 34명 | ❌ | ✅ 15명 |
| 의사 라벨 | ✅ | ✅ | ❌ | ❌ |
| **오류 유형** | ✅ 유일 | ❌ | ❌ | ❌ |
| **시공간 위치** | ✅ 유일 | ❌ | ❌ | ❌ |
| 다중 센서 | ✅ 4종 | ✅ 3종 | ✅ 2종 | ❌ |

> **KERAAL 차별점**: 유일한 "오류 유형 + 시공간 위치" 라벨 데이터셋

---

## 3. KERAAL 성능 순위 (검증된 연구만)

### Challenge 2: 오류 유형 분류 (Error Classification)

| 순위 | 모델 | RTK | CTK | ELK | 평균 | 논문 |
|------|------|-----|-----|-----|------|------|
| **1** | **Skeleton Transformer** | - | - | - | **SOTA** | ICORR 2025 |
| **2** | LSTM (Best) | **64.4%** | **56.2%** | **43.0%** | **54.5%** | IJCNN 2024 |
| **3** | LSTM (Avg) | 53.9% | 49.1% | 31.6% | 44.9% | IJCNN 2024 |
| **4** | STGCN | >GMM | >GMM | >GMM | >GMM | 2024 |
| **5** | GMM+SVM | 27.8% | 33.3% | 25.3% | 28.8% | IJCNN 2024 |

> **참고**: Skeleton Transformer는 "SOTA"로 보고되었으나 구체적 수치 미공개

### Challenge 1: 동작 평가 (Motion Assessment)

| 모델 | 특징 | 문제점 |
|------|------|--------|
| STGCN | Few-shot에서도 GMM 대비 우수 | - |
| GMM | 빠른 학습, 라벨 불필요 | 대부분 "정상"으로 분류 |
| LSTM | 시간 패턴 학습 | 오류를 정상으로 오분류 |

### 센서별 성능 비교

```
Kinect (깊이) ≈ OpenPose (2D) ≈ BlazePose (3D 추정)
→ 결론: RGB 카메라만으로 충분, 깊이 센서 추가 이점 없음
```

### 연도별 발전

| 연도 | 주요 모델 | 특징 |
|------|-----------|------|
| 2024 | GMM, LSTM | 베이스라인 확립 |
| 2024 | STGCN | 그래프 신경망 도입 |
| 2025 | Skeleton Transformer | KERAAL SOTA |
| 2025 | PhysioFormer | 3개 데이터셋 SOTA |

---

## 3.1 참고 모델 상세 (KERAAL 미사용)

### D-STGCNT (2024) - KIMORE, UI-PRMD 전용

| 항목 | 내용 |
|------|------|
| **논문** | D-STGCNT: Dense Spatio-Temporal Graph Conv-GRU Network |
| **링크** | [arXiv](https://arxiv.org/html/2401.06150) |
| **데이터셋** | KIMORE, UI-PRMD (KERAAL ❌) |

**모델 구조**:
```
Input (Skeleton Sequence)
    ↓
[Dense STGC-GRU Block] ← 그래프 기반 공간 특성 + Dense Connection
    ↓
[Positional Encoding] ← 시간적 순서 정보 보존
    ↓
[Transformer Encoder] ← Multi-Head Self-Attention
    ↓
[Regression Head] → 운동 품질 점수 (0-50 또는 0-1)
```

**핵심 기여**:
| 기여 | 설명 |
|------|------|
| ConvGRU | LSTM 대비 계산 효율성 향상 |
| Dense Connection | 그래디언트 흐름 개선 |
| Multi-Head Attention | 관절별 중요도 학습 |
| 속도 | LSTM 대비 **5배 빠름** |

**KIMORE 결과 (Ex5)**:
| MAD | RMSE | MAPE |
|-----|------|------|
| 0.399 | 0.735 | 1.217% |

**UI-PRMD 결과 (평균)**:
| MAD | RMSE | MAPE |
|-----|------|------|
| 0.012 | 0.020 | 1.444% |

---

### PhysioFormer (2025) - KERAAL 언급

| 항목 | 내용 |
|------|------|
| **논문** | PhysioFormer: Spatio-Temporal Transformer for Physical Rehabilitation |
| **저자** | Marusic, Nguyen, Tapus |
| **링크** | [Springer](https://link.springer.com/chapter/10.1007/978-981-96-3525-2_14) |
| **데이터셋** | KIMORE, UI-PRMD, KERAAL (⚠️ 수치 미확인) |

**모델 구조**:
```
SkateFormer 기반 아키텍처
    ↓
[Skeletal-Temporal Self-Attention]
    ↓
[Joint Relation Grouping] ← 관절 간 관계 기반 그룹화
    ↓
[Quality Score Prediction]
```

**핵심 기여**:
| 기여 | 설명 |
|------|------|
| SkateFormer 적용 | Action Recognition → Rehabilitation |
| Joint Grouping | 해부학적 관계 기반 어텐션 |
| Multi-Dataset | 3개 벤치마크 통합 평가 |

**결과**: KIMORE, UI-PRMD, KERAAL 모두 SOTA (구체적 수치는 논문 참조)

---

## 4. 연구 로드맵 (PhysioKorea)

### 🎯 연구 목표
PhysioKorea 환자앱의 **홈 운동 자동 평가 시스템** 개발을 위한 기반 연구

### 📈 목표 성능

| Challenge | 현재 SOTA | 우리 목표 | 비고 |
|-----------|----------|----------|------|
| **Challenge 2 (RTK)** | 64.4% (LSTM) | **70%+** | 오류 유형 분류 |
| **Challenge 1** | ~80% (STGCN) | **85%+** | 정상/오류 이진 분류 |

---

### Phase 1: 베이스라인 재현 (1-2주)

**목표**: 논문 결과 재현 및 평가 파이프라인 구축

| # | 태스크 | 상태 | 산출물 |
|---|--------|------|--------|
| 1.1 | 전체 데이터셋 다운로드 (24GB) | ⬜ | `downloads/keraal_full/` |
| 1.2 | LOSO 평가 프로토콜 구현 | ⬜ | `evaluator.py` |
| 1.3 | LSTM 베이스라인 재현 | ⬜ | 목표: 64.4% (RTK) |
| 1.4 | 클래스 불균형 분석 | ⬜ | 분포 리포트 |
| 1.5 | 결과 비교 표 작성 | ⬜ | 논문 vs 우리 결과 |

**체크포인트**: LSTM RTK 60%+ 달성

---

### Phase 2: 모델 개선 (2-3주)

**목표**: 기존 SOTA 초과 또는 근접

| # | 태스크 | 상태 | 예상 성능 |
|---|--------|------|----------|
| 2.1 | STGCN 구현 | ⬜ | >GMM |
| 2.2 | 데이터 증강 (회전/스케일/노이즈) | ⬜ | +5~10% |
| 2.3 | Focal Loss 적용 | ⬜ | 불균형 해결 |
| 2.4 | Transformer 실험 (LITEMV/VanTran) | ⬜ | SOTA 도전 |
| 2.5 | 앙상블 모델 | ⬜ | 최종 성능 |

**모델 선택 가이드**:
```
[빠른 학습 필요] → GMM (수초)
[일반적 상황]    → STGCN (20분)
[최고 성능 필요] → Transformer (1시간+)
[모바일 배포]    → LITEMV (경량)
```

**체크포인트**: RTK 70%+ 달성

---

### Phase 3: PhysioKorea 연동 (2-3주)

**목표**: patient-app 실시간 운동 평가 기능 구현

| # | 태스크 | 상태 | 연동 대상 |
|---|--------|------|----------|
| 3.1 | MediaPipe 33 → KERAAL 25 관절 매핑 | ⬜ | `joint_mapper.py` |
| 3.2 | TFLite/ONNX 모델 변환 | ⬜ | `model.tflite` |
| 3.3 | patient-app 추론 연동 | ⬜ | React Native |
| 3.4 | 한국어 오류 피드백 | ⬜ | 오류별 메시지 |
| 3.5 | hosting 대시보드 연동 | ⬜ | 결과 시각화 |

**관절 매핑 (MediaPipe 33 → KERAAL 25)**:
```python
JOINT_MAP = {
    # MediaPipe → KERAAL (Kinect)
    23: 12,  # Left_hip → HipLeft
    24: 16,  # Right_hip → HipRight
    25: 13,  # Left_knee → KneeLeft
    26: 17,  # Right_knee → KneeRight
    11: 4,   # Left_shoulder → ShoulderLeft
    12: 8,   # Right_shoulder → ShoulderRight
    # ... 나머지 매핑
}
```

**체크포인트**: patient-app 실시간 추론 동작

---

### 💡 연구 차별점 (기존 연구 대비)

| 차별점 | 기존 연구 | PhysioKorea |
|--------|----------|-------------|
| **배포 환경** | 오프라인 실험 | 모바일 실시간 |
| **피드백 언어** | 영어/프랑스어 | **한국어** |
| **사용자** | 연구용 | 실제 환자 |
| **연동** | 독립 시스템 | 치료사 대시보드 통합 |

### 🔬 추가 연구 기회

1. **Challenge 3, 4**: 오류 발생 위치/시간 탐지 (미개척 분야)
2. **Multi-Dataset**: KIMORE, UI-PRMD 교차 검증
3. **PhysioKorea 자체 데이터**: 한국인 운동 데이터 수집
4. **Transfer Learning**: KERAAL → PhysioKorea 운동 전이 학습

---

## 5. 참고 자료

### 공식 리소스
- **데이터셋**: https://keraal.enstb.org/KeraalDataset.html
- **GitHub**: https://github.com/nguyensmai/KeraalDataset
- **STGCN 코드**: https://github.com/fokhruli/STGCN-rehab

### 주요 논문 링크
1. [KERAAL Dataset Paper (IJCNN 2024)](https://arxiv.org/html/2407.00521)
2. [GMM vs STGCN (2024)](https://arxiv.org/html/2408.02855)
3. [PhysioFormer (2025)](https://link.springer.com/chapter/10.1007/978-981-96-3525-2_14)
4. [Deep Learning Benchmark (2025)](https://arxiv.org/html/2507.21018v1)
5. [D-STGCNT (2024)](https://arxiv.org/html/2401.06150)

---

## 6. 논문 검증 상태

| 논문 | KERAAL 사용 | 검증 상태 |
|------|-------------|-----------|
| KERAAL Dataset (IJCNN 2024) | ✅ 원본 | 공식 베이스라인 |
| GMM vs STGCN (2024) | ✅ 직접 실험 | F1, Accuracy 결과 확인 |
| Skeleton Transformer (ICORR 2025) | ✅ 직접 실험 | SOTA, 오류 분류 |
| PhysioFormer (2025) | ⚠️ 언급됨 | 구체적 수치 미확인 |
| D-STGCNT (2024) | ❌ 미사용 | KIMORE, UI-PRMD만 |
| DL Benchmark (2025) | ✅ 직접 실험 | 6개 태스크 평가 |

### 검증 기준
- ✅ **직접 실험**: 논문에서 KERAAL 실험 결과 (정확도, F1 등) 명시
- ⚠️ **언급됨**: 데이터셋 목록에 포함되었으나 구체적 결과 미확인
- ❌ **미사용**: 다른 데이터셋만 사용

---

*문서 작성일: 2026-01-07*
*최종 검증일: 2026-01-07*
*PhysioKorea MLOps Team*
