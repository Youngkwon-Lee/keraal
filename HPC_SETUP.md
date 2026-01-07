# KERAAL HPC Environment Setup

HPC 클러스터에서 KERAAL 재활 운동 분류 모델 학습 환경 구성 가이드

## 1. HPC 환경 개요

### 하드웨어 (Hawkeye 기준)
- **GPU**: Tesla V100 (또는 A100)
- **CUDA**: 11.8+
- **메모리**: GPU 16GB+

### 소프트웨어
- **OS**: Linux (CentOS/Ubuntu)
- **Python**: 3.10+
- **패키지 관리**: Miniconda3

---

## 2. 초기 설정

### 2.1 Miniconda 설치 (최초 1회)

```bash
# Miniconda 다운로드 및 설치
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3

# 환경 변수 설정
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# conda 초기화
conda init bash
source ~/.bashrc
```

### 2.2 KERAAL 가상환경 생성

```bash
# 가상환경 생성 (Python 3.10 + CUDA 지원 PyTorch)
conda create -n keraal python=3.10 -y
conda activate keraal

# PyTorch with CUDA (HPC에 맞는 버전 선택)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 필수 패키지
pip install -r requirements.txt
```

### 2.3 프로젝트 클론

```bash
# GitHub에서 클론
cd ~
git clone https://github.com/Youngkwon-Lee/keraal.git
cd keraal

# 또는 기존 저장소가 있으면
cd ~/keraal
git pull origin main
```

---

## 3. 데이터 다운로드

### 3.1 KERAAL 데이터셋 다운로드

**다운로드 URL**: https://keraal.enstb.org/KeraalDataset.html

```bash
# 데이터 디렉토리 생성
mkdir -p ~/keraal/data/raw
cd ~/keraal/data/raw

# Group1A (환자, 라벨 있음) - 1.4GB
wget -O group1A.tar.xz "https://keraal.enstb.org/downloads/group1A.tar.xz"

# Group2A (건강인, 라벨 있음) - 338MB
wget -O group2A.tar.xz "https://keraal.enstb.org/downloads/group2A.tar.xz"

# Group3 (오류 라벨 540개) - 8.2GB ⭐ 권장
wget -O group3.tar.xz "https://keraal.enstb.org/downloads/group3.tar.xz"

# 압축 해제
for f in *.tar.xz; do
    echo "Extracting $f..."
    tar -xJf $f
done
```

### 3.2 데이터셋 구조 (압축 해제 후)

```
~/keraal/data/raw/
├── group1A/           # 환자 데이터 (12명, 라벨 있음)
│   ├── annotator/     # .anvil 어노테이션 파일
│   ├── kinect/        # Kinect 스켈레톤 (25 joints)
│   ├── blazepose/     # BlazePose (33 joints)
│   ├── openpose/      # OpenPose (14 joints)
│   └── videos/        # 원본 비디오
│
├── group2A/           # 건강인 데이터 (6명, 라벨 있음)
│   └── (동일 구조)
│
└── group3/            # 오류 라벨 데이터 (540개)
    ├── kinect/        # 파일명에 라벨 포함
    ├── vicon/         # Vicon 모션캡처
    └── videos/
```

### 3.3 다운로드 우선순위

| 순위 | 그룹 | 용량 | 이유 |
|------|------|------|------|
| **1** | Group1A + Group2A | 1.7GB | 라벨 있음, 빠른 시작 |
| **2** | Group3 | 8.2GB | 540개 라벨, 학습용 |
| **3** | Group1B | 13GB | 환자 추가 데이터 |

---

## 4. 학습 실행

### 4.1 데이터 전처리

```bash
conda activate keraal
cd ~/keraal

# 데이터 변환 (Hawkeye 포맷)
python keraal_converter.py

# 결과: processed/keraal_hawkeye_format.pkl
```

### 4.2 LOSO 학습 실행

```bash
# GPU 할당
export CUDA_VISIBLE_DEVICES=0

# LOSO 학습 (LSTM baseline)
python train_loso.py

# 백그라운드 실행 (로그 저장)
nohup python train_loso.py > results/loso_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 4.3 HPC Job 제출 (SLURM 예시)

```bash
# scripts/hpc/train_loso.sh 참조
sbatch scripts/hpc/train_loso.sh
```

---

## 5. Git 워크플로우 (HPC)

### 5.1 HPC에서 Git 사용

**주의**: 일부 HPC는 SSH 포트 22 차단됨

```bash
# HTTPS로 클론 (SSH 대신)
git clone https://github.com/Youngkwon-Lee/keraal.git

# 또는 wget으로 개별 파일
wget -O train_loso.py "https://raw.githubusercontent.com/Youngkwon-Lee/keraal/main/train_loso.py"
```

### 5.2 결과 동기화

```bash
# HPC → Local (결과 다운로드)
scp user@hpc:~/keraal/results/*.pkl ./results/

# Local → HPC (코드 업데이트)
scp train_loso.py user@hpc:~/keraal/
```

---

## 6. 디렉토리 구조 (HPC)

```
~/keraal/                      # HPC 프로젝트 루트
├── data/
│   ├── raw/                   # 원본 데이터 (group1A, 2A, 3)
│   └── processed/             # 전처리된 pkl 파일
├── models/
│   └── trained/               # 학습된 모델 체크포인트
├── results/                   # 학습 로그 및 결과
├── scripts/
│   └── hpc/                   # HPC 실행 스크립트
├── keraal_converter.py        # 데이터 변환
├── train_loso.py              # LOSO 학습
├── requirements.txt           # 의존성
└── HPC_SETUP.md               # 이 문서
```

---

## 7. 트러블슈팅

### 7.1 CUDA Out of Memory

```bash
# 배치 사이즈 줄이기
python train_loso.py --batch_size 16

# 또는 train_loso.py Config 수정
BATCH_SIZE = 16  # 32에서 줄임
```

### 7.2 데이터 로딩 오류

```bash
# pickle 호환성 문제
python -c "
import sys
sys.path.insert(0, '.')
from keraal_converter import KeraalSample
import pickle
with open('processed/keraal_hawkeye_format.pkl', 'rb') as f:
    data = pickle.load(f)
print(f'Loaded {len(data[\"samples\"])} samples')
"
```

### 7.3 nohup 로그 확인

```bash
# 실시간 로그 확인
tail -f results/loso_*.log

# 프로세스 확인
ps aux | grep python
```

---

## 8. 권장 학습 설정

### 8.1 LSTM Baseline (IJCNN 2024 재현)

```python
# train_loso.py Config
EPOCHS = 1000
BATCH_SIZE = 32
LEARNING_RATE = 0.01
HIDDEN_SIZE = 128
NUM_LAYERS = 3
```

### 8.2 성능 목표

| 운동 | IJCNN Best | IJCNN Avg | 목표 |
|------|-----------|-----------|------|
| RTK | 64.4% | 53.9% | **65%+** |
| CTK | 56.2% | 49.1% | **57%+** |
| ELK | 43.0% | 31.6% | **44%+** |

---

## 9. 참고 자료

- **KERAAL Dataset**: https://keraal.enstb.org
- **IJCNN 2024 Paper**: https://arxiv.org/html/2407.00521
- **Hawkeye HPC 참조**: D:/Hawkeye/CLAUDE.md

---

*문서 작성일: 2026-01-07*
*PhysioKorea MLOps Team*
