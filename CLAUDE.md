# CLAUDE.md - KERAAL Project

## Project Overview

**KERAAL**: Kinect Exercise Recognition and Assessment for Assisted Living
- 저요통 환자 재활 운동 분류 시스템
- PhysioKorea MLOps 연구 프로젝트

## Quick Start

```bash
# 환경 설정
conda activate keraal
cd ~/keraal

# 데이터 다운로드 (HPC)
bash scripts/hpc/download_data.sh quick  # 1.7GB (Group1A + 2A)

# 데이터 전처리
python keraal_converter.py

# LOSO 학습
python train_loso.py
```

## Directory Structure

```
keraal/
├── data/
│   ├── raw/              # 원본 데이터 (group1A, 2A, 3)
│   └── processed/        # 전처리된 pkl
├── models/trained/       # 학습된 모델
├── results/              # 학습 로그 및 결과
├── scripts/hpc/          # HPC 실행 스크립트
├── keraal_converter.py   # 데이터 변환
├── train_loso.py         # LOSO 학습 (메인)
├── train_keraal_*.py     # 기타 학습 스크립트
├── requirements.txt      # 의존성
├── HPC_SETUP.md          # HPC 환경 설정 가이드
└── BASELINE.md           # 베이스라인 연구 정리
```

## Key Files

| 파일 | 설명 |
|------|------|
| `train_loso.py` | **메인 학습 스크립트** (LOSO 프로토콜) |
| `keraal_converter.py` | 데이터 로더 및 변환기 |
| `BASELINE.md` | 기존 연구 및 성능 기준 |
| `HPC_SETUP.md` | HPC 환경 설정 가이드 |

## HPC Environment

```bash
# 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh
conda activate keraal

# GPU 사용
export CUDA_VISIBLE_DEVICES=0

# 학습 실행
nohup python train_loso.py > results/log.txt 2>&1 &
```

## Performance Targets (vs IJCNN 2024)

| 운동 | IJCNN Best | IJCNN Avg | 목표 |
|------|-----------|-----------|------|
| RTK | 64.4% | 53.9% | **65%+** |
| CTK | 56.2% | 49.1% | **57%+** |
| ELK | 43.0% | 31.6% | **44%+** |

## Git Workflow

```bash
# 로컬 → GitHub
git add . && git commit -m "message" && git push origin main

# HPC (git 안될 때)
wget -O train_loso.py "https://raw.githubusercontent.com/Youngkwon-Lee/keraal/main/train_loso.py"
```

## Links

- **Dataset**: https://keraal.enstb.org
- **Paper**: https://arxiv.org/html/2407.00521
- **PhysioKorea**: C:/Users/YK/physiokorea

## Data Pipeline

```
[Kinect/BlazePose] → [KeraalConverter] → [pkl] → [LOSO Training] → [Model]
     25/33 joints       keraal_converter.py    train_loso.py
```

## Evaluation Protocol

- **LOSO**: Leave-One-Subject-Out (가장 엄격)
- **Metrics**: Balanced Accuracy, F1 Score
- **Runs**: 5회 평균 ± 표준편차

---

*Last updated: 2026-01-07*
