"""
Keraal Skeleton Visualizer
===========================
Keraal 데이터셋 스켈레톤 시각화

Author: PhysioKorea Team
Date: 2025-01-06
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import argparse
import sys

# Import KeraalSample from converter
sys.path.insert(0, str(Path(__file__).parent))
from keraal_converter import KeraalSample


# ============================================================
# Skeleton Connections
# ============================================================

# Kinect 25 joints connections
KINECT_CONNECTIONS = [
    # Spine
    (0, 1), (1, 20), (20, 2), (2, 3),  # SpineBase → SpineMid → SpineShoulder → Neck → Head
    # Left arm
    (20, 4), (4, 5), (5, 6), (6, 7),   # SpineShoulder → ShoulderL → ElbowL → WristL → HandL
    (7, 21), (7, 22),                   # HandL → HandTipL, ThumbL
    # Right arm
    (20, 8), (8, 9), (9, 10), (10, 11), # SpineShoulder → ShoulderR → ElbowR → WristR → HandR
    (11, 23), (11, 24),                 # HandR → HandTipR, ThumbR
    # Left leg
    (0, 12), (12, 13), (13, 14), (14, 15),  # SpineBase → HipL → KneeL → AnkleL → FootL
    # Right leg
    (0, 16), (16, 17), (17, 18), (18, 19),  # SpineBase → HipR → KneeR → AnkleR → FootR
]

# BlazePose 33 joints connections
BLAZEPOSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),  # Nose → Eyes
    (3, 7), (6, 8),  # Eyes → Ears
    (9, 10),  # Mouth
    # Body
    (11, 12),  # Shoulders
    (11, 13), (13, 15),  # Left arm
    (12, 14), (14, 16),  # Right arm
    (15, 17), (15, 19), (15, 21),  # Left hand
    (16, 18), (16, 20), (16, 22),  # Right hand
    (11, 23), (12, 24),  # Shoulders → Hips
    (23, 24),  # Hips
    (23, 25), (25, 27),  # Left leg
    (24, 26), (26, 28),  # Right leg
    (27, 29), (29, 31),  # Left foot
    (28, 30), (30, 32),  # Right foot
]

# Joint names
KINECT_JOINTS = [
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight"
]


# ============================================================
# Visualization Functions
# ============================================================

def plot_skeleton_frame(ax, skeleton, connections, frame_idx=0, title="", skeleton_type="kinect"):
    """단일 프레임 스켈레톤 시각화"""
    ax.clear()

    if skeleton_type == "kinect":
        # Kinect: x, y, z (depth camera coordinates)
        x = skeleton[frame_idx, :, 0]
        y = skeleton[frame_idx, :, 1]
        z = skeleton[frame_idx, :, 2]
    else:
        # BlazePose: x, y normalized (0-1), z is depth
        x = skeleton[frame_idx, :, 0]
        y = skeleton[frame_idx, :, 1]
        z = skeleton[frame_idx, :, 2]

    # Plot joints
    ax.scatter(x, -y, c='blue', s=50, zorder=5)

    # Plot connections
    for (i, j) in connections:
        if i < len(x) and j < len(x):
            ax.plot([x[i], x[j]], [-y[i], -y[j]], 'b-', linewidth=2, zorder=1)

    ax.set_title(f"{title}\nFrame {frame_idx+1}/{skeleton.shape[0]}")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if skeleton_type == "blazepose":
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 0)


def plot_skeleton_comparison(kinect, blazepose, frame_idx=0, title=""):
    """Kinect vs BlazePose 비교"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Kinect
    plot_skeleton_frame(axes[0], kinect, KINECT_CONNECTIONS, frame_idx,
                       f"Kinect (25 joints)", "kinect")

    # BlazePose
    plot_skeleton_frame(axes[1], blazepose, BLAZEPOSE_CONNECTIONS, frame_idx,
                       f"BlazePose (33 joints)", "blazepose")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_all_samples_grid(samples, skeleton_type="kinect"):
    """모든 샘플 그리드 시각화 (중간 프레임)"""
    n_samples = len(samples)
    cols = 4
    rows = (n_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
    axes = axes.flatten() if n_samples > 1 else [axes]

    connections = KINECT_CONNECTIONS if skeleton_type == "kinect" else BLAZEPOSE_CONNECTIONS

    for idx, sample in enumerate(samples):
        ax = axes[idx]

        if skeleton_type == "kinect" and sample.kinect is not None:
            skeleton = sample.kinect
        elif skeleton_type == "blazepose" and sample.blazepose is not None:
            skeleton = sample.blazepose
        else:
            ax.text(0.5, 0.5, "No data", ha='center', va='center')
            ax.set_title(sample.sample_id[:20])
            continue

        # 중간 프레임
        mid_frame = skeleton.shape[0] // 2

        if skeleton_type == "kinect":
            x = skeleton[mid_frame, :, 0]
            y = skeleton[mid_frame, :, 1]
        else:
            x = skeleton[mid_frame, :, 0]
            y = skeleton[mid_frame, :, 1]

        # Plot
        ax.scatter(x, -y, c='blue', s=20, zorder=5)
        for (i, j) in connections:
            if i < len(x) and j < len(x):
                ax.plot([x[i], x[j]], [-y[i], -y[j]], 'b-', linewidth=1, zorder=1)

        # Label color
        if sample.is_correct:
            label_color = 'green'
            label_text = 'Correct'
        else:
            label_color = 'red'
            label_text = f'{sample.error_type}'

        ax.set_title(f"{sample.exercise} | {label_text}", color=label_color, fontsize=10)
        ax.set_aspect('equal')
        ax.axis('off')

    # Hide empty axes
    for idx in range(n_samples, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f"Keraal Samples ({skeleton_type.upper()}) - Mid Frame", fontsize=14)
    plt.tight_layout()
    return fig


def plot_trajectory(samples, joint_idx=3, skeleton_type="kinect"):
    """관절 궤적 시각화 (예: Head)"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    joint_name = KINECT_JOINTS[joint_idx] if skeleton_type == "kinect" else f"Joint {joint_idx}"

    for idx, sample in enumerate(samples):
        if idx >= 8:
            break

        ax = axes[idx]

        if skeleton_type == "kinect" and sample.kinect is not None:
            skeleton = sample.kinect
        elif skeleton_type == "blazepose" and sample.blazepose is not None:
            skeleton = sample.blazepose
        else:
            continue

        # 관절 궤적
        x = skeleton[:, joint_idx, 0]
        y = skeleton[:, joint_idx, 1]
        t = np.arange(len(x))

        # 색상: 시간에 따라 변화
        scatter = ax.scatter(x, -y, c=t, cmap='viridis', s=10, alpha=0.7)
        ax.plot(x, -y, 'k-', alpha=0.3, linewidth=0.5)

        # 시작/끝 표시
        ax.scatter(x[0], -y[0], c='green', s=100, marker='o', zorder=10, label='Start')
        ax.scatter(x[-1], -y[-1], c='red', s=100, marker='x', zorder=10, label='End')

        # Label
        if sample.is_correct:
            label_color = 'green'
            label_text = 'Correct'
        else:
            label_color = 'red'
            label_text = sample.error_type

        ax.set_title(f"{sample.exercise} | {label_text}", color=label_color, fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"{joint_name} Trajectory Over Time ({skeleton_type.upper()})", fontsize=14)
    plt.tight_layout()
    return fig


def plot_joint_timeseries(samples, joint_idx=3, skeleton_type="kinect"):
    """관절 좌표 시계열 그래프"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    joint_name = KINECT_JOINTS[joint_idx] if skeleton_type == "kinect" else f"Joint {joint_idx}"
    coord_names = ['X', 'Y', 'Z']

    colors = {'Correct': 'green', 'Error1': 'red', 'Error2': 'orange', 'Error3': 'purple'}

    for sample in samples:
        if skeleton_type == "kinect" and sample.kinect is not None:
            skeleton = sample.kinect
        elif skeleton_type == "blazepose" and sample.blazepose is not None:
            skeleton = sample.blazepose
        else:
            continue

        label = 'Correct' if sample.is_correct else sample.error_type
        color = colors.get(label, 'blue')

        for coord_idx in range(3):
            values = skeleton[:, joint_idx, coord_idx]
            axes[coord_idx].plot(values, color=color, alpha=0.7,
                                label=f"{sample.exercise}-{label}" if coord_idx == 0 else "")

    for coord_idx in range(3):
        axes[coord_idx].set_ylabel(f'{coord_names[coord_idx]}')
        axes[coord_idx].grid(True, alpha=0.3)
        axes[coord_idx].legend(loc='upper right', fontsize=8)

    axes[2].set_xlabel('Frame')
    plt.suptitle(f"{joint_name} Coordinates Over Time ({skeleton_type.upper()})", fontsize=14)
    plt.tight_layout()
    return fig


def create_animation(skeleton, connections, output_path, skeleton_type="kinect", fps=15):
    """스켈레톤 애니메이션 생성"""
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame_idx):
        ax.clear()

        x = skeleton[frame_idx, :, 0]
        y = skeleton[frame_idx, :, 1]

        ax.scatter(x, -y, c='blue', s=50, zorder=5)
        for (i, j) in connections:
            if i < len(x) and j < len(x):
                ax.plot([x[i], x[j]], [-y[i], -y[j]], 'b-', linewidth=2, zorder=1)

        ax.set_title(f"Frame {frame_idx+1}/{skeleton.shape[0]}")
        ax.set_aspect('equal')

        if skeleton_type == "blazepose":
            ax.set_xlim(0, 1)
            ax.set_ylim(-1, 0)

        return ax,

    anim = FuncAnimation(fig, update, frames=skeleton.shape[0], interval=1000/fps, blit=False)
    anim.save(output_path, writer='pillow', fps=fps)
    plt.close()
    print(f"Saved animation to {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    """메인 시각화"""

    # 데이터 로드
    data_path = Path(r"D:\keraal\processed\keraal_hawkeye_format.pkl")
    output_dir = Path(r"D:\keraal\visualizations")
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Keraal Skeleton Visualizer")
    print("=" * 60)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    print(f"Loaded {len(samples)} samples")

    # 1. 모든 샘플 그리드 (Kinect)
    print("\n[1] Plotting all samples grid (Kinect)...")
    fig = plot_all_samples_grid(samples, skeleton_type="kinect")
    fig.savefig(output_dir / "all_samples_kinect.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 모든 샘플 그리드 (BlazePose)
    print("[2] Plotting all samples grid (BlazePose)...")
    fig = plot_all_samples_grid(samples, skeleton_type="blazepose")
    fig.savefig(output_dir / "all_samples_blazepose.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Head 궤적 비교
    print("[3] Plotting head trajectory...")
    fig = plot_trajectory(samples, joint_idx=3, skeleton_type="kinect")  # Head = index 3
    fig.savefig(output_dir / "head_trajectory_kinect.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. 시계열 비교 (SpineMid - 몸통 중심)
    print("[4] Plotting spine timeseries...")
    fig = plot_joint_timeseries(samples, joint_idx=1, skeleton_type="kinect")  # SpineMid
    fig.savefig(output_dir / "spine_timeseries_kinect.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Kinect vs BlazePose 비교 (첫 번째 샘플)
    print("[5] Plotting Kinect vs BlazePose comparison...")
    sample = samples[0]
    if sample.kinect is not None and sample.blazepose is not None:
        fig = plot_skeleton_comparison(
            sample.kinect, sample.blazepose,
            frame_idx=sample.kinect.shape[0]//2,
            title=f"{sample.sample_id} | {sample.exercise} | {'Correct' if sample.is_correct else sample.error_type}"
        )
        fig.savefig(output_dir / "kinect_vs_blazepose.png", dpi=150, bbox_inches='tight')
        plt.close()

    # 6. 애니메이션 (첫 번째 샘플)
    print("[6] Creating animation (first sample)...")
    if samples[0].kinect is not None:
        create_animation(
            samples[0].kinect,
            KINECT_CONNECTIONS,
            output_dir / "skeleton_animation.gif",
            skeleton_type="kinect",
            fps=15
        )

    print(f"\n[Done] Saved visualizations to {output_dir}")
    print("=" * 60)

    # 파일 목록 출력
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
