"""
Keraal Dataset Converter
========================
Keraal 데이터셋을 Hawkeye 형식으로 변환하는 파이프라인

Author: PhysioKorea Team
Date: 2025-01-06
"""

import numpy as np
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pickle


# ============================================================
# Constants
# ============================================================

# Kinect 25 joints (순서 중요!)
KINECT_JOINTS = [
    "SpineBase", "SpineMid", "Neck", "Head",
    "ShoulderLeft", "ElbowLeft", "WristLeft", "HandLeft",
    "ShoulderRight", "ElbowRight", "WristRight", "HandRight",
    "HipLeft", "KneeLeft", "AnkleLeft", "FootLeft",
    "HipRight", "KneeRight", "AnkleRight", "FootRight",
    "SpineShoulder", "HandTipLeft", "ThumbLeft", "HandTipRight", "ThumbRight"
]

# BlazePose 33 joints
BLAZEPOSE_JOINTS = [
    "Nose", "Left_eye_inner", "Left_eye", "Left_eye_outer",
    "Right_eye_inner", "Right_eye", "Right_eye_outer",
    "Left_ear", "Right_ear", "Mouth_left", "Mouth_right",
    "Left_shoulder", "Right_shoulder", "Left_elbow", "Right_elbow",
    "Left_wrist", "Right_wrist", "Left_pinky", "Right_pinky",
    "Left_index", "Right_index", "Left_thumb", "Right_thumb",
    "Left_hip", "Right_hip", "Left_knee", "Right_knee",
    "Left_ankle", "Right_ankle", "Left_heel", "Right_heel",
    "Left_foot_index", "Right_foot_index"
]

# Kinect → Hawkeye 매핑 (유사 관절 매핑)
KINECT_TO_HAWKEYE_MAP = {
    0: 0,   # SpineBase → Hip center
    1: 1,   # SpineMid → Spine
    2: 2,   # Neck → Neck
    3: 3,   # Head → Head
    4: 11,  # ShoulderLeft → Left_shoulder
    5: 13,  # ElbowLeft → Left_elbow
    6: 15,  # WristLeft → Left_wrist
    8: 12,  # ShoulderRight → Right_shoulder
    9: 14,  # ElbowRight → Right_elbow
    10: 16, # WristRight → Right_wrist
    12: 23, # HipLeft → Left_hip
    13: 25, # KneeLeft → Left_knee
    14: 27, # AnkleLeft → Left_ankle
    16: 24, # HipRight → Right_hip
    17: 26, # KneeRight → Right_knee
    18: 28, # AnkleRight → Right_ankle
}


# ============================================================
# Data Classes
# ============================================================

@dataclass
class KeraalSample:
    """Keraal 데이터 샘플"""
    sample_id: str
    group: str
    exercise: str  # CTK, ELK, RTK

    # Skeleton data
    kinect: Optional[np.ndarray] = None      # (T, 25, 7) or (T, 25, 3)
    blazepose: Optional[np.ndarray] = None   # (T, 33, 3)
    openpose: Optional[np.ndarray] = None    # (T, 14, 2)
    vicon: Optional[np.ndarray] = None       # (T, 17, 7)

    # Labels
    is_correct: Optional[bool] = None
    error_type: Optional[str] = None         # Error1, Error2, Error3, none
    error_severity: Optional[str] = None     # SmallError, BigError
    body_part: Optional[str] = None          # BothArms, LeftArm, etc.

    # Temporal info
    start_time: float = 0.0
    end_time: float = 0.0


# ============================================================
# Loaders
# ============================================================

def load_kinect_txt(filepath: str, position_only: bool = True) -> np.ndarray:
    """
    Kinect TXT 파일 로드

    Args:
        filepath: Kinect txt 파일 경로
        position_only: True면 position(x,y,z)만, False면 quaternion 포함

    Returns:
        np.ndarray: (T, 25, 3) or (T, 25, 7)
    """
    data = np.loadtxt(filepath)
    num_frames = data.shape[0]

    # 175 values = 25 joints × 7 (x, y, z, qx, qy, qz, qw)
    data = data.reshape(num_frames, 25, 7)

    if position_only:
        # Position만 추출 (x, y, z)
        return data[:, :, :3]

    return data


def load_blazepose_json(filepath: str) -> np.ndarray:
    """
    BlazePose JSON 파일 로드

    Args:
        filepath: BlazePose json 파일 경로

    Returns:
        np.ndarray: (T, 33, 3)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # JSON 구조: {"positions": {"1.0": {"Nose": [x,y,z], ...}, "2.0": {...}}}
    if 'positions' in data:
        positions = data['positions']
    else:
        positions = data

    # 프레임 번호 정렬 (float 문자열 처리: "1.0", "2.0", ...)
    frames = sorted(positions.keys(), key=lambda x: float(x))
    num_frames = len(frames)

    skeleton = np.zeros((num_frames, 33, 3))

    for i, frame_key in enumerate(frames):
        frame_data = positions[frame_key]
        for j, joint_name in enumerate(BLAZEPOSE_JOINTS):
            if joint_name in frame_data:
                skeleton[i, j] = frame_data[joint_name]

    return skeleton


def load_openpose_json(filepath: str) -> np.ndarray:
    """
    OpenPose JSON 파일 로드

    Args:
        filepath: OpenPose json 파일 경로

    Returns:
        np.ndarray: (T, 14, 2)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    # JSON 구조: {"positions": {"1.0": {...}, "2.0": {...}}}
    if 'positions' in data:
        positions = data['positions']
    else:
        positions = data

    frames = sorted(positions.keys(), key=lambda x: float(x))
    num_frames = len(frames)

    # OpenPose COCO format: 14 joints
    openpose_joints = [
        "Head", "mShoulder", "rShoulder", "rElbow", "rWrist",
        "lShoulder", "lElbow", "lWrist", "rHip", "rKnee",
        "rAnkle", "lHip", "lKnee", "lAnkle"
    ]

    skeleton = np.zeros((num_frames, 14, 2))

    for i, frame_key in enumerate(frames):
        frame_data = positions[frame_key]
        for j, joint_name in enumerate(openpose_joints):
            if joint_name in frame_data:
                skeleton[i, j] = frame_data[joint_name][:2]

    return skeleton


def load_vicon_txt(filepath: str, position_only: bool = True) -> np.ndarray:
    """
    Vicon TXT 파일 로드

    Args:
        filepath: Vicon txt 파일 경로
        position_only: True면 position만

    Returns:
        np.ndarray: (T, 17, 3) or (T, 17, 7)
    """
    data = np.loadtxt(filepath)
    num_frames = data.shape[0]

    # 119 values = 17 targets × 7 (x, y, z, qx, qy, qz, qw)
    data = data.reshape(num_frames, 17, 7)

    if position_only:
        return data[:, :, :3]

    return data


def parse_anvil_annotation(filepath: str) -> Dict:
    """
    Anvil XML annotation 파싱

    Args:
        filepath: .anvil 파일 경로

    Returns:
        Dict: annotation 정보
    """
    tree = ET.parse(filepath)
    root = tree.getroot()

    result = {
        'is_correct': None,
        'error_type': None,
        'error_severity': None,
        'body_part': None,
        'start_time': 0.0,
        'end_time': 0.0
    }

    for track in root.iter('track'):
        track_name = track.attrib.get('name', '')

        if track_name == 'Global evaluation':
            el = track.find('el')
            if el is not None:
                result['start_time'] = float(el.attrib.get('start', 0))
                result['end_time'] = float(el.attrib.get('end', 0))

                attr = el.find('attribute')
                if attr is not None:
                    evaluation = attr.text
                    result['is_correct'] = (evaluation == 'Correct')

        elif track_name == 'Global error':
            el = track.find('el')
            if el is not None:
                for attr in el.findall('attribute'):
                    attr_name = attr.attrib.get('name', '')
                    if attr_name == 'type':
                        result['error_type'] = attr.text
                    elif attr_name == 'Evaluation':
                        result['error_severity'] = attr.text
                    elif attr_name == 'bodyPart':
                        result['body_part'] = attr.text

    return result


def parse_group3_filename(filename: str) -> Dict:
    """
    Group3 파일명에서 레이블 추출

    예: G3-Kinect-CTK-P1T1-Unknown-E1B1-0.txt
        → Error1, BodyPart1

    Args:
        filename: 파일명

    Returns:
        Dict: 레이블 정보
    """
    parts = filename.replace('.txt', '').replace('.json', '').split('-')

    result = {
        'is_correct': False,
        'error_type': None,
        'body_part': None,
        'exercise': None
    }

    # Exercise type (CTK, ELK, RTK)
    if len(parts) >= 3:
        result['exercise'] = parts[2]

    # Label (C=Correct, E1B1=Error1 BodyPart1)
    if len(parts) >= 6:
        label = parts[5]
        if label == 'C':
            result['is_correct'] = True
            result['error_type'] = 'none'
        elif label.startswith('E') and 'B' in label:
            result['is_correct'] = False
            # E1B1 → Error1, BodyPart1
            error_num = label.split('B')[0].replace('E', '')
            body_num = label.split('B')[1]
            result['error_type'] = f'Error{error_num}'
            result['body_part'] = f'BodyPart{body_num}'

    return result


# ============================================================
# Converter
# ============================================================

class KeraalConverter:
    """Keraal 데이터셋 변환기"""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.samples: List[KeraalSample] = []

    def load_all_samples(self) -> List[KeraalSample]:
        """모든 샘플 로드"""
        self.samples = []

        # Group1A, Group2A (annotated)
        for group in ['group1A', 'group2A']:
            self._load_annotated_group(group)

        # Group3 (label in filename)
        self._load_group3()

        print(f"Loaded {len(self.samples)} samples")
        return self.samples

    def _load_annotated_group(self, group: str):
        """Annotation이 있는 그룹 로드"""
        group_path = self.data_root / group
        if not group_path.exists():
            return

        annotator_path = group_path / 'annotator'
        if not annotator_path.exists():
            return

        for anvil_file in annotator_path.glob('*.anvil'):
            sample_id = anvil_file.stem

            # Parse annotation
            anno = parse_anvil_annotation(str(anvil_file))

            # Extract exercise type from filename
            # G1A-CTK-R1-Brest-022 → CTK
            parts = sample_id.split('-')
            exercise = parts[1] if len(parts) > 1 else 'Unknown'

            sample = KeraalSample(
                sample_id=sample_id,
                group=group,
                exercise=exercise,
                is_correct=anno['is_correct'],
                error_type=anno['error_type'],
                error_severity=anno['error_severity'],
                body_part=anno['body_part'],
                start_time=anno['start_time'],
                end_time=anno['end_time']
            )

            # Load skeleton data
            self._load_skeleton_data(sample, group_path, sample_id)

            self.samples.append(sample)

    def _load_group3(self):
        """Group3 로드 (파일명에 레이블)"""
        group_path = self.data_root / 'group3'
        if not group_path.exists():
            return

        kinect_path = group_path / 'kinect'
        if not kinect_path.exists():
            return

        for kinect_file in kinect_path.glob('*.txt'):
            filename = kinect_file.name
            sample_id = kinect_file.stem

            # Parse label from filename
            label = parse_group3_filename(filename)

            sample = KeraalSample(
                sample_id=sample_id,
                group='group3',
                exercise=label['exercise'],
                is_correct=label['is_correct'],
                error_type=label['error_type'],
                body_part=label['body_part']
            )

            # Load Kinect data
            sample.kinect = load_kinect_txt(str(kinect_file), position_only=True)

            # Load Vicon if exists
            vicon_file = group_path / 'vicon' / filename
            if vicon_file.exists():
                sample.vicon = load_vicon_txt(str(vicon_file), position_only=True)

            # Load BlazePose if exists
            bp_file = group_path / 'blazepose' / filename.replace('.txt', '.json').replace('Kinect', 'BP')
            if bp_file.exists():
                sample.blazepose = load_blazepose_json(str(bp_file))

            self.samples.append(sample)

    def _load_skeleton_data(self, sample: KeraalSample, group_path: Path, sample_id: str):
        """스켈레톤 데이터 로드"""
        # Kinect
        kinect_path = group_path / 'kinect'
        for kinect_file in kinect_path.glob(f'*{sample_id.split("-")[-1]}*.txt'):
            sample.kinect = load_kinect_txt(str(kinect_file), position_only=True)
            break

        # BlazePose
        bp_path = group_path / 'blazepose'
        for bp_file in bp_path.glob(f'*{sample_id.split("-")[-1]}*.json'):
            sample.blazepose = load_blazepose_json(str(bp_file))
            break

        # OpenPose
        op_path = group_path / 'openpose'
        for op_file in op_path.glob(f'*{sample_id.split("-")[-1]}*.json'):
            sample.openpose = load_openpose_json(str(op_file))
            break

    def to_hawkeye_format(self, use_blazepose: bool = True) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Hawkeye 형식으로 변환

        Args:
            use_blazepose: True면 BlazePose 사용, False면 Kinect 사용

        Returns:
            Tuple[X, y, sample_ids]:
                X: (N, T, J, 3) skeleton sequences
                y: (N,) labels (0=Correct, 1=Error1, 2=Error2, 3=Error3)
                sample_ids: 샘플 ID 리스트
        """
        X_list = []
        y_list = []
        ids_list = []

        # Label mapping
        label_map = {'none': 0, 'Error1': 1, 'Error2': 2, 'Error3': 3}

        for sample in self.samples:
            # Select skeleton
            if use_blazepose and sample.blazepose is not None:
                skeleton = sample.blazepose  # (T, 33, 3)
            elif sample.kinect is not None:
                skeleton = sample.kinect      # (T, 25, 3)
            else:
                continue

            # Get label
            if sample.is_correct:
                label = 0
            elif sample.error_type in label_map:
                label = label_map[sample.error_type]
            else:
                label = 0  # default to correct if unknown

            X_list.append(skeleton)
            y_list.append(label)
            ids_list.append(sample.sample_id)

        return X_list, np.array(y_list), ids_list

    def get_statistics(self) -> Dict:
        """데이터셋 통계"""
        stats = {
            'total_samples': len(self.samples),
            'by_group': {},
            'by_exercise': {},
            'by_label': {'Correct': 0, 'Error1': 0, 'Error2': 0, 'Error3': 0},
            'frame_lengths': []
        }

        for sample in self.samples:
            # By group
            stats['by_group'][sample.group] = stats['by_group'].get(sample.group, 0) + 1

            # By exercise
            stats['by_exercise'][sample.exercise] = stats['by_exercise'].get(sample.exercise, 0) + 1

            # By label
            if sample.is_correct:
                stats['by_label']['Correct'] += 1
            elif sample.error_type:
                stats['by_label'][sample.error_type] = stats['by_label'].get(sample.error_type, 0) + 1

            # Frame lengths
            if sample.kinect is not None:
                stats['frame_lengths'].append(sample.kinect.shape[0])
            elif sample.blazepose is not None:
                stats['frame_lengths'].append(sample.blazepose.shape[0])

        if stats['frame_lengths']:
            stats['avg_frames'] = np.mean(stats['frame_lengths'])
            stats['min_frames'] = np.min(stats['frame_lengths'])
            stats['max_frames'] = np.max(stats['frame_lengths'])

        return stats


# ============================================================
# Main
# ============================================================

def main():
    """샘플 데이터 변환 테스트"""

    # 샘플 데이터 경로
    data_root = Path(r"D:\keraal\keraal_sample_2022\keraal_sample_2022")

    print("=" * 60)
    print("Keraal → Hawkeye Converter")
    print("=" * 60)

    # 변환기 초기화
    converter = KeraalConverter(str(data_root))

    # 모든 샘플 로드
    samples = converter.load_all_samples()

    # 통계 출력
    stats = converter.get_statistics()
    print(f"\n[Statistics]")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  By group: {stats['by_group']}")
    print(f"  By exercise: {stats['by_exercise']}")
    print(f"  By label: {stats['by_label']}")
    if 'avg_frames' in stats:
        print(f"  Avg frames: {stats['avg_frames']:.1f}")
        print(f"  Min frames: {stats['min_frames']}")
        print(f"  Max frames: {stats['max_frames']}")

    # 샘플 상세 정보 출력
    print(f"\n[Sample Details]")
    for sample in samples:
        skeleton_info = []
        if sample.kinect is not None:
            skeleton_info.append(f"Kinect{sample.kinect.shape}")
        if sample.blazepose is not None:
            skeleton_info.append(f"BlazePose{sample.blazepose.shape}")
        if sample.vicon is not None:
            skeleton_info.append(f"Vicon{sample.vicon.shape}")

        label = "Correct" if sample.is_correct else f"{sample.error_type}/{sample.body_part}"
        print(f"  {sample.sample_id}: {sample.exercise} | {label} | {', '.join(skeleton_info)}")

    # Hawkeye 형식으로 변환
    print(f"\n[Converting to Hawkeye format...]")
    X_list, y, ids = converter.to_hawkeye_format(use_blazepose=False)

    print(f"  Converted {len(X_list)} samples")
    print(f"  Labels: {np.bincount(y)}")
    print(f"  Shapes: {[x.shape for x in X_list]}")

    # 저장
    output_path = Path(r"D:\keraal\processed")
    output_path.mkdir(exist_ok=True)

    output_file = output_path / "keraal_hawkeye_format.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({
            'X': X_list,
            'y': y,
            'sample_ids': ids,
            'samples': samples,
            'stats': stats
        }, f)

    print(f"\n[Saved to {output_file}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
