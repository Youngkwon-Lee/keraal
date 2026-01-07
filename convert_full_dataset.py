"""
Convert Full Keraal Dataset (Group2A)
51 annotated samples with BlazePose, Kinect, OpenPose skeletons
"""
import sys
import numpy as np
import pickle
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Data Classes
# ============================================================
@dataclass
class KeraalSample:
    """Keraal data sample"""
    sample_id: str
    group: str
    exercise: str  # CTK, ELK, RTK

    # Skeleton data
    kinect: Optional[np.ndarray] = None
    blazepose: Optional[np.ndarray] = None
    openpose: Optional[np.ndarray] = None

    # Labels
    is_correct: Optional[bool] = None
    error_type: Optional[str] = None
    error_severity: Optional[str] = None
    body_part: Optional[str] = None

    # Temporal info
    start_time: float = 0.0
    end_time: float = 0.0


# ============================================================
# Data Loaders
# ============================================================
def load_kinect_txt(filepath: str) -> np.ndarray:
    """Load Kinect skeleton from TXT file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            values = list(map(float, line.strip().split()))
            if len(values) == 175:  # 25 joints * 7 (x,y,z,qx,qy,qz,qw)
                frame = np.array(values).reshape(25, 7)
                data.append(frame[:, :3])  # Keep only x, y, z
            elif len(values) == 75:  # 25 joints * 3 (x,y,z)
                frame = np.array(values).reshape(25, 3)
                data.append(frame)
    return np.array(data) if data else None


def load_blazepose_json(filepath: str) -> np.ndarray:
    """Load BlazePose skeleton from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Handle different JSON structures
    if 'positions' in data:
        positions = data['positions']
    else:
        positions = data

    frames = sorted(positions.keys(), key=lambda x: float(x))

    BLAZEPOSE_JOINTS = [
        'Nose', 'LeftEyeInner', 'LeftEye', 'LeftEyeOuter',
        'RightEyeInner', 'RightEye', 'RightEyeOuter',
        'LeftEar', 'RightEar', 'MouthLeft', 'MouthRight',
        'LeftShoulder', 'RightShoulder', 'LeftElbow', 'RightElbow',
        'LeftWrist', 'RightWrist', 'LeftPinky', 'RightPinky',
        'LeftIndex', 'RightIndex', 'LeftThumb', 'RightThumb',
        'LeftHip', 'RightHip', 'LeftKnee', 'RightKnee',
        'LeftAnkle', 'RightAnkle', 'LeftHeel', 'RightHeel',
        'LeftFootIndex', 'RightFootIndex'
    ]

    skeleton_data = []
    for frame_key in frames:
        frame_data = positions[frame_key]
        joints = []
        for joint_name in BLAZEPOSE_JOINTS:
            if joint_name in frame_data:
                coords = frame_data[joint_name]
                if isinstance(coords, list) and len(coords) >= 3:
                    joints.append(coords[:3])
                else:
                    joints.append([0, 0, 0])
            else:
                joints.append([0, 0, 0])
        skeleton_data.append(joints)

    return np.array(skeleton_data, dtype=np.float32)


def load_openpose_json(filepath: str) -> np.ndarray:
    """Load OpenPose skeleton from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    if 'positions' in data:
        positions = data['positions']
    else:
        positions = data

    frames = sorted(positions.keys(), key=lambda x: float(x))

    OPENPOSE_JOINTS = [
        'Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
        'LShoulder', 'LElbow', 'LWrist', 'RHip', 'RKnee',
        'RAnkle', 'LHip', 'LKnee', 'LAnkle'
    ]

    skeleton_data = []
    for frame_key in frames:
        frame_data = positions[frame_key]
        joints = []
        for joint_name in OPENPOSE_JOINTS:
            if joint_name in frame_data:
                coords = frame_data[joint_name]
                if isinstance(coords, list) and len(coords) >= 2:
                    joints.append(coords[:2] + [0] if len(coords) == 2 else coords[:3])
                else:
                    joints.append([0, 0, 0])
            else:
                joints.append([0, 0, 0])
        skeleton_data.append(joints)

    return np.array(skeleton_data, dtype=np.float32)


def parse_anvil_annotation(filepath: str) -> Dict:
    """Parse Anvil XML annotation file."""
    tree = ET.parse(filepath)
    root = tree.getroot()

    annotations = {
        'is_correct': True,
        'error_type': None,
        'error_severity': None,
        'body_part': None,
        'start_time': 0.0,
        'end_time': 0.0
    }

    # Find annotation tracks
    for track in root.findall('.//track'):
        track_name = track.get('name', '')

        for el in track.findall('.//el'):
            start = float(el.get('start', 0))
            end = float(el.get('end', 0))

            if end > annotations['end_time']:
                annotations['end_time'] = end

            # Check for error annotations
            for attr in el.findall('.//attribute'):
                attr_name = attr.get('name', '')
                value = attr.text if attr.text else ''

                if 'evaluation' in track_name.lower() or 'eval' in attr_name.lower():
                    if 'incorrect' in value.lower() or 'error' in value.lower():
                        annotations['is_correct'] = False

                if 'errortype' in attr_name.lower() or 'error_type' in attr_name.lower():
                    annotations['error_type'] = value
                    annotations['is_correct'] = False

                if 'severity' in attr_name.lower():
                    annotations['error_severity'] = value

                if 'bodypart' in attr_name.lower() or 'body_part' in attr_name.lower():
                    annotations['body_part'] = value

    return annotations


def parse_sample_id(filename: str) -> Dict:
    """Parse sample ID to extract metadata."""
    # Format: G2A-{TYPE}-S{Subject}-{Location}-{Number}
    # Example: G2A-CTK-S1-Brest-029
    parts = filename.replace('.anvil', '').replace('.json', '').replace('.txt', '').split('-')

    result = {
        'group': 'G2A',
        'exercise': 'Unknown',
        'subject': 'Unknown',
        'location': 'Unknown',
        'number': '000'
    }

    if len(parts) >= 2:
        result['group'] = parts[0]

    # Find exercise type
    for part in parts:
        if part in ['CTK', 'ELK', 'RTK']:
            result['exercise'] = part
            break

    # Find subject
    for part in parts:
        if part.startswith('S') and len(part) <= 3:
            result['subject'] = part
            break

    # Location and number
    if len(parts) >= 4:
        result['location'] = parts[-2] if not parts[-2].isdigit() else 'Unknown'
        result['number'] = parts[-1]

    return result


# ============================================================
# Main Converter
# ============================================================
def convert_group2a(base_dir: Path) -> List[KeraalSample]:
    """Convert Group2A dataset to KeraalSample objects."""
    samples = []

    annotation_dir = base_dir / 'annotatorA'
    kinect_dir = base_dir / 'kinect'
    blazepose_dir = base_dir / 'blazepose'
    openpose_dir = base_dir / 'openpose'

    # Get all annotation files
    anvil_files = list(annotation_dir.glob('G2A-*.anvil'))
    print(f"Found {len(anvil_files)} annotation files")

    for anvil_file in anvil_files:
        sample_id = anvil_file.stem
        metadata = parse_sample_id(sample_id)

        # Parse annotations
        try:
            annotations = parse_anvil_annotation(str(anvil_file))
        except Exception as e:
            print(f"Warning: Could not parse {anvil_file}: {e}")
            annotations = {'is_correct': True, 'error_type': None}

        # Create sample
        sample = KeraalSample(
            sample_id=sample_id,
            group=metadata['group'],
            exercise=metadata['exercise'],
            is_correct=annotations['is_correct'],
            error_type=annotations['error_type'],
            error_severity=annotations.get('error_severity'),
            body_part=annotations.get('body_part'),
            start_time=annotations.get('start_time', 0),
            end_time=annotations.get('end_time', 0)
        )

        # Load Kinect
        kinect_pattern = f"G2A-Kinect-{metadata['exercise']}-{metadata['subject']}-*-{metadata['number']}.txt"
        kinect_files = list(kinect_dir.glob(kinect_pattern))
        if kinect_files:
            try:
                sample.kinect = load_kinect_txt(str(kinect_files[0]))
            except Exception as e:
                print(f"Warning: Kinect load failed for {sample_id}: {e}")

        # Load BlazePose
        bp_pattern = f"G2A-BP-{metadata['exercise']}-{metadata['subject']}-*-{metadata['number']}.json"
        bp_files = list(blazepose_dir.glob(bp_pattern))
        if bp_files:
            try:
                sample.blazepose = load_blazepose_json(str(bp_files[0]))
            except Exception as e:
                print(f"Warning: BlazePose load failed for {sample_id}: {e}")

        # Load OpenPose
        op_pattern = f"G2A-OP-{metadata['exercise']}-{metadata['subject']}-*-{metadata['number']}.json"
        op_files = list(openpose_dir.glob(op_pattern))
        if op_files:
            try:
                sample.openpose = load_openpose_json(str(op_files[0]))
            except Exception as e:
                print(f"Warning: OpenPose load failed for {sample_id}: {e}")

        # Only add if we have at least one skeleton
        if sample.kinect is not None or sample.blazepose is not None or sample.openpose is not None:
            samples.append(sample)

    return samples


def main():
    print("="*60)
    print("Keraal Full Dataset Converter (Group2A)")
    print("="*60)

    base_dir = Path("D:/keraal")
    output_dir = base_dir / "processed"
    output_dir.mkdir(exist_ok=True)

    # Convert
    print("\nConverting Group2A...")
    samples = convert_group2a(base_dir)
    print(f"Converted {len(samples)} samples")

    # Statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)

    # Label distribution
    correct_count = sum(1 for s in samples if s.is_correct)
    error_count = len(samples) - correct_count
    print(f"\nLabel Distribution:")
    print(f"  Correct: {correct_count}")
    print(f"  Error: {error_count}")

    # Exercise distribution
    exercise_counts = {}
    for s in samples:
        exercise_counts[s.exercise] = exercise_counts.get(s.exercise, 0) + 1
    print(f"\nExercise Distribution:")
    for ex, count in sorted(exercise_counts.items()):
        print(f"  {ex}: {count}")

    # Skeleton availability
    kinect_count = sum(1 for s in samples if s.kinect is not None)
    blazepose_count = sum(1 for s in samples if s.blazepose is not None)
    openpose_count = sum(1 for s in samples if s.openpose is not None)
    print(f"\nSkeleton Availability:")
    print(f"  Kinect: {kinect_count}")
    print(f"  BlazePose: {blazepose_count}")
    print(f"  OpenPose: {openpose_count}")

    # Frame statistics
    frame_lengths = []
    for s in samples:
        if s.kinect is not None:
            frame_lengths.append(len(s.kinect))
        elif s.blazepose is not None:
            frame_lengths.append(len(s.blazepose))

    if frame_lengths:
        print(f"\nFrame Statistics:")
        print(f"  Min: {min(frame_lengths)}")
        print(f"  Max: {max(frame_lengths)}")
        print(f"  Mean: {np.mean(frame_lengths):.1f}")

    # Save
    output_file = output_dir / "keraal_group2a_full.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump({
            'samples': samples,
            'stats': {
                'total': len(samples),
                'correct': correct_count,
                'error': error_count,
                'exercises': exercise_counts
            }
        }, f)
    print(f"\nSaved to {output_file}")

    return samples


if __name__ == '__main__':
    main()
