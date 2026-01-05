#!/usr/bin/env python3
"""
Generate Hugging Face dataset from edge-AI cough counting raw data.

Usage:
    # Full dataset
    python scripts/generate_hf_dataset.py --data-dir ./public_dataset --output-dir ./hf_dataset

    # Test mode (first 3 subjects)
    python scripts/generate_hf_dataset.py --data-dir ./public_dataset --output-dir ./hf_dataset_test --test-mode

    # Custom test subjects
    python scripts/generate_hf_dataset.py --data-dir ./public_dataset --output-dir ./hf_dataset_test --test-subjects 14287 14342 14547
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Audio, Sequence, Value, Array2D, DatasetDict
from tqdm import tqdm

# Import from existing helpers
sys.path.append('./src')
from helpers import Trial, Movement, Noise, Sound, load_audio, load_annotation, load_imu, FS_AUDIO, FS_IMU


class HFDatasetGenerator:
    """Generator for HF-format cough counting dataset."""

    def __init__(self, data_dir: str, output_dir: str, seed: int = 42):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.features = self._define_features()

    def _define_features(self) -> Features:
        """Define HF dataset schema."""
        return Features({
            "outward_facing_mic": Audio(sampling_rate=16000),
            "body_facing_mic": Audio(sampling_rate=16000),
            "imu": Array2D(shape=(None, 6), dtype="float32"),
            "ground_truth": {
                "start_times": Sequence(Value("float32")),
                "end_times": Sequence(Value("float32"))
            },
            "subject_id": Value("string"),
            "gender": Value("string"),
            "bmi": Value("float32"),
            "trial": Value("int8"),
            "movement": Value("string"),
            "noise": Value("string"),
            "sound": Value("string"),
            "duration_seconds": Value("float32"),
            "num_coughs": Value("int16"),
            "has_ground_truth": Value("bool"),
        })

    def get_all_subjects(self) -> List[str]:
        """Scan data directory for subject IDs."""
        subjects = []
        for item in self.data_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                subjects.append(item.name)
        return sorted(subjects)

    def create_subject_splits(self, subject_ids: List[str]) -> Dict[str, List[str]]:
        """Create deterministic train/val/test split."""
        np.random.seed(self.seed)
        shuffled_ids = np.array(sorted(subject_ids))
        np.random.shuffle(shuffled_ids)

        n = len(shuffled_ids)

        # Special handling for small numbers of subjects to ensure all splits get at least 1
        if n < 7:
            if n >= 3:
                # With 3-6 subjects: give 1 to val, 1 to test, rest to train
                train_end = max(1, n - 2)
                val_end = train_end + 1
            elif n == 2:
                # With 2 subjects: 1 train, 1 test, 0 val (val will be empty)
                train_end = 1
                val_end = 1
            else:  # n == 1
                # With 1 subject: put it in train
                train_end = 1
                val_end = 1
        else:
            # Normal split for 7+ subjects
            train_end = int(n * 0.70)
            val_end = train_end + int(n * 0.15)

        return {
            "train": shuffled_ids[:train_end].tolist(),
            "val": shuffled_ids[train_end:val_end].tolist(),
            "test": shuffled_ids[val_end:].tolist()
        }

    def load_biodata(self, subject_id: str) -> Dict:
        """Load subject metadata from biodata.json."""
        biodata_path = self.data_dir / subject_id / "biodata.json"
        with open(biodata_path, 'r') as f:
            return json.load(f)

    def process_recording(self, subject_id: str, trial: Trial, mov: Movement,
                         noise: Noise, sound: Sound, biodata: Dict) -> Optional[Dict]:
        """
        Process a single recording into HF format.

        Returns:
            Dict with all fields, or None if recording doesn't exist
        """
        # Check if recording exists
        rec_path = (self.data_dir / subject_id / f"trial_{trial.value}" /
                   f"mov_{mov.value}" / f"background_noise_{noise.value}" / sound.value)

        if not rec_path.exists():
            return None

        # Check if required files exist
        air_mic_path = rec_path / "outward_facing_mic.wav"
        skin_mic_path = rec_path / "body_facing_mic.wav"
        imu_path = rec_path / "imu.csv"

        if not air_mic_path.exists() or not skin_mic_path.exists() or not imu_path.exists():
            # Silently skip recordings with missing files
            return None

        try:
            # Load audio
            air_mic, skin_mic = load_audio(
                str(self.data_dir) + '/', subject_id, trial, mov, noise, sound
            )

            # Load IMU
            imu_obj = load_imu(
                str(self.data_dir) + '/', subject_id, trial, mov, noise, sound
            )

            # Stack IMU channels: accel x,y,z + gyro Y,P,R
            imu_array = np.stack([
                imu_obj.x, imu_obj.y, imu_obj.z,
                imu_obj.Y, imu_obj.P, imu_obj.R
            ], axis=1).astype(np.float32)

            # Load ground truth if cough recording
            if sound == Sound.COUGH:
                try:
                    start_times, end_times = load_annotation(
                        str(self.data_dir) + '/', subject_id, trial, mov, noise, sound
                    )
                    ground_truth = {
                        "start_times": start_times,
                        "end_times": end_times
                    }
                    num_coughs = len(start_times)
                    has_ground_truth = True
                except Exception:
                    # Cough recording without annotations (shouldn't happen but handle gracefully)
                    ground_truth = {"start_times": [], "end_times": []}
                    num_coughs = 0
                    has_ground_truth = False
            else:
                ground_truth = {"start_times": [], "end_times": []}
                num_coughs = 0
                has_ground_truth = False

            # Compute duration
            duration = len(air_mic) / FS_AUDIO

            # Build row
            row = {
                "outward_facing_mic": {
                    "path": str(rec_path / "outward_facing_mic.wav"),
                    "array": air_mic,
                    "sampling_rate": FS_AUDIO
                },
                "body_facing_mic": {
                    "path": str(rec_path / "body_facing_mic.wav"),
                    "array": skin_mic,
                    "sampling_rate": FS_AUDIO
                },
                "imu": imu_array,
                "ground_truth": ground_truth,
                "subject_id": subject_id,
                "gender": biodata["Gender"],
                "bmi": float(biodata["BMI"]),
                "trial": int(trial.value),
                "movement": mov.value,
                "noise": noise.value,
                "sound": sound.value,
                "duration_seconds": float(duration),
                "num_coughs": num_coughs,
                "has_ground_truth": has_ground_truth,
            }

            return row

        except Exception as e:
            print(f"Error processing {subject_id}/trial_{trial.value}/mov_{mov.value}/background_noise_{noise.value}/{sound.value}: {e}")
            return None

    def generate_split(self, subject_ids: List[str], split_name: str) -> Dataset:
        """Generate dataset for one split (train/val/test)."""
        rows = []
        total_attempts = 0
        skipped_count = 0

        print(f"\nProcessing {split_name} split ({len(subject_ids)} subjects)...")

        for subject_id in tqdm(subject_ids, desc=f"{split_name}"):
            # Load biodata once per subject
            biodata = self.load_biodata(subject_id)

            # Iterate through all 96 combinations
            for trial in Trial:
                for mov in Movement:
                    for noise in Noise:
                        for sound in Sound:
                            total_attempts += 1
                            row = self.process_recording(
                                subject_id, trial, mov, noise, sound, biodata
                            )
                            if row is not None:
                                rows.append(row)
                            else:
                                skipped_count += 1

        print(f"{split_name}: Collected {len(rows)} recordings")
        if skipped_count > 0:
            print(f"{split_name}: Skipped {skipped_count} recordings (missing files or errors)")

        # Create dataset - handle empty splits
        if len(rows) == 0:
            # For empty splits, create a dummy row then filter it out to get proper schema
            dummy_row = {
                "outward_facing_mic": {"path": "", "array": np.array([0], dtype=np.float32), "sampling_rate": 16000},
                "body_facing_mic": {"path": "", "array": np.array([0], dtype=np.float32), "sampling_rate": 16000},
                "imu": np.array([[0.0] * 6], dtype=np.float32),
                "ground_truth": {"start_times": [], "end_times": []},
                "subject_id": "_DUMMY_",
                "gender": "",
                "bmi": 0.0,
                "trial": 0,
                "movement": "",
                "noise": "",
                "sound": "",
                "duration_seconds": 0.0,
                "num_coughs": 0,
                "has_ground_truth": False,
            }
            dataset = Dataset.from_list([dummy_row], features=self.features)
            # Filter out the dummy row
            dataset = dataset.filter(lambda x: x["subject_id"] != "_DUMMY_")
        else:
            dataset = Dataset.from_list(rows, features=self.features)

        return dataset

    def generate(self, subjects: Optional[List[str]] = None):
        """
        Generate complete HF dataset.

        Args:
            subjects: If provided, only process these subjects (for test mode)
        """
        # Get subjects
        if subjects is None:
            subjects = self.get_all_subjects()
        else:
            subjects = sorted(subjects)

        print(f"Total subjects: {len(subjects)}")
        print(f"Subjects: {subjects}")

        # Create splits
        splits_map = self.create_subject_splits(subjects)

        # Save split mapping
        split_info = {
            "metadata": {
                "seed": self.seed,
                "train_ratio": 0.70,
                "val_ratio": 0.15,
                "test_ratio": 0.15,
                "total_subjects": len(subjects)
            },
            "splits": splits_map
        }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "split_info.json", 'w') as f:
            json.dump(split_info, f, indent=2)

        print(f"\nSplit distribution:")
        print(f"  Train: {len(splits_map['train'])} subjects - {splits_map['train']}")
        print(f"  Val:   {len(splits_map['val'])} subjects - {splits_map['val']}")
        print(f"  Test:  {len(splits_map['test'])} subjects - {splits_map['test']}")

        # Warn if using special split logic for small numbers
        if len(subjects) < 7:
            print(f"\n⚠️  NOTE: Using adjusted split ratios for {len(subjects)} subjects")
            print(f"   (Standard 70/15/15 split requires at least 7 subjects)")
            if len(subjects) < 3:
                print(f"   WARNING: Validation split will be empty with fewer than 3 subjects!")

        # Generate datasets - only include non-empty splits
        dataset_dict_data = {}
        for split_name in ["train", "val", "test"]:
            if len(splits_map[split_name]) > 0:
                dataset_dict_data[split_name] = self.generate_split(splits_map[split_name], split_name)
            else:
                print(f"\nSkipping {split_name} split (no subjects assigned)")

        dataset_dict = DatasetDict(dataset_dict_data)

        # Save to disk
        print(f"\nSaving to {self.output_dir}...")
        dataset_dict.save_to_disk(str(self.output_dir))

        # Also save individual parquet files for easier inspection
        for split_name, dataset in dataset_dict.items():
            parquet_path = self.output_dir / f"{split_name}.parquet"
            dataset.to_parquet(str(parquet_path))
            print(f"  {split_name}.parquet: {len(dataset)} rows")

        print("\nDataset generation complete!")
        return dataset_dict


def main():
    parser = argparse.ArgumentParser(
        description="Generate Hugging Face dataset from edge-AI cough counting data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./public_dataset",
        help="Path to raw dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./hf_dataset",
        help="Path to output directory for HF dataset"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Test mode: process only 3 subjects"
    )
    parser.add_argument(
        "--test-subjects",
        type=str,
        nargs='+',
        help="Specific subjects to process in test mode (e.g., 14287 14342 14547)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits"
    )

    args = parser.parse_args()

    # Initialize generator
    generator = HFDatasetGenerator(args.data_dir, args.output_dir, args.seed)

    # Determine subjects to process
    if args.test_subjects:
        subjects = args.test_subjects
        print(f"Test mode: Processing specified subjects {subjects}")
    elif args.test_mode:
        all_subjects = generator.get_all_subjects()
        subjects = all_subjects[:3]  # First 3 subjects
        print(f"Test mode: Processing first 3 subjects {subjects}")
    else:
        subjects = None  # Process all

    # Generate dataset
    generator.generate(subjects=subjects)


if __name__ == "__main__":
    main()
