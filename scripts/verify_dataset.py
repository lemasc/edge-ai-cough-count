#!/usr/bin/env python3
"""
Verify the integrity of the generated Hugging Face dataset.

Usage:
    python scripts/verify_dataset.py --dataset-dir ./hf_dataset
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict

from datasets import load_from_disk
import numpy as np


class DatasetVerifier:
    """Verifier for HF cough counting dataset."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(lambda: defaultdict(int))

    def log_error(self, message: str):
        """Log an error."""
        self.errors.append(f"ERROR: {message}")
        print(f"❌ {message}")

    def log_warning(self, message: str):
        """Log a warning."""
        self.warnings.append(f"WARNING: {message}")
        print(f"⚠️  {message}")

    def log_success(self, message: str):
        """Log a success."""
        print(f"✓ {message}")

    def verify_files_exist(self) -> bool:
        """Check that expected files exist."""
        print("\n" + "="*60)
        print("1. Checking file existence...")
        print("="*60)

        required_files = ["dataset_dict.json", "split_info.json"]
        all_exist = True

        for filename in required_files:
            filepath = self.dataset_dir / filename
            if filepath.exists():
                self.log_success(f"{filename} exists")
            else:
                self.log_error(f"{filename} not found")
                all_exist = False

        # Check parquet files
        for split in ["train", "val", "test"]:
            parquet_path = self.dataset_dir / f"{split}.parquet"
            if parquet_path.exists():
                self.log_success(f"{split}.parquet exists")
            else:
                self.log_warning(f"{split}.parquet not found (optional)")

        return all_exist

    def verify_split_integrity(self, dataset_dict) -> bool:
        """Verify no subject appears in multiple splits."""
        print("\n" + "="*60)
        print("2. Verifying split integrity (no data leakage)...")
        print("="*60)

        split_subjects = {}

        for split_name in ["train", "val", "test"]:
            subjects = set(dataset_dict[split_name]["subject_id"])
            split_subjects[split_name] = subjects
            print(f"{split_name}: {len(subjects)} unique subjects")

        # Check for overlap
        has_leakage = False

        # Train-Val overlap
        train_val_overlap = split_subjects["train"] & split_subjects["val"]
        if train_val_overlap:
            self.log_error(f"Train-Val overlap: {train_val_overlap}")
            has_leakage = True
        else:
            self.log_success("No Train-Val overlap")

        # Train-Test overlap
        train_test_overlap = split_subjects["train"] & split_subjects["test"]
        if train_test_overlap:
            self.log_error(f"Train-Test overlap: {train_test_overlap}")
            has_leakage = True
        else:
            self.log_success("No Train-Test overlap")

        # Val-Test overlap
        val_test_overlap = split_subjects["val"] & split_subjects["test"]
        if val_test_overlap:
            self.log_error(f"Val-Test overlap: {val_test_overlap}")
            has_leakage = True
        else:
            self.log_success("No Val-Test overlap")

        return not has_leakage

    def verify_schema(self, dataset_dict) -> bool:
        """Verify dataset schema matches expected."""
        print("\n" + "="*60)
        print("3. Verifying schema...")
        print("="*60)

        expected_columns = [
            "outward_facing_mic", "body_facing_mic", "imu", "ground_truth",
            "subject_id", "gender", "bmi", "trial", "movement", "noise", "sound",
            "duration_seconds", "num_coughs", "has_ground_truth"
        ]

        all_valid = True

        for split_name, dataset in dataset_dict.items():
            columns = dataset.column_names
            missing = set(expected_columns) - set(columns)
            extra = set(columns) - set(expected_columns)

            if missing:
                self.log_error(f"{split_name}: Missing columns {missing}")
                all_valid = False
            if extra:
                self.log_warning(f"{split_name}: Extra columns {extra}")

            if not missing and not extra:
                self.log_success(f"{split_name}: Schema matches")

        return all_valid

    def verify_data_integrity(self, dataset_dict) -> bool:
        """Verify data types, ranges, and consistency."""
        print("\n" + "="*60)
        print("4. Verifying data integrity...")
        print("="*60)

        all_valid = True

        for split_name, dataset in dataset_dict.items():
            print(f"\n{split_name} split:")

            # Sample first 10 rows
            sample_size = min(10, len(dataset))
            for i in range(sample_size):
                row = dataset[i]

                # Check outward mic
                if row["outward_facing_mic"]["sampling_rate"] != 16000:
                    self.log_error(
                        f"Row {i}: outward_facing_mic sampling rate = "
                        f"{row['outward_facing_mic']['sampling_rate']} (expected 16000)"
                    )
                    all_valid = False

                # Check body mic
                if row["body_facing_mic"]["sampling_rate"] != 16000:
                    self.log_error(
                        f"Row {i}: body_facing_mic sampling rate = "
                        f"{row['body_facing_mic']['sampling_rate']} (expected 16000)"
                    )
                    all_valid = False

                # Check IMU shape (IMU is stored as nested list when loaded from disk)
                imu_data = row["imu"]
                if isinstance(imu_data, list):
                    # Convert to numpy for shape check
                    imu_array = np.array(imu_data)
                    imu_shape = imu_array.shape
                else:
                    imu_shape = imu_data.shape

                if len(imu_shape) < 2 or imu_shape[1] != 6:
                    self.log_error(
                        f"Row {i}: IMU has shape {imu_shape} (expected (n, 6))"
                    )
                    all_valid = False

                # Check ground truth consistency
                if row["has_ground_truth"]:
                    if len(row["ground_truth"]["start_times"]) == 0:
                        self.log_warning(
                            f"Row {i}: has_ground_truth=True but empty annotations"
                        )
                    if row["num_coughs"] != len(row["ground_truth"]["start_times"]):
                        self.log_error(
                            f"Row {i}: num_coughs={row['num_coughs']} but "
                            f"{len(row['ground_truth']['start_times'])} annotations"
                        )
                        all_valid = False
                else:
                    if row["num_coughs"] != 0:
                        self.log_error(
                            f"Row {i}: has_ground_truth=False but num_coughs={row['num_coughs']}"
                        )
                        all_valid = False

            self.log_success(f"{split_name}: Sampled data integrity checks passed")

        return all_valid

    def compute_statistics(self, dataset_dict):
        """Compute and display dataset statistics."""
        print("\n" + "="*60)
        print("5. Computing statistics...")
        print("="*60)

        for split_name, dataset in dataset_dict.items():
            print(f"\n{split_name.upper()} Split:")
            print(f"  Total recordings: {len(dataset)}")

            # Count by subject
            subjects = set(dataset["subject_id"])
            print(f"  Unique subjects: {len(subjects)}")

            # Count cough recordings
            cough_recs = sum(1 for has_gt in dataset["has_ground_truth"] if has_gt)
            print(f"  Cough recordings: {cough_recs}")
            print(f"  Non-cough recordings: {len(dataset) - cough_recs}")

            # Total coughs
            total_coughs = sum(dataset["num_coughs"])
            print(f"  Total cough events: {total_coughs}")

            # Duration
            total_duration = sum(dataset["duration_seconds"])
            print(f"  Total duration: {total_duration/3600:.2f} hours")

            # Gender distribution
            genders = {}
            for gender in dataset["gender"]:
                genders[gender] = genders.get(gender, 0) + 1
            print(f"  Gender distribution: {genders}")

            # Movement distribution
            movements = {}
            for mov in dataset["movement"]:
                movements[mov] = movements.get(mov, 0) + 1
            print(f"  Movement distribution: {movements}")

            # Sound distribution
            sounds = {}
            for sound in dataset["sound"]:
                sounds[sound] = sounds.get(sound, 0) + 1
            print(f"  Sound type distribution: {sounds}")

        # Cross-split summary
        print("\n" + "="*60)
        print("OVERALL SUMMARY:")
        print("="*60)

        total_recordings = sum(len(dataset_dict[split]) for split in ["train", "val", "test"])
        total_subjects = len(set(
            subj for split in ["train", "val", "test"]
            for subj in dataset_dict[split]["subject_id"]
        ))
        total_coughs = sum(
            sum(dataset_dict[split]["num_coughs"])
            for split in ["train", "val", "test"]
        )
        total_hours = sum(
            sum(dataset_dict[split]["duration_seconds"])
            for split in ["train", "val", "test"]
        ) / 3600

        print(f"  Total subjects: {total_subjects}")
        print(f"  Total recordings: {total_recordings}")
        print(f"  Total cough events: {total_coughs}")
        print(f"  Total duration: {total_hours:.2f} hours")

    def verify(self) -> bool:
        """Run all verification checks."""
        print("\n" + "="*80)
        print(" DATASET VERIFICATION")
        print("="*80)

        # 1. Check files exist
        if not self.verify_files_exist():
            print("\n❌ File existence check failed")
            return False

        # Load dataset
        print(f"\nLoading dataset from {self.dataset_dir}...")
        try:
            dataset_dict = load_from_disk(str(self.dataset_dir))
        except Exception as e:
            self.log_error(f"Failed to load dataset: {e}")
            return False

        # 2. Verify split integrity
        split_ok = self.verify_split_integrity(dataset_dict)

        # 3. Verify schema
        schema_ok = self.verify_schema(dataset_dict)

        # 4. Verify data integrity
        data_ok = self.verify_data_integrity(dataset_dict)

        # 5. Compute statistics
        self.compute_statistics(dataset_dict)

        # Final report
        print("\n" + "="*80)
        print(" VERIFICATION REPORT")
        print("="*80)

        if self.errors:
            print(f"\n❌ {len(self.errors)} ERRORS:")
            for error in self.errors:
                print(f"  {error}")

        if self.warnings:
            print(f"\n⚠️  {len(self.warnings)} WARNINGS:")
            for warning in self.warnings:
                print(f"  {warning}")

        all_ok = split_ok and schema_ok and data_ok and len(self.errors) == 0

        if all_ok:
            print("\n✓ All verification checks passed!")
        else:
            print("\n❌ Verification failed - please review errors above")

        return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify Hugging Face dataset integrity"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="./hf_dataset",
        help="Path to HF dataset directory"
    )

    args = parser.parse_args()

    verifier = DatasetVerifier(args.dataset_dir)
    success = verifier.verify()

    exit(0 if success else 1)


if __name__ == "__main__":
    main()
