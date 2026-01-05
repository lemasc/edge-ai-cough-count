#!/usr/bin/env python3
"""
Upload the generated dataset to Hugging Face Hub.

Prerequisites:
    - HF CLI login: huggingface-cli login
    - Or set HF_TOKEN environment variable

Usage:
    # Public dataset
    python scripts/upload_to_hf.py --dataset-dir ./hf_dataset --repo-id username/edge-ai-cough-count

    # Private dataset
    python scripts/upload_to_hf.py --dataset-dir ./hf_dataset --repo-id username/edge-ai-cough-count --private

    # With dataset card
    python scripts/upload_to_hf.py --dataset-dir ./hf_dataset --repo-id username/edge-ai-cough-count --card dataset_card_template.md
"""

import argparse
import os
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi, create_repo


class HFUploader:
    """Uploader for HF cough counting dataset."""

    def __init__(self, dataset_dir: str, repo_id: str, private: bool = False, token: str = None):
        self.dataset_dir = Path(dataset_dir)
        self.repo_id = repo_id
        self.private = private
        self.token = token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.token)

    def create_repository(self):
        """Create HF repository if it doesn't exist."""
        print(f"Creating repository: {self.repo_id} (private={self.private})")

        try:
            url = create_repo(
                repo_id=self.repo_id,
                repo_type="dataset",
                private=self.private,
                exist_ok=True,
                token=self.token
            )
            print(f"✓ Repository ready: {url}")
            return url
        except Exception as e:
            print(f"❌ Failed to create repository: {e}")
            raise

    def upload_dataset(self):
        """Upload dataset to HF Hub."""
        print(f"\nLoading dataset from {self.dataset_dir}...")

        try:
            dataset_dict = load_from_disk(str(self.dataset_dir))
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            raise

        print(f"Dataset loaded:")
        print(f"  Train: {len(dataset_dict['train'])} rows")
        print(f"  Val: {len(dataset_dict['val'])} rows")
        print(f"  Test: {len(dataset_dict['test'])} rows")

        print(f"\nUploading to {self.repo_id}...")

        try:
            dataset_dict.push_to_hub(
                repo_id=self.repo_id,
                private=self.private,
                token=self.token
            )
            print("✓ Dataset uploaded successfully!")
        except Exception as e:
            print(f"❌ Failed to upload dataset: {e}")
            raise

    def upload_card(self, card_path: str):
        """Upload dataset card (README.md) to repository."""
        card_file = Path(card_path)

        if not card_file.exists():
            print(f"⚠️  Dataset card not found: {card_path}")
            return

        print(f"\nUploading dataset card from {card_path}...")

        try:
            self.api.upload_file(
                path_or_fileobj=str(card_file),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                repo_type="dataset",
                token=self.token
            )
            print("✓ Dataset card uploaded!")
        except Exception as e:
            print(f"❌ Failed to upload card: {e}")
            raise

    def upload(self, card_path: str = None):
        """Execute full upload process."""
        print("="*80)
        print(" UPLOADING TO HUGGING FACE HUB")
        print("="*80)

        # Create repo
        repo_url = self.create_repository()

        # Upload dataset
        self.upload_dataset()

        # Upload card if provided
        if card_path:
            self.upload_card(card_path)
        else:
            print("\n⚠️  No dataset card provided (use --card to upload)")

        print("\n" + "="*80)
        print(" UPLOAD COMPLETE")
        print("="*80)
        print(f"\nDataset available at:")
        print(f"  https://huggingface.co/datasets/{self.repo_id}")
        print(f"\nLoad with:")
        print(f"  from datasets import load_dataset")
        print(f"  dataset = load_dataset('{self.repo_id}')")


def main():
    parser = argparse.ArgumentParser(
        description="Upload dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to HF dataset directory"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HF repository ID (e.g., username/dataset-name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--card",
        type=str,
        help="Path to dataset card (README.md)"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HF API token (or set HF_TOKEN env variable)"
    )

    args = parser.parse_args()

    uploader = HFUploader(
        dataset_dir=args.dataset_dir,
        repo_id=args.repo_id,
        private=args.private,
        token=args.token
    )

    uploader.upload(card_path=args.card)


if __name__ == "__main__":
    main()
