#!/usr/bin/env python3
"""
SpaceNet Dataset Image-Label Matcher

This script matches satellite images (.tif) with their corresponding building
labels (.geojson) based on image identifiers extracted from filenames.

Intended for rooftop segmentation ML pipelines using SpaceNet dataset.
"""

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import random


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class FileMatch:
    """Represents a matched image-label pair."""
    image_id: str
    image_path: str
    label_path: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for JSON serialization."""
        return {
            "image": self.image_path,
            "label": self.label_path
        }


class DatasetMatcher:
    """
    Matches satellite images with their corresponding GeoJSON labels.

    Handles SpaceNet dataset naming conventions where images and labels share
    the same image identifier (e.g., img1, img1002) but have different prefixes.
    """

    # Regex pattern to extract image ID (e.g., img1, img1002) from filenames
    IMAGE_ID_PATTERN = re.compile(r'img\d+', re.IGNORECASE)

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        output_dir: Optional[Path] = None
    ):
        """
        Initialize the dataset matcher.

        Args:
            images_dir: Directory containing .tif image files
            labels_dir: Directory containing .geojson label files
            output_dir: Directory for output files (default: current directory)
        """
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()

        # Storage for results
        self.image_files: Dict[str, Path] = {}  # image_id -> path
        self.label_files: Dict[str, Path] = {}  # image_id -> path
        self.matched_pairs: List[FileMatch] = []
        self.missing_labels: List[str] = []  # image_ids without labels
        self.missing_images: List[str] = []  # image_ids without images
        self.invalid_files: List[Tuple[str, str]] = []  # (path, reason)

    def _extract_image_id(self, filename: str) -> Optional[str]:
        """
        Extract image identifier from filename.

        Expected format: SN2_buildings_train_AOI_2_Vegas_PS-RGB_img1.tif
        Extracts: img1

        Args:
            filename: The filename to parse

        Returns:
            The image identifier (e.g., 'img1') or None if not found
        """
        match = self.IMAGE_ID_PATTERN.search(filename)
        if match:
            return match.group(0).lower()
        return None

    def _get_file_extension(self, filename: str) -> str:
        """Get the file extension from filename."""
        return Path(filename).suffix.lower()

    def scan_directories(self) -> None:
        """
        Scan both directories and index files by image ID.

        Populates self.image_files and self.label_files dictionaries.
        """
        logger.info(f"Scanning images directory: {self.images_dir}")
        logger.info(f"Scanning labels directory: {self.labels_dir}")

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        # Scan image files
        image_count = 0
        invalid_image_count = 0
        for filepath in self.images_dir.iterdir():
            if not filepath.is_file():
                continue

            ext = self._get_file_extension(filepath.name)
            if ext not in {'.tif', '.tiff'}:
                continue

            image_id = self._extract_image_id(filepath.name)
            if image_id:
                if image_id in self.image_files:
                    logger.warning(
                        f"Duplicate image ID '{image_id}' found: {filepath} "
                        f"(already mapped to {self.image_files[image_id]})"
                    )
                    self.invalid_files.append((str(filepath), "duplicate_image_id"))
                else:
                    self.image_files[image_id] = filepath
                    image_count += 1
            else:
                logger.warning(f"Could not extract image ID from: {filepath.name}")
                self.invalid_files.append((str(filepath), "invalid_naming_format"))
                invalid_image_count += 1

        # Scan label files
        label_count = 0
        invalid_label_count = 0
        for filepath in self.labels_dir.iterdir():
            if not filepath.is_file():
                continue

            ext = self._get_file_extension(filepath.name)
            if ext != '.geojson':
                continue

            image_id = self._extract_image_id(filepath.name)
            if image_id:
                if image_id in self.label_files:
                    logger.warning(
                        f"Duplicate label ID '{image_id}' found: {filepath} "
                        f"(already mapped to {self.label_files[image_id]})"
                    )
                    self.invalid_files.append((str(filepath), "duplicate_label_id"))
                else:
                    self.label_files[image_id] = filepath
                    label_count += 1
            else:
                logger.warning(f"Could not extract image ID from: {filepath.name}")
                self.invalid_files.append((str(filepath), "invalid_naming_format"))
                invalid_label_count += 1

        logger.info(f"Found {image_count} images, {label_count} labels")
        if invalid_image_count or invalid_label_count:
            logger.warning(
                f"Skipped {invalid_image_count + invalid_label_count} files with invalid naming"
            )

    def validate_pair(self, image_path: Path, label_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate that both files exist and are non-empty.

        Args:
            image_path: Path to the image file
            label_path: Path to the label file

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check image file
        if not image_path.exists():
            return False, f"Image file does not exist: {image_path}"

        if image_path.stat().st_size == 0:
            return False, f"Image file is empty: {image_path}"

        # Check label file
        if not label_path.exists():
            return False, f"Label file does not exist: {label_path}"

        if label_path.stat().st_size == 0:
            return False, f"Label file is empty: {label_path}"

        return True, None

    def find_matches(self) -> None:
        """
        Match images with labels and validate pairs.

        Populates self.matched_pairs, self.missing_labels, and self.missing_images.
        """
        logger.info("Matching images with labels...")

        # Find matched pairs (images with corresponding labels)
        for image_id, image_path in self.image_files.items():
            if image_id in self.label_files:
                label_path = self.label_files[image_id]

                # Validate the pair
                is_valid, error_msg = self.validate_pair(image_path, label_path)

                if is_valid:
                    match = FileMatch(
                        image_id=image_id,
                        image_path=str(image_path),
                        label_path=str(label_path)
                    )
                    self.matched_pairs.append(match)
                else:
                    logger.error(f"Validation failed for {image_id}: {error_msg}")
                    self.invalid_files.append((str(image_path), error_msg))
            else:
                self.missing_labels.append(image_id)

        # Find images without corresponding labels
        for image_id in self.label_files:
            if image_id not in self.image_files:
                self.missing_images.append(image_id)

        # Sort pairs numerically by image_id
        def extract_number(image_id: str) -> int:
            """Extract numeric part from image_id for sorting."""
            match = re.search(r'\d+', image_id)
            return int(match.group()) if match else 0

        self.matched_pairs.sort(key=lambda x: extract_number(x.image_id))
        self.missing_labels.sort(key=extract_number)
        self.missing_images.sort(key=extract_number)

        logger.info(f"Found {len(self.matched_pairs)} matched pairs")
        if self.missing_labels:
            logger.warning(f"Found {len(self.missing_labels)} images without labels")
        if self.missing_images:
            logger.warning(f"Found {len(self.missing_images)} labels without images")

    def print_summary(self) -> None:
        """Print a summary of the matching results."""
        total_images = len(self.image_files)
        total_labels = len(self.label_files)
        matched_count = len(self.matched_pairs)
        missing_labels_count = len(self.missing_labels)
        missing_images_count = len(self.missing_images)
        invalid_count = len(self.invalid_files)

        print("\n" + "=" * 60)
        print("DATASET MATCHING SUMMARY")
        print("=" * 60)
        print(f"Total images scanned:       {total_images}")
        print(f"Total labels scanned:       {total_labels}")
        print(f"Matched pairs:              {matched_count}")
        print(f"Missing labels:             {missing_labels_count}")
        print(f"Missing images:             {missing_images_count}")
        print(f"Invalid/duplicate files:    {invalid_count}")
        print("=" * 60)

        if matched_count > 0:
            coverage = (matched_count / total_images) * 100 if total_images > 0 else 0
            print(f"Coverage: {coverage:.1f}% of images have matching labels")

    def save_results(self) -> None:
        """Save matching results to output files."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save matched pairs as JSON
        matched_path = self.output_dir / "matched_pairs.json"
        with open(matched_path, 'w', encoding='utf-8') as f:
            json.dump(
                [pair.to_dict() for pair in self.matched_pairs],
                f,
                indent=2,
                ensure_ascii=False
            )
        logger.info(f"Saved {len(self.matched_pairs)} matched pairs to {matched_path}")

        # Save missing labels
        if self.missing_labels:
            missing_labels_path = self.output_dir / "missing_labels.txt"
            with open(missing_labels_path, 'w', encoding='utf-8') as f:
                for image_id in self.missing_labels:
                    f.write(f"{image_id}\n")
            logger.info(f"Saved {len(self.missing_labels)} missing labels to {missing_labels_path}")

        # Save missing images
        if self.missing_images:
            missing_images_path = self.output_dir / "missing_images.txt"
            with open(missing_images_path, 'w', encoding='utf-8') as f:
                for image_id in self.missing_images:
                    f.write(f"{image_id}\n")
            logger.info(f"Saved {len(self.missing_images)} missing images to {missing_images_path}")

        # Save invalid files report
        if self.invalid_files:
            invalid_path = self.output_dir / "invalid_files.txt"
            with open(invalid_path, 'w', encoding='utf-8') as f:
                for filepath, reason in self.invalid_files:
                    f.write(f"{filepath}\t{reason}\n")
            logger.info(f"Saved {len(self.invalid_files)} invalid files to {invalid_path}")

    def print_random_samples(self, count: int = 3) -> None:
        """
        Print random sample pairs for visual sanity check.

        Args:
            count: Number of samples to display
        """
        if not self.matched_pairs:
            logger.warning("No matched pairs to sample from")
            return

        sample_count = min(count, len(self.matched_pairs))
        samples = random.sample(self.matched_pairs, sample_count)

        print("\n" + "-" * 60)
        print(f"RANDOM SAMPLE PAIRS (showing {sample_count} of {len(self.matched_pairs)})")
        print("-" * 60)

        for i, match in enumerate(samples, 1):
            print(f"\nSample {i}:")
            print(f"  Image ID: {match.image_id}")
            print(f"  Image:    {match.image_path}")
            print(f"  Label:    {match.label_path}")

            # Verify files exist
            img_exists = Path(match.image_path).exists()
            lbl_exists = Path(match.label_path).exists()
            print(f"  Status:   Image exists: {img_exists}, Label exists: {lbl_exists}")

        print("-" * 60)

    def run(self, sample_count: int = 3) -> Dict[str, any]:
        """
        Execute the full matching workflow.

        Args:
            sample_count: Number of random samples to display

        Returns:
            Dictionary with matching statistics
        """
        try:
            # Scan directories
            self.scan_directories()

            # Find matches
            self.find_matches()

            # Print summary
            self.print_summary()

            # Print random samples
            self.print_random_samples(sample_count)

            # Save results
            self.save_results()

            return {
                "total_images": len(self.image_files),
                "total_labels": len(self.label_files),
                "matched_pairs": len(self.matched_pairs),
                "missing_labels": len(self.missing_labels),
                "missing_images": len(self.missing_images),
                "invalid_files": len(self.invalid_files),
                "success": True
            }

        except Exception as e:
            logger.error(f"Error during matching: {e}")
            return {"success": False, "error": str(e)}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Match SpaceNet satellite images with their building labels",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("SN2_Vegas/PS-RGB"),
        help="Directory containing .tif image files"
    )

    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("SN2_Vegas/geojson_buildings"),
        help="Directory containing .geojson label files"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for output files"
    )

    parser.add_argument(
        "--sample-count",
        type=int,
        default=3,
        help="Number of random sample pairs to display"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create matcher and run
    matcher = DatasetMatcher(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir
    )

    results = matcher.run(sample_count=args.sample_count)

    # Exit with appropriate code
    sys.exit(0 if results.get("success") else 1)


if __name__ == "__main__":
    main()
