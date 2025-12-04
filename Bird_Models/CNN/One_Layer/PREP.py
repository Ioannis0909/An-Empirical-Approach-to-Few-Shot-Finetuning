"""
PREP.py - Data Preparation and Preprocessing
Handles all data loading, splitting, and episode sampling before model training.
"""

import os
import random
from typing import List, Tuple, Dict
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, IterableDataset, DataLoader
import math


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_and_split_classes(
    classes_file: str,
    train_ratio: float = 0.50,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Load classes from classes.txt and randomly split into train/val/test classes.

    Args:
        classes_file: Path to classes.txt file
        train_ratio: Proportion of classes for training (default 0.50 for 100/200)
        val_ratio: Proportion of classes for validation (default 0.25 for 50/200)
        test_ratio: Proportion of classes for testing (default 0.25 for 50/200)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_classes, val_classes, test_classes)
    """
    random.seed(seed)

    # Validate ratios
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    # Read classes from file
    all_classes = []
    with open(classes_file, 'r') as f:
        for line in f:
            # Parse format: "1 001.Black_footed_Albatross"
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                class_name = parts[1]
                all_classes.append(class_name)

    # Shuffle and split into 3 disjoint sets
    random.shuffle(all_classes)
    n_total = len(all_classes)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_classes = sorted(all_classes[:n_train])
    val_classes = sorted(all_classes[n_train:n_train + n_val])
    test_classes = sorted(all_classes[n_train + n_val:])

    print(f"\nTotal classes: {n_total}")
    print(f"Train classes: {len(train_classes)} ({train_ratio*100:.0f}%)")
    print(f"Val classes: {len(val_classes)} ({val_ratio*100:.0f}%)")
    print(f"Test classes: {len(test_classes)} ({test_ratio*100:.0f}%)")

    return train_classes, val_classes, test_classes


# ============================================================================
# DATA TRANSFORMS
# ============================================================================

def get_train_transforms() -> transforms.Compose:
    """
    Get image transformations for TRAINING with augmentation.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_eval_transforms() -> transforms.Compose:
    """
    Get image transformations for EVALUATION without augmentation.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


# ============================================================================
# DATASET SPLITTING
# ============================================================================

def load_all_images_for_classes(root_dir: str, classes: List[str]) -> Dict[str, List[str]]:
    """
    Load all images for given classes (no splitting).

    Args:
        root_dir: Path to images directory
        classes: List of class names

    Returns:
        Dictionary mapping class names to all image paths for that class
    """
    root = Path(root_dir)
    class_to_images = {}

    print(f"\nLoading {len(classes)} classes...")
    for class_name in classes:
        class_dir = root / class_name
        images = [str(img) for img in class_dir.glob('*') if img.is_file()]
        class_to_images[class_name] = images
        print(f"{class_name}: {len(images)} images")

    return class_to_images


def validate_class_splits(train_classes: List[str], val_classes: List[str], test_classes: List[str]):
    """
    Verify that class sets are disjoint and complete.

    Args:
        train_classes: List of training class names
        val_classes: List of validation class names
        test_classes: List of test class names

    Raises:
        AssertionError: If classes overlap or total count is incorrect
    """
    train_set = set(train_classes)
    val_set = set(val_classes)
    test_set = set(test_classes)

    # Check disjoint
    assert len(train_set & val_set) == 0, "Train and val classes overlap!"
    assert len(train_set & test_set) == 0, "Train and test classes overlap!"
    assert len(val_set & test_set) == 0, "Val and test classes overlap!"

    # Check totals
    total = len(train_classes) + len(val_classes) + len(test_classes)
    assert total == 200, f"Expected 200 total classes, got {total}"

    print(f"\n✓ Class splits validated:")
    print(f"  Train: {len(train_classes)} classes")
    print(f"  Val:   {len(val_classes)} classes")
    print(f"  Test:  {len(test_classes)} classes")
    print(f"  Total: {total} classes (all disjoint)")


# ============================================================================
# DATASET CLASS
# ============================================================================

class CaptchaDataset(Dataset):
    """Dataset for episodic sampling."""
    
    def __init__(self, class_to_images: Dict[str, List[str]], classes: List[str], 
                 transform=None):
        """
        Args:
            class_to_images: Dict mapping class names to list of image paths
            classes: Ordered list of class names
            transform: Optional transform
        """
        self.classes = classes
        self.class_to_images = class_to_images
        self.transform = transform
    
    def get_class_images(self, class_name: str) -> List[str]:
        """Get all image paths for a specific class."""
        return self.class_to_images.get(class_name, [])
    
    def load_image(self, img_path: str) -> torch.Tensor:
        """Load and transform a single image."""
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                dummy = Image.new('RGB', (224, 224), (0, 0, 0))
                return self.transform(dummy)
            else:
                return torch.zeros(3, 224, 224)


# ============================================================================
# EPISODE SAMPLER
# ============================================================================

class EpisodeSampler(IterableDataset):
    """Samples N-way K-shot episodes for prototypical networks."""

    def __init__(self,
                 dataset: CaptchaDataset,
                 n_way: int,
                 k_shot: int,
                 q_query: int,
                 episodes_per_epoch: int,
                 seed: int = None):
        """
        Args:
            dataset: CaptchaDataset instance
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            q_query: Number of query examples per class
            episodes_per_epoch: Number of episodes to sample per epoch
            seed: Random seed for reproducible episode sampling (optional)
        """
        super().__init__()

        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch
        self.seed = seed

        # Validate
        if n_way > len(dataset.classes):
            raise ValueError(
                f"n_way ({n_way}) cannot exceed number of classes ({len(dataset.classes)})"
            )

        for class_name in dataset.classes:
            images = dataset.get_class_images(class_name)
            if len(images) < k_shot + q_query:
                raise ValueError(
                    f"Class {class_name} has only {len(images)} images, "
                    f"need at least {k_shot + q_query} for {k_shot}-shot with {q_query} queries"
                )

    def sample_episode(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample one episode.

        Returns:
            support_images: (n_way * k_shot, C, H, W)
            query_images: (n_way * q_query, C, H, W)
            query_labels: (n_way * q_query,) with labels in [0, n_way-1]
        """
        # Sample N classes
        sampled_classes = random.sample(self.dataset.classes, self.n_way)
        
        support_images = []
        query_images = []
        query_labels = []
        
        for episode_label, class_name in enumerate(sampled_classes):
            class_images = self.dataset.get_class_images(class_name)
            sampled_images = random.sample(class_images, self.k_shot + self.q_query)
            
            support_imgs = sampled_images[:self.k_shot]
            query_imgs = sampled_images[self.k_shot:]
            
            for img_path in support_imgs:
                support_images.append(self.dataset.load_image(img_path))
            
            for img_path in query_imgs:
                query_images.append(self.dataset.load_image(img_path))
                query_labels.append(episode_label)
        
        support_images = torch.stack(support_images)
        query_images = torch.stack(query_images)
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        
        return support_images, query_images, query_labels

    def __iter__(self):
        """
        IterableDataset interface.
        Handles multi-worker splits so total episodes_per_epoch
        is respected across all workers.
        """
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process data loading
            start = 0
            end = self.episodes_per_epoch
            worker_id = 0
        else:
            # Split episodes across workers
            per_worker = math.ceil(self.episodes_per_epoch / worker_info.num_workers)
            start = worker_info.id * per_worker
            end = min(start + per_worker, self.episodes_per_epoch)
            worker_id = worker_info.id

        # Set seed for reproducible episode sampling (only used for test evaluation)
        if self.seed is not None:
            random.seed(self.seed + worker_id)

        for _ in range(start, end):
            yield self.sample_episode()

    def __len__(self):
        return self.episodes_per_epoch


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def prepare_training_data(
    data_root: str,
    classes_file: str,
    train_ratio: float = 0.50,
    val_ratio: float = 0.25,
    test_ratio: float = 0.25,
    seed: int = 42,
    num_workers: int = 4
):
    """
    Prepare training, validation, and test data with disjoint class sets.

    Args:
        data_root: Path to images directory (e.g., 'CUB_200_2011/CUB_200_2011/images')
        classes_file: Path to classes.txt file
        train_ratio: Proportion of classes for training (default 0.50 for 100/200)
        val_ratio: Proportion of classes for validation (default 0.25 for 50/200)
        test_ratio: Proportion of classes for testing (default 0.25 for 50/200)
        seed: Random seed
        num_workers: Number of parallel workers for data loading

    Returns:
        train_dataset, val_dataset, test_dataset, num_workers
    """
    # Load and split classes into 3 disjoint sets
    train_classes, val_classes, test_classes = load_and_split_classes(
        classes_file,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    # Validate that splits are disjoint
    validate_class_splits(train_classes, val_classes, test_classes)

    # Load all images for each class set
    train_class_to_images = load_all_images_for_classes(data_root, train_classes)
    val_class_to_images = load_all_images_for_classes(data_root, val_classes)
    test_class_to_images = load_all_images_for_classes(data_root, test_classes)

    # Create transforms
    train_transform = get_train_transforms()
    eval_transform = get_eval_transforms()

    # Create datasets
    train_dataset = CaptchaDataset(
        class_to_images=train_class_to_images,
        classes=train_classes,
        transform=train_transform
    )

    val_dataset = CaptchaDataset(
        class_to_images=val_class_to_images,
        classes=val_classes,
        transform=eval_transform
    )

    test_dataset = CaptchaDataset(
        class_to_images=test_class_to_images,
        classes=test_classes,
        transform=eval_transform
    )

    return train_dataset, val_dataset, test_dataset, num_workers


def prepare_test_data(data_root: str, classes_file: str, seed: int = 42):
    """
    Prepare test class data for final evaluation.

    Args:
        data_root: Path to images directory (e.g., 'CUB_200_2011/CUB_200_2011/images')
        classes_file: Path to classes.txt file
        seed: Random seed (must match the seed used in prepare_training_data)

    Returns:
        test_dataset
    """
    # Load and split classes
    _, _, test_classes = load_and_split_classes(
        classes_file,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=seed
    )

    # Load test classes
    class_to_images = load_all_images_for_classes(
        root_dir=data_root,
        classes=test_classes
    )

    # Create transform (no augmentation for evaluation)
    eval_transform = get_eval_transforms()

    # Create dataset
    test_dataset = CaptchaDataset(
        class_to_images=class_to_images,
        classes=test_classes,
        transform=eval_transform
    )

    return test_dataset
