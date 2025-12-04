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

def load_and_split_classes(classes_file: str, base_ratio: float = 0.7, seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Load classes from classes.txt and randomly split into base and novel classes.
    
    Args:
        classes_file: Path to classes.txt file
        base_ratio: Proportion of classes to use as base classes (default 0.7 for ~140/200)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (base_classes, novel_classes)
    """
    random.seed(seed)
    
    # Read classes from file
    all_classes = []
    with open(classes_file, 'r') as f:
        for line in f:
            # Parse format: "1 001.Black_footed_Albatross"
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                class_name = parts[1]
                all_classes.append(class_name)
    
    # Shuffle and split
    random.shuffle(all_classes)
    n_base = int(len(all_classes) * base_ratio)
    
    base_classes = sorted(all_classes[:n_base])
    novel_classes = sorted(all_classes[n_base:])
    
    print(f"\nTotal classes: {len(all_classes)}")
    print(f"Base classes: {len(base_classes)}")
    print(f"Novel classes: {len(novel_classes)}")
    
    return base_classes, novel_classes


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

def split_dataset_by_images(root_dir: str, classes: List[str], 
                            train_ratio: float = 0.70, seed: int = 42) -> Dict[str, Dict[str, List[str]]]:
    """
    Split dataset images into train/val for each class.
    
    Args:
        root_dir: Path to images directory
        classes: List of class names
        train_ratio: Proportion for training (rest goes to validation)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping split name to class-to-images dict
    """
    random.seed(seed)
    root = Path(root_dir)
    
    splits = {
        'train': {},
        'val': {}
    }
    
    print("\nSplitting base classes into train/val...")
    for class_name in classes:
        class_dir = root / class_name
        all_images = [str(img) for img in class_dir.glob('*') if img.is_file()]
        
        # Shuffle
        random.shuffle(all_images)
        
        # Calculate split index
        n_total = len(all_images)
        n_train = int(n_total * train_ratio)
        
        # Split
        train_images = all_images[:n_train]
        val_images = all_images[n_train:]
        
        splits['train'][class_name] = train_images
        splits['val'][class_name] = val_images
        
        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val")
    
    return splits


def load_novel_classes(root_dir: str, classes: List[str]) -> Dict[str, List[str]]:
    """
    Load novel class images (no splitting, use all images).
    
    Args:
        root_dir: Path to images directory
        classes: List of novel class names
    
    Returns:
        Dictionary mapping class names to image paths
    """
    root = Path(root_dir)
    class_to_images = {}
    
    print("\nLoading novel classes...")
    for class_name in classes:
        class_dir = root / class_name
        images = [str(img) for img in class_dir.glob('*') if img.is_file()]
        class_to_images[class_name] = images
        print(f"{class_name}: {len(images)} images")
    
    return class_to_images


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

def prepare_base_data(data_root: str, classes_file: str, train_ratio: float = 0.75,
                     seed: int = 42, num_workers: int = 4):
    """
    Prepare base class data for training.
    
    Args:
        data_root: Path to images directory (e.g., 'CUB_200_2011/CUB_200_2011/images')
        classes_file: Path to classes.txt file
        train_ratio: Proportion for training split (rest goes to validation)
        seed: Random seed
        num_workers: Number of parallel workers for data loading
    
    Returns:
        train_dataset, val_dataset, num_workers
    """
    # Load and split classes
    base_classes, _ = load_and_split_classes(classes_file, base_ratio=0.7, seed=seed)
    
    # Split data
    splits = split_dataset_by_images(
        root_dir=data_root,
        classes=base_classes,
        train_ratio=train_ratio,
        seed=seed
    )
    
    # Create transforms
    train_transform = get_train_transforms()
    eval_transform = get_eval_transforms()
    
    # Create datasets
    train_dataset = CaptchaDataset(
        class_to_images=splits['train'],
        classes=base_classes,
        transform=train_transform
    )
    
    val_dataset = CaptchaDataset(
        class_to_images=splits['val'],
        classes=base_classes,
        transform=eval_transform
    )
    
    return train_dataset, val_dataset, num_workers


def prepare_novel_data(data_root: str, classes_file: str, seed: int = 42):
    """
    Prepare novel class data for evaluation.
    
    Args:
        data_root: Path to images directory (e.g., 'CUB_200_2011/CUB_200_2011/images')
        classes_file: Path to classes.txt file
        seed: Random seed (must match the seed used in prepare_base_data)
    
    Returns:
        novel_dataset
    """
    # Load and split classes
    _, novel_classes = load_and_split_classes(classes_file, base_ratio=0.7, seed=seed)
    
    # Load novel classes
    class_to_images = load_novel_classes(
        root_dir=data_root,
        classes=novel_classes  # Use dynamically loaded novel classes
    )
    
    # Create transform (no augmentation for evaluation)
    eval_transform = get_eval_transforms()
    
    # Create dataset
    novel_dataset = CaptchaDataset(
        class_to_images=class_to_images,
        classes=novel_classes,  # Use dynamically loaded novel classes
        transform=eval_transform
    )
    
    return novel_dataset
