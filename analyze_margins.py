#!/usr/bin/env python3
"""
Margin Analysis: Decision Boundary Margins for Prototypical Networks

Computes and visualizes margin distributions for all 8 model variants to show
how embedding quality affects decision boundaries.

Margin Definition:
- d_correct = distance to correct class prototype
- d_nearest_wrong = distance to nearest incorrect class prototype
- margin = d_nearest_wrong - d_correct
- Positive margin → correct prediction (query closer to correct prototype)
- Negative margin → incorrect prediction

Usage:
    # Analyze all 8 models
    python analyze_margins.py \
        --data_root CUB_200_2011/images \
        --classes_file CUB_200_2011/classes.txt

    # Analyze specific models
    python analyze_margins.py \
        --data_root CUB_200_2011/images \
        --classes_file CUB_200_2011/classes.txt \
        --models CNN/Fully_Tuned Transformers/Fully_Tuned

    # Quick test (no visualizations)
    python analyze_margins.py \
        --data_root CUB_200_2011/images \
        --classes_file CUB_200_2011/classes.txt \
        --models CNN/One_Layer \
        --n_episodes 10 \
        --no_viz
"""

import argparse
import json
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODELS_CONFIG = {
    'CNN': {
        'Base_Model': {
            'path': 'CNN/Base_Model',
            'checkpoint': 'CNN/Base_Model/outputs/checkpoints/best_model.pt',
            'embedding_dim': 512,
            'description': 'Frozen ResNet18, no projection'
        },
        'One_Layer': {
            'path': 'CNN/One_Layer',
            'checkpoint': 'CNN/One_Layer/outputs/checkpoints/best_model.pt',
            'embedding_dim': 256,
            'description': 'Frozen ResNet18 + 512→256 projection'
        },
        'Two_Layer': {
            'path': 'CNN/Two_Layer',
            'checkpoint': 'CNN/Two_Layer/outputs/checkpoints/best_model.pt',
            'embedding_dim': 256,
            'description': 'Frozen ResNet18 + 512→384→256 projection'
        },
        'Fully_Tuned': {
            'path': 'CNN/Fully_Tuned',
            'checkpoint': 'CNN/Fully_Tuned/outputs/checkpoints/best_model.pt',
            'embedding_dim': 512,
            'description': 'Fine-tuned ResNet18'
        }
    },
    'Transformers': {
        'Base_Model': {
            'path': 'Transformers/Base_Model',
            'checkpoint': 'Transformers/Base_Model/outputs/checkpoints/best_model.pt',
            'embedding_dim': 768,
            'description': 'Frozen ViT-B/16, no projection'
        },
        'One_Layer': {
            'path': 'Transformers/One_Layer',
            'checkpoint': 'Transformers/One_Layer/outputs/checkpoints/best_model.pt',
            'embedding_dim': 512,
            'description': 'Frozen ViT-B/16 + 768→512 projection'
        },
        'Two_Layer': {
            'path': 'Transformers/Two_Layer',
            'checkpoint': 'Transformers/Two_Layer/outputs/checkpoints/best_model.pt',
            'embedding_dim': 512,
            'description': 'Frozen ViT-B/16 + 768→640→512 projection'
        },
        'Fully_Tuned': {
            'path': 'Transformers/Fully_Tuned',
            'checkpoint': 'Transformers/Fully_Tuned/outputs/checkpoints/best_model.pt',
            'embedding_dim': 768,
            'description': 'Fine-tuned ViT-B/16'
        }
    }
}


# ============================================================================
# DATASET AND MODEL LOADING
# ============================================================================

class SimpleDataset:
    """Simple dataset for loading images by class."""

    def __init__(self, class_to_images: Dict[str, List[str]], transform=None):
        self.class_to_images = class_to_images
        self.transform = transform
        self.classes = sorted(class_to_images.keys())

    def get_class_images(self, class_name: str) -> List[str]:
        return self.class_to_images.get(class_name, [])


def load_model_variant(architecture: str, variant: str, device: str) -> nn.Module:
    """
    Dynamically load a model variant.

    Args:
        architecture: 'CNN' or 'Transformers'
        variant: 'Base_Model', 'One_Layer', 'Two_Layer', 'Fully_Tuned'
        device: Device to load on

    Returns:
        Embedding network (extracted from PrototypicalNetwork)
    """
    config = MODELS_CONFIG[architecture][variant]
    model_path = Path(config['path'])
    checkpoint_path = Path(config['checkpoint'])

    # Add model directory to path
    sys.path.insert(0, str(model_path))

    try:
        # Import the variant's MODEL module
        spec = importlib.util.spec_from_file_location("MODEL", model_path / "MODEL.py")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # Check if checkpoint exists
        if checkpoint_path.exists():
            # Load using the variant's load_model function
            model = model_module.load_model(
                str(checkpoint_path),
                device=device,
                embedding_dim=config['embedding_dim']
            )

            # Extract embedding network from PrototypicalNetwork
            if hasattr(model, 'embedding_network'):
                embedding_net = model.embedding_network
            else:
                embedding_net = model
        else:
            # Fallback: Create frozen pretrained model (for Base_Model variants)
            print(f"  Checkpoint not found, using frozen pretrained model")
            # Base_Model variants don't accept embedding_dim parameter
            if variant == 'Base_Model':
                embedding_net = model_module.EmbeddingNetwork(pretrained=True)
            else:
                embedding_net = model_module.EmbeddingNetwork(
                    pretrained=True,
                    embedding_dim=config['embedding_dim']
                )
            embedding_net.to(device)

        embedding_net.eval()

    finally:
        # Remove from path
        sys.path.remove(str(model_path))

    return embedding_net


def load_test_data(data_root: str, test_classes: List[str], samples_per_class: int = 100):
    """
    Load test dataset.

    Args:
        data_root: Path to images directory
        test_classes: List of test class names
        samples_per_class: Maximum samples per class

    Returns:
        Dictionary mapping class names to image paths
    """
    class_to_images = {}

    for class_name in test_classes:
        class_dir = Path(data_root) / class_name
        if not class_dir.exists():
            print(f"Warning: Class directory not found: {class_dir}")
            continue

        images = [str(p) for p in class_dir.glob('*') if p.is_file()]
        class_to_images[class_name] = images[:samples_per_class]

    return class_to_images


# ============================================================================
# EPISODE SAMPLING
# ============================================================================

class EpisodeSampler:
    """Sample few-shot episodes from a dataset."""

    def __init__(self,
                 class_to_images: Dict[str, List[str]],
                 transform,
                 n_way: int,
                 k_shot: int,
                 q_query: int,
                 n_episodes: int,
                 seed: int = 42):
        self.class_to_images = class_to_images
        self.transform = transform
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_episodes = n_episodes
        self.seed = seed

        self.classes = sorted(class_to_images.keys())

        # Validate
        min_samples = k_shot + q_query
        for cls in self.classes:
            if len(class_to_images[cls]) < min_samples:
                raise ValueError(
                    f"Class {cls} has {len(class_to_images[cls])} images, "
                    f"but need {min_samples} ({k_shot} support + {q_query} query)"
                )

    def __iter__(self):
        import random
        rng = random.Random(self.seed)

        for ep in range(self.n_episodes):
            # Sample n_way classes
            episode_classes = rng.sample(self.classes, self.n_way)

            support_images = []
            query_images = []
            query_labels = []

            for class_idx, class_name in enumerate(episode_classes):
                # Sample images for this class
                class_images = self.class_to_images[class_name]
                sampled = rng.sample(class_images, self.k_shot + self.q_query)

                support_imgs = sampled[:self.k_shot]
                query_imgs = sampled[self.k_shot:]

                # Load and transform
                for img_path in support_imgs:
                    img = Image.open(img_path).convert('RGB')
                    support_images.append(self.transform(img))

                for img_path in query_imgs:
                    img = Image.open(img_path).convert('RGB')
                    query_images.append(self.transform(img))
                    query_labels.append(class_idx)

            # Stack into tensors
            support_tensor = torch.stack(support_images)
            query_tensor = torch.stack(query_images)
            labels_tensor = torch.tensor(query_labels, dtype=torch.long)

            yield support_tensor, query_tensor, labels_tensor

    def __len__(self):
        return self.n_episodes


# ============================================================================
# MARGIN COMPUTATION
# ============================================================================

def compute_episode_margins(
    model: nn.Module,
    support_images: torch.Tensor,
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    n_way: int,
    k_shot: int,
    device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute margins for one episode.

    Args:
        model: Embedding network
        support_images: (n_way * k_shot, 3, H, W)
        query_images: (n_way * q_query, 3, H, W)
        query_labels: (n_way * q_query,)
        n_way: Number of classes
        k_shot: Support examples per class
        device: Device to use

    Returns:
        margins: (n_way * q_query,)
        d_correct: (n_way * q_query,)
        d_nearest_wrong: (n_way * q_query,)
    """
    model.eval()

    with torch.no_grad():
        # Move to device
        support_images = support_images.to(device, non_blocking=True)
        query_images = query_images.to(device, non_blocking=True)
        query_labels = query_labels.to(device, non_blocking=True)

        # Extract embeddings
        support_embeddings = model(support_images)  # (n_way * k_shot, emb_dim)
        query_embeddings = model(query_images)      # (n_queries, emb_dim)

        # Compute prototypes
        embedding_dim = support_embeddings.size(-1)
        support_embeddings = support_embeddings.view(n_way, k_shot, embedding_dim)
        prototypes = support_embeddings.mean(dim=1)  # (n_way, emb_dim)

        # Compute distances from each query to each prototype
        distances = torch.cdist(query_embeddings, prototypes, p=2)  # (n_queries, n_way)

        # For each query, get distance to correct class
        d_correct = distances[torch.arange(len(query_labels)), query_labels]

        # For each query, get distance to nearest wrong class
        wrong_distances = distances.clone()
        wrong_distances[torch.arange(len(query_labels)), query_labels] = float('inf')
        d_nearest_wrong = wrong_distances.min(dim=1).values

        # Compute margins
        margins = (d_nearest_wrong - d_correct).cpu().numpy()
        d_correct = d_correct.cpu().numpy()
        d_nearest_wrong = d_nearest_wrong.cpu().numpy()

    return margins, d_correct, d_nearest_wrong


def analyze_model_margins(
    architecture: str,
    variant: str,
    class_to_images: Dict[str, List[str]],
    transform,
    n_episodes: int,
    n_way: int,
    k_shot: int,
    q_query: int,
    device: str,
    seed: int
) -> pd.DataFrame:
    """
    Analyze margins for one model across multiple test episodes.

    Returns:
        DataFrame with columns: ['model', 'architecture', 'variant', 'episode',
                                 'query_idx', 'true_class', 'margin', 'd_correct',
                                 'd_nearest_wrong', 'correct']
    """
    print(f"\n{'='*70}")
    print(f"Analyzing: {architecture}/{variant}")
    print(f"{'='*70}")

    # Load model
    model = load_model_variant(architecture, variant, device)

    # Create episode sampler
    episode_sampler = EpisodeSampler(
        class_to_images=class_to_images,
        transform=transform,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        n_episodes=n_episodes,
        seed=seed
    )

    results = []

    for episode_idx, (support_imgs, query_imgs, query_lbls) in enumerate(
        tqdm(episode_sampler, desc=f"{architecture}/{variant}", total=n_episodes)
    ):
        # Compute margins for this episode
        margins, d_correct, d_nearest_wrong = compute_episode_margins(
            model, support_imgs, query_imgs, query_lbls, n_way, k_shot, device
        )

        # Store results
        for query_idx, (margin, d_c, d_nw, label) in enumerate(
            zip(margins, d_correct, d_nearest_wrong, query_lbls.cpu().numpy())
        ):
            results.append({
                'model': f"{architecture}/{variant}",
                'architecture': architecture,
                'variant': variant,
                'episode': episode_idx,
                'query_idx': query_idx,
                'true_class': int(label),
                'margin': float(margin),
                'd_correct': float(d_c),
                'd_nearest_wrong': float(d_nw),
                'correct': margin > 0
            })

    # Clear GPU memory
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return pd.DataFrame(results)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_margin_histogram(
    df: pd.DataFrame,
    model_name: str,
    output_path: Path,
    bins: int = 50
):
    """Plot margin distribution for a single model."""
    margins = df['margin'].values

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    ax.hist(margins, bins=bins, density=True, alpha=0.7,
            color='steelblue', edgecolor='black')

    # Add vertical line at margin = 0 (decision boundary)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
               label='Decision Boundary (margin=0)')

    # Add statistics
    mean_margin = margins.mean()
    median_margin = np.median(margins)

    ax.axvline(x=mean_margin, color='green', linestyle=':', linewidth=1.5,
               label=f'Mean: {mean_margin:.3f}')

    ax.set_xlabel('Margin (d_nearest_wrong - d_correct)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Margin Distribution - {model_name}\n',
                 fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved histogram to {output_path.name}")


def plot_comparison_frozen_vs_tuned(
    df: pd.DataFrame,
    architecture: str,
    output_path: Path
):
    """Compare Base_Model vs Fully_Tuned for one architecture."""
    base_model = f"{architecture}/Base_Model"
    tuned_model = f"{architecture}/Fully_Tuned"

    df_base = df[df['model'] == base_model]
    df_tuned = df[df['model'] == tuned_model]

    if df_base.empty or df_tuned.empty:
        print(f"  Skipping comparison for {architecture} (missing data)")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot both distributions
    ax.hist(df_base['margin'].values, bins=50, density=True, alpha=0.5,
            color='blue', label='Frozen',
            edgecolor='black')
    ax.hist(df_tuned['margin'].values, bins=50, density=True, alpha=0.5,
            color='orange', label='Fine-tuned',
            edgecolor='black')

    # Decision boundary
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
               label='Decision Boundary')

    ax.set_xlabel('Margin', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(
        f'{architecture}: Frozen vs Fine-tuned Margin Distributions',
        fontsize=14, fontweight='bold'
    )
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved comparison to {output_path.name}")


def plot_base_vs_variants(
    df: pd.DataFrame,
    architecture: str,
    output_path: Path
):
    """Compare Base_Model vs One_Layer, Two_Layer, and Fully_Tuned for one architecture."""
    base_model = f"{architecture}/Base_Model"
    one_layer = f"{architecture}/One_Layer"
    two_layer = f"{architecture}/Two_Layer"
    tuned_model = f"{architecture}/Fully_Tuned"

    df_base = df[df['model'] == base_model]
    df_one = df[df['model'] == one_layer]
    df_two = df[df['model'] == two_layer]
    df_tuned = df[df['model'] == tuned_model]

    if df_base.empty:
        print(f"  Skipping comparison for {architecture} (missing base model)")
        return

    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{architecture}: Base Model vs Variants', fontsize=16, fontweight='bold')

    # Plot 1: Base vs One_Layer
    if not df_one.empty:
        ax = axes[0, 0]
        ax.hist(df_base['margin'].values, bins=50, density=True, alpha=0.5,
                color='blue', label='Base',
                edgecolor='black')
        ax.hist(df_one['margin'].values, bins=50, density=True, alpha=0.5,
                color='green', label='One-Layer',
                edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
        ax.set_xlabel('Margin', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Base vs One-Layer Projection', fontsize=12, fontweight='bold')
        ax.legend()

    # Plot 2: Base vs Two_Layer
    if not df_two.empty:
        ax = axes[0, 1]
        ax.hist(df_base['margin'].values, bins=50, density=True, alpha=0.5,
                color='blue', label='Base',
                edgecolor='black')
        ax.hist(df_two['margin'].values, bins=50, density=True, alpha=0.5,
                color='purple', label='Two-Layer',
                edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
        ax.set_xlabel('Margin', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Base vs Two-Layer Projection', fontsize=12, fontweight='bold')
        ax.legend()

    # Plot 3: Base vs Fully_Tuned
    if not df_tuned.empty:
        ax = axes[1, 0]
        ax.hist(df_base['margin'].values, bins=50, density=True, alpha=0.5,
                color='blue', label='Base',
                edgecolor='black')
        ax.hist(df_tuned['margin'].values, bins=50, density=True, alpha=0.5,
                color='orange', label='Fully Tuned',
                edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
        ax.set_xlabel('Margin', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Base vs Fully Tuned', fontsize=12, fontweight='bold')
        ax.legend()

    # Plot 4: All variants together
    ax = axes[1, 1]
    ax.hist(df_base['margin'].values, bins=50, density=True, alpha=0.4,
            color='blue', label='Base',
            edgecolor='black')
    if not df_one.empty:
        ax.hist(df_one['margin'].values, bins=50, density=True, alpha=0.4,
                color='green', label='One-Layer',
                edgecolor='black')
    if not df_two.empty:
        ax.hist(df_two['margin'].values, bins=50, density=True, alpha=0.4,
                color='purple', label='Two-Layer',
                edgecolor='black')
    if not df_tuned.empty:
        ax.hist(df_tuned['margin'].values, bins=50, density=True, alpha=0.4,
                color='orange', label='Fully Tuned',
                edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
    ax.set_xlabel('Margin', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('All Variants Comparison', fontsize=12, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved base vs variants comparison to {output_path.name}")


def plot_all_models_comparison(df: pd.DataFrame, output_path: Path):
    """Violin plot comparing margin distributions across all models."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Prepare data for violin plot
    models = sorted(df['model'].unique())
    data = [df[df['model'] == model]['margin'].values for model in models]

    # Create violin plot
    parts = ax.violinplot(data, positions=range(len(models)),
                          showmeans=True, showmedians=True)

    # Customize colors
    for pc in parts['bodies']:
        pc.set_facecolor('steelblue')
        pc.set_alpha(0.7)

    # Decision boundary line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2,
               label='Decision Boundary')

    # Labels and formatting
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Margin', fontsize=12)
    ax.set_title('Margin Distributions Across All Models',
                 fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved all-models comparison to {output_path.name}")


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_all_margins(
    data_root: str,
    classes_file: str,
    output_dir: str,
    device: str,
    selected_models: Optional[List[str]] = None,
    n_episodes: int = 500,
    n_way: int = 5,
    k_shot: int = 5,
    q_query: int = 15,
    seed: int = 42,
    create_visualizations: bool = True
):
    """
    Analyze margins for selected models.

    Args:
        data_root: Path to images directory
        classes_file: Path to classes.txt
        output_dir: Output directory
        device: Device to use
        selected_models: List of model specs (e.g., ['CNN/Fully_Tuned']) or None for all
        n_episodes: Number of test episodes per model
        n_way: Number of classes per episode
        k_shot: Support examples per class
        q_query: Query examples per class
        seed: Random seed
        create_visualizations: Whether to create plots
    """
    output_path = Path(output_dir) / 'margins'
    output_path.mkdir(parents=True, exist_ok=True)

    if create_visualizations:
        viz_path = output_path / 'visualizations'
        viz_path.mkdir(exist_ok=True)

    # Parse model selection
    if selected_models is None:
        # Default: all 8 models
        models_to_run = [
            (arch, var)
            for arch in ['CNN', 'Transformers']
            for var in ['Base_Model', 'One_Layer', 'Two_Layer', 'Fully_Tuned']
        ]
    else:
        # Parse user-specified models (e.g., 'CNN/Fully_Tuned')
        models_to_run = []
        for spec in selected_models:
            parts = spec.split('/')
            if len(parts) != 2:
                print(f"Warning: Invalid model spec '{spec}' (expected 'Architecture/Variant')")
                continue
            arch, var = parts
            if arch not in MODELS_CONFIG or var not in MODELS_CONFIG[arch]:
                print(f"Warning: Unknown model '{spec}'")
                continue
            models_to_run.append((arch, var))

    print(f"\n{'='*70}")
    print(f"MARGIN ANALYSIS")
    print(f"{'='*70}")
    print(f"Analyzing {len(models_to_run)} model(s)")
    print(f"Protocol: {n_way}-way {k_shot}-shot, {q_query} queries/class")
    print(f"Episodes: {n_episodes} per model")
    print(f"Expected margins per model: {n_episodes * n_way * q_query}")
    print(f"Seed: {seed}")

    # Load test classes
    print("\nLoading test data...")
    sys.path.insert(0, 'Transformers/Fully_Tuned')
    from PREP import load_and_split_classes, get_eval_transforms

    _, _, test_classes = load_and_split_classes(
        classes_file=classes_file,
        seed=seed
    )

    print(f"Test classes: {len(test_classes)}")

    # Load test images
    class_to_images = load_test_data(data_root, test_classes)
    print(f"Loaded images from {len(class_to_images)} classes")

    # Get transform
    transform = get_eval_transforms()

    sys.path.remove('Transformers/Fully_Tuned')

    # Validate checkpoints before running
    print("\nValidating checkpoints...")
    available_models = []
    for arch, var in models_to_run:
        checkpoint_path = Path(MODELS_CONFIG[arch][var]['checkpoint'])
        if checkpoint_path.exists() or var == 'Base_Model':
            available_models.append((arch, var))
        else:
            print(f"  Warning: {arch}/{var} checkpoint not found at {checkpoint_path}")

    if not available_models:
        print("ERROR: No valid model checkpoints found!")
        return None

    print(f"Found {len(available_models)} models with valid checkpoints")
    models_to_run = available_models

    # Analyze each model
    all_results = []
    failed_models = []

    for idx, (architecture, variant) in enumerate(models_to_run, 1):
        print(f"\n[{idx}/{len(models_to_run)}] Processing {architecture}/{variant}")
        try:
            df_model = analyze_model_margins(
                architecture, variant, class_to_images, transform,
                n_episodes, n_way, k_shot, q_query, device, seed
            )
            all_results.append(df_model)

            # Create per-model visualization
            if create_visualizations:
                viz_file = viz_path / f"margins_{architecture}_{variant}.png"
                plot_margin_histogram(df_model, f"{architecture}/{variant}", viz_file)

        except Exception as e:
            print(f"ERROR processing {architecture}/{variant}: {e}")
            import traceback
            traceback.print_exc()
            failed_models.append(f"{architecture}/{variant}")

    # Combine all results
    if not all_results:
        print("\nERROR: No models completed successfully!")
        return None

    df_all = pd.concat(all_results, ignore_index=True)

    # Save to CSV
    csv_path = output_path / 'margins_all.csv'
    df_all.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"Saved all margins to: {csv_path}")

    # Save summary statistics
    summary = {}
    for model in sorted(df_all['model'].unique()):
        df_model = df_all[df_all['model'] == model]
        summary[model] = {
            'accuracy': float((df_model['margin'] > 0).mean()),
            'mean_margin': float(df_model['margin'].mean()),
            'median_margin': float(df_model['margin'].median()),
            'std_margin': float(df_model['margin'].std()),
            'num_samples': len(df_model)
        }

    summary_path = output_path / 'margins_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {summary_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Accuracy':>10} {'Mean Margin':>12} {'Std Margin':>12}")
    print("-" * 70)
    for model in sorted(summary.keys()):
        stats = summary[model]
        print(f"{model:<30} {stats['accuracy']:>9.2%} {stats['mean_margin']:>12.4f} {stats['std_margin']:>12.4f}")

    # Create comparison visualizations
    if create_visualizations:
        print(f"\n{'='*70}")
        print("Creating comparison visualizations...")

        # CNN: Base vs all variants
        plot_base_vs_variants(
            df_all, 'CNN', viz_path / 'comparison_CNN_base_vs_variants.png'
        )

        # Transformers: Base vs all variants
        plot_base_vs_variants(
            df_all, 'Transformers', viz_path / 'comparison_Transformers_base_vs_variants.png'
        )

        # CNN: Frozen vs Tuned (legacy plot)
        plot_comparison_frozen_vs_tuned(
            df_all, 'CNN', viz_path / 'comparison_CNN_frozen_vs_tuned.png'
        )

        # Transformers: Frozen vs Tuned (legacy plot)
        plot_comparison_frozen_vs_tuned(
            df_all, 'Transformers', viz_path / 'comparison_Transformers_frozen_vs_tuned.png'
        )

        # All models
        plot_all_models_comparison(
            df_all, viz_path / 'comparison_all_models.png'
        )

    # Summary of successful and failed models
    successful_models = len(all_results)
    total_models = len(models_to_run)

    print(f"\n{'='*70}")
    print("Margin analysis complete!")
    print(f"{'='*70}")
    print(f"Successful: {successful_models}/{total_models}")
    if failed_models:
        print(f"Failed models: {', '.join(failed_models)}")
    print(f"\nResults saved to: {output_path}")

    return df_all


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Margin Analysis: Compute decision boundary margins for ProtoNets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all 8 models
  python analyze_margins.py --data_root CUB_200_2011/images --classes_file CUB_200_2011/classes.txt

  # Analyze specific models
  python analyze_margins.py --data_root CUB_200_2011/images --classes_file CUB_200_2011/classes.txt \\
      --models CNN/Fully_Tuned Transformers/Fully_Tuned

  # Quick test
  python analyze_margins.py --data_root CUB_200_2011/images --classes_file CUB_200_2011/classes.txt \\
      --models CNN/One_Layer --n_episodes 10 --no_viz
        """
    )

    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to CUB images directory (e.g., CUB_200_2011/images)'
    )

    parser.add_argument(
        '--classes_file',
        type=str,
        required=True,
        help='Path to classes.txt file'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./analysis_results',
        help='Output directory for results (default: ./analysis_results)'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Models to analyze (e.g., CNN/Fully_Tuned Transformers/One_Layer). Default: all 8 models'
    )

    parser.add_argument(
        '--n_episodes',
        type=int,
        default=500,
        help='Number of test episodes per model (default: 100)'
    )

    parser.add_argument(
        '--n_way',
        type=int,
        default=5,
        help='Number of classes per episode (default: 5)'
    )

    parser.add_argument(
        '--k_shot',
        type=int,
        default=5,
        help='Support examples per class (default: 5)'
    )

    parser.add_argument(
        '--q_query',
        type=int,
        default=15,
        help='Query examples per class (default: 15)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )

    parser.add_argument(
        '--no_viz',
        action='store_true',
        help='Skip creating visualizations (only save CSV)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'

    analyze_all_margins(
        data_root=args.data_root,
        classes_file=args.classes_file,
        output_dir=args.output_dir,
        device=args.device,
        selected_models=args.models,
        n_episodes=args.n_episodes,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        seed=args.seed,
        create_visualizations=not args.no_viz
    )


if __name__ == '__main__':
    main()
