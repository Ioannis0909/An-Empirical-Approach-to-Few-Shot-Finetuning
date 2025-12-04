#!/usr/bin/env python3
"""
Test on Training Classes - Overfitting Diagnostic Tool

This script evaluates trained models on TRAINING classes (instead of test classes)
to diagnose whether poor test performance is due to:
1. Overfitting: High accuracy on train classes, low on test classes
2. Degradation: Low accuracy on both train and test classes

The models are loaded in eval mode (no retraining) and tested using the same
few-shot episode sampling protocol as normal testing.

Usage:
    # Test all models on training classes
    python test_on_train_classes.py \
        --data_root Bird_Dataset/images \
        --classes_file Bird_Dataset/classes.txt

    # Test specific models
    python test_on_train_classes.py \
        --data_root Bird_Dataset/images \
        --classes_file Bird_Dataset/classes.txt \
        --models CNN/One_Layer

    # Compare with test results
    python test_on_train_classes.py \
        --data_root Bird_Dataset/images \
        --classes_file Bird_Dataset/classes.txt \
        --compare_with_test
"""

import argparse
import json
import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm


# ============================================================================
# MODEL CONFIGURATION (same as analyze_margins.py)
# ============================================================================

MODELS_CONFIG = {
    'CNN': {
        'One_Layer': {
            'path': 'CNN/One_Layer',
            'checkpoint': 'CNN/One_Layer/outputs/checkpoints/best_model.pt',
            'embedding_dim': 256,
        },
        'Two_Layer': {
            'path': 'CNN/Two_Layer',
            'checkpoint': 'CNN/Two_Layer/outputs/checkpoints/best_model.pt',
            'embedding_dim': 256,
        },
        'Fully_Tuned': {
            'path': 'CNN/Fully_Tuned',
            'checkpoint': 'CNN/Fully_Tuned/outputs/checkpoints/best_model.pt',
            'embedding_dim': 512,
        }
    },
    'Transformers': {
        'One_Layer': {
            'path': 'Transformers/One_Layer',
            'checkpoint': 'Transformers/One_Layer/outputs/checkpoints/best_model.pt',
            'embedding_dim': 512,
        },
        'Two_Layer': {
            'path': 'Transformers/Two_Layer',
            'checkpoint': 'Transformers/Two_Layer/outputs/checkpoints/best_model.pt',
            'embedding_dim': 512,
        },
        'Fully_Tuned': {
            'path': 'Transformers/Fully_Tuned',
            'checkpoint': 'Transformers/Fully_Tuned/outputs/checkpoints/best_model.pt',
            'embedding_dim': 768,
        }
    }
}


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_variant(architecture: str, variant: str, device: str) -> nn.Module:
    """Load a model variant in eval mode."""
    config = MODELS_CONFIG[architecture][variant]
    model_path = Path(config['path'])
    checkpoint_path = Path(config['checkpoint'])

    sys.path.insert(0, str(model_path))

    try:
        spec = importlib.util.spec_from_file_location("MODEL", model_path / "MODEL.py")
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        if checkpoint_path.exists():
            model = model_module.load_model(
                str(checkpoint_path),
                device=device,
                embedding_dim=config['embedding_dim']
            )

            if hasattr(model, 'embedding_network'):
                embedding_net = model.embedding_network
            else:
                embedding_net = model
        else:
            print(f"  Warning: Checkpoint not found, using pretrained base")
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
        sys.path.remove(str(model_path))

    return embedding_net


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model_on_classes(
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
) -> Dict[str, float]:
    """
    Evaluate a single model on given classes.

    Returns:
        Dictionary with accuracy statistics
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {architecture}/{variant}")
    print(f"{'='*70}")

    # Load model
    model = load_model_variant(architecture, variant, device)

    # Import episode sampler from the model's PREP module
    model_path = Path(MODELS_CONFIG[architecture][variant]['path'])
    sys.path.insert(0, str(model_path))

    try:
        from PREP import EpisodeSampler, CaptchaDataset

        # Create dataset
        classes = sorted(class_to_images.keys())
        dataset = CaptchaDataset(
            class_to_images=class_to_images,
            classes=classes,
            transform=transform
        )

        # Create episode sampler
        sampler = EpisodeSampler(
            dataset=dataset,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
            episodes_per_epoch=n_episodes,
            seed=seed
        )

        # Evaluate
        accuracies = []

        model.eval()
        with torch.no_grad():
            for support_imgs, query_imgs, query_lbls in tqdm(
                sampler,
                desc=f"{architecture}/{variant}",
                total=n_episodes
            ):
                support_imgs = support_imgs.to(device, non_blocking=True)
                query_imgs = query_imgs.to(device, non_blocking=True)
                query_lbls = query_lbls.to(device, non_blocking=True)

                # Extract embeddings
                support_embs = model(support_imgs)
                query_embs = model(query_imgs)

                # Compute prototypes
                embedding_dim = support_embs.size(-1)
                support_embs = support_embs.view(n_way, k_shot, embedding_dim)
                prototypes = support_embs.mean(dim=1)

                # Compute distances and predictions
                distances = torch.cdist(query_embs, prototypes, p=2)
                predictions = distances.argmin(dim=1)

                # Calculate accuracy
                acc = (predictions == query_lbls).float().mean().item()
                accuracies.append(acc)

        # Calculate statistics
        mean_acc = float(np.mean(accuracies))
        std_acc = float(np.std(accuracies))
        ci_95 = float(1.96 * std_acc / np.sqrt(n_episodes))

        print(f"Accuracy: {mean_acc:.4f} +/- {ci_95:.4f} (95% CI)")

        results = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'ci_95': ci_95,
            'n_episodes': n_episodes
        }

    finally:
        sys.path.remove(str(model_path))

    # Clear GPU memory
    del model
    if device == 'cuda':
        torch.cuda.empty_cache()

    return results


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_train_class_performance(
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
    compare_with_test: bool = False
):
    """
    Test models on training classes to diagnose overfitting.
    """
    output_path = Path(output_dir) / 'train_class_evaluation'
    output_path.mkdir(parents=True, exist_ok=True)

    # Parse model selection
    if selected_models is None:
        models_to_run = [
            (arch, var)
            for arch in ['CNN', 'Transformers']
            for var in ['Base_Model', 'One_Layer', 'Two_Layer', 'Fully_Tuned']
        ]
    else:
        models_to_run = []
        for spec in selected_models:
            parts = spec.split('/')
            if len(parts) != 2:
                print(f"Warning: Invalid model spec '{spec}'")
                continue
            arch, var = parts
            if arch not in MODELS_CONFIG or var not in MODELS_CONFIG[arch]:
                print(f"Warning: Unknown model '{spec}'")
                continue
            models_to_run.append((arch, var))

    print(f"\n{'='*70}")
    print(f"TRAINING CLASS EVALUATION - Overfitting Diagnostic")
    print(f"{'='*70}")
    print(f"Evaluating {len(models_to_run)} model(s)")
    print(f"Protocol: {n_way}-way {k_shot}-shot, {q_query} queries/class")
    print(f"Episodes: {n_episodes} per model")
    print(f"Seed: {seed}")

    # Load TRAINING classes (not test classes!)
    print("\nLoading TRAINING classes...")
    sys.path.insert(0, 'CNN/Fully_Tuned')
    from PREP import load_and_split_classes, get_eval_transforms, load_all_images_for_classes

    train_classes, val_classes, test_classes = load_and_split_classes(
        classes_file=classes_file,
        seed=seed
    )

    print(f"\nUsing TRAINING classes: {len(train_classes)} classes")
    print(f"(Normal testing uses TEST classes: {len(test_classes)} classes)")

    # Load training class images
    train_class_to_images = load_all_images_for_classes(data_root, train_classes)
    print(f"Loaded images from {len(train_class_to_images)} training classes")

    # Get transform (eval mode, no augmentation)
    transform = get_eval_transforms()

    sys.path.remove('CNN/Fully_Tuned')

    # Validate checkpoints
    print("\nValidating checkpoints...")
    available_models = []
    for arch, var in models_to_run:
        checkpoint_path = Path(MODELS_CONFIG[arch][var]['checkpoint'])
        if checkpoint_path.exists() or var == 'Base_Model':
            available_models.append((arch, var))
        else:
            print(f"  Warning: {arch}/{var} checkpoint not found")

    if not available_models:
        print("ERROR: No valid model checkpoints found!")
        return None

    print(f"Found {len(available_models)} models with valid checkpoints")
    models_to_run = available_models

    # Evaluate each model on TRAINING classes
    results = {}

    for idx, (architecture, variant) in enumerate(models_to_run, 1):
        print(f"\n[{idx}/{len(models_to_run)}] Processing {architecture}/{variant}")
        try:
            model_results = evaluate_model_on_classes(
                architecture, variant,
                train_class_to_images, transform,
                n_episodes, n_way, k_shot, q_query,
                device, seed
            )
            results[f"{architecture}/{variant}"] = model_results

        except Exception as e:
            print(f"ERROR processing {architecture}/{variant}: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    results_path = output_path / 'train_class_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Saved results to: {results_path}")

    # Print summary
    print(f"\n{'='*70}")
    print("RESULTS ON TRAINING CLASSES")
    print(f"{'='*70}")
    print(f"{'Model':<30} {'Accuracy':>12} {'95% CI':>12}")
    print("-" * 70)

    for model_name in sorted(results.keys()):
        stats = results[model_name]
        acc = stats['mean_accuracy']
        ci = stats['ci_95']
        print(f"{model_name:<30} {acc:>11.2%} ± {ci:>10.4f}")

    # Compare with test results if requested
    if compare_with_test:
        print(f"\n{'='*70}")
        print("COMPARISON: TRAINING vs TEST CLASSES")
        print(f"{'='*70}")

        # Try to load test results
        test_results_path = Path('./analysis_results/few_shot_results')
        comparison_data = []

        for model_name in sorted(results.keys()):
            train_acc = results[model_name]['mean_accuracy']

            # Try to find corresponding test results
            arch, var = model_name.split('/')
            model_test_path = Path(f"{arch}/{var}/outputs/few_shot_results/few_shot_results.json")

            test_acc = None
            if model_test_path.exists():
                try:
                    with open(model_test_path, 'r') as f:
                        test_data = json.load(f)
                        # Get 5-shot results
                        if '5' in test_data.get('k_shot_results', {}):
                            test_acc = test_data['k_shot_results']['5']['mean_accuracy']
                except Exception as e:
                    print(f"  Warning: Could not load test results for {model_name}: {e}")

            comparison_data.append({
                'model': model_name,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'gap': (train_acc - test_acc) if test_acc is not None else None
            })

        # Print comparison table
        print(f"\n{'Model':<30} {'Train Acc':>12} {'Test Acc':>12} {'Gap':>12} {'Diagnosis':<20}")
        print("-" * 100)

        for item in comparison_data:
            model = item['model']
            train = item['train_acc']
            test = item['test_acc']
            gap = item['gap']

            if test is None:
                print(f"{model:<30} {train:>11.2%} {'N/A':>12} {'N/A':>12} {'No test data':<20}")
            else:
                diagnosis = "Overfitting" if gap > 0.10 else ("Good generalization" if gap < 0.05 else "Mild overfitting")
                print(f"{model:<30} {train:>11.2%} {test:>11.2%} {gap:>11.2%} {diagnosis:<20}")

        # Save comparison
        comparison_path = output_path / 'train_vs_test_comparison.json'
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"\nComparison saved to: {comparison_path}")

    print(f"\n{'='*70}")
    print("Diagnostic Guide:")
    print("  - Large gap (>10%): Strong overfitting - model memorized train classes")
    print("  - Small gap (<5%): Good generalization")
    print("  - Both low: Poor model performance (not overfitting, just weak)")
    print(f"{'='*70}")

    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test models on training classes to diagnose overfitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to images directory'
    )

    parser.add_argument(
        '--classes_file',
        type=str,
        required=True,
        help='Path to classes.txt'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./analysis_results',
        help='Output directory (default: ./analysis_results)'
    )

    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=None,
        help='Models to test (e.g., CNN/Fully_Tuned). Default: all'
    )

    parser.add_argument(
        '--n_episodes',
        type=int,
        default=500,
        help='Number of episodes (default: 500)'
    )

    parser.add_argument(
        '--n_way',
        type=int,
        default=5,
        help='N-way (default: 5)'
    )

    parser.add_argument(
        '--k_shot',
        type=int,
        default=5,
        help='K-shot (default: 5)'
    )

    parser.add_argument(
        '--q_query',
        type=int,
        default=15,
        help='Query samples per class (default: 15)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device (default: cuda)'
    )

    parser.add_argument(
        '--compare_with_test',
        action='store_true',
        help='Compare with existing test results'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        args.device = 'cpu'

    analyze_train_class_performance(
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
        compare_with_test=args.compare_with_test
    )


if __name__ == '__main__':
    main()
