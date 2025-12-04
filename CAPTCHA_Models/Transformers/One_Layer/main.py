#!/usr/bin/env python3
"""
main.py - Complete Few-Shot Classification Pipeline for CUB-200-2011

Usage:
    python main.py --data_root CUB_200_2011/CUB_200_2011/images --classes_file CUB_200_2011/CUB_200_2011/classes.txt --mode train
    python main.py --data_root CUB_200_2011/CUB_200_2011/images --classes_file CUB_200_2011/CUB_200_2011/classes.txt --mode eval --checkpoint path/to/best_model.pt
    python main.py --data_root CUB_200_2011/images --classes_file CUB_200_2011/classes.txt --mode both
"""

import argparse
import torch
from torch.utils.data import DataLoader

from PREP import prepare_training_data, prepare_test_data, EpisodeSampler
from MODEL import (
    EmbeddingNetwork,
    PrototypicalNetwork,
    TrainingConfig,
    train_model,
    load_model
)
from FEW_SHOT import FewShotConfig, run_few_shot_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description='Few-Shot Classification on CUB-200-2011')

    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to CUB images directory (e.g., CUB_200_2011/CUB_200_2011/images)'
    )
    
    parser.add_argument(
        '--classes_file',
        type=str,
        required=True,
        help='Path to classes.txt file (e.g., CUB_200_2011/CUB_200_2011/classes.txt)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=99,
        help='Random seed for base/novel split (default: 99)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='both',
        choices=['train', 'eval', 'both'],
        help='Mode: train, eval, or both (default: both)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for evaluation mode (required for --mode eval)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Enforce GPU-only setup (with fallback if unavailable)
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. This code is optimized for GPU-only runs.")
        device = 'cpu'
    else:
        device = 'cuda'

    print("=" * 60)
    print("FEW-SHOT CLASSIFICATION - CUB-200-2011")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Classes file: {args.classes_file}")
    print(f"Random seed: {args.seed}")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print("=" * 60)

    # ========================================================================
    # TRAINING PHASE
    # ========================================================================
    if args.mode in ['train', 'both']:
        print("\n" + "=" * 60)
        print("TRAINING PHASE")
        print("=" * 60)

        # Config
        config = TrainingConfig()
        config.device = device

        # Prepare training data with 3-way class split
        train_dataset, val_dataset, test_dataset, _ = prepare_training_data(
            data_root=args.data_root,
            classes_file=args.classes_file,
            train_ratio=5/11,  # 5 train classes
            val_ratio=3/11,    # 3 val classes (unseen during training)
            test_ratio=3/11,   # 3 test classes (for final evaluation)
            seed=args.seed,
            num_workers=config.num_workers
        )

        # Episode samplers (logical episode generators)
        train_episode_sampler = EpisodeSampler(
            dataset=train_dataset,
            n_way=config.n_way,
            k_shot=config.k_shot,
            q_query=config.q_query,
            episodes_per_epoch=config.episodes_per_epoch,
        )

        val_episode_sampler = EpisodeSampler(
            dataset=val_dataset,
            n_way=config.n_way,
            k_shot=config.k_shot,
            q_query=config.val_q_query,
            episodes_per_epoch=config.val_episodes_per_epoch,
        )

        # DataLoaders with pin_memory + prefetch_episodes
        train_loader = DataLoader(
            train_episode_sampler,
            batch_size=None,  # each yield is (support, query, labels)
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_episodes,
            persistent_workers=config.num_workers > 0,
        )

        val_loader = DataLoader(
            val_episode_sampler,
            batch_size=None,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            prefetch_factor=config.prefetch_episodes,
            persistent_workers=config.num_workers > 0,
        )
        
        # Model initialization
        embedding_net = EmbeddingNetwork(
            pretrained=True,
            #freeze_backbone=True
        )
        model = PrototypicalNetwork(embedding_network=embedding_net)

        # Train
        history = train_model(
            model=model,
            train_sampler=train_loader,
            val_sampler=val_loader,
            config=config
        )

        # Use best checkpoint for downstream eval (if mode == both)
        args.checkpoint = f"{config.save_dir}/best_model.pt"

    # ========================================================================
    # EVALUATION PHASE
    # ========================================================================
    if args.mode in ['eval', 'both']:
        print("\n" + "=" * 60)
        print("FEW-SHOT EVALUATION PHASE")
        print("=" * 60)

        if args.checkpoint is None:
            print("Error: --checkpoint is required for eval mode")
            return

        # Prepare test class data (50 unseen test classes)
        test_dataset = prepare_test_data(
            data_root=args.data_root,
            classes_file=args.classes_file,
            seed=args.seed
        )

        # Few-shot eval config
        eval_config = FewShotConfig()
        eval_config.device = device
        # Load trained model
        print(f"\nLoading model from {args.checkpoint}")
        model = load_model(args.checkpoint, device=device)

        # Run evaluation
        results = run_few_shot_evaluation(
            model=model,
            novel_dataset=test_dataset,
            config=eval_config
        )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()