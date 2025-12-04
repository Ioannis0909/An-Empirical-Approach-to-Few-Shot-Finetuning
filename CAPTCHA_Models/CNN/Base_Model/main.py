#!/usr/bin/env python3
"""
main.py - CUB-200-2011 Few-Shot Classification with Frozen ImageNet Baseline

python main.py \
    --data_root CUB_200_2011/images \
    --classes_file CUB_200_2011/classes.txt \
    --seed 42
"""

#!/usr/bin/env python3
"""
main.py - CUB-200-2011 Few-Shot Classification with Frozen ImageNet Baseline
"""

import argparse
import torch

from PREP import prepare_novel_data
from MODEL import create_frozen_baseline
from FEW_SHOT import FewShotConfig, run_few_shot_evaluation


def parse_args():
    parser = argparse.ArgumentParser(description='CUB Few-Shot Classification - Frozen Baseline')
    
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
        default=42,
        help='Random seed for base/novel split (default: 42)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("CUB-200-2011 FEW-SHOT CLASSIFICATION - FROZEN BASELINE")
    print("=" * 60)
    print(f"Data root: {args.data_root}")
    print(f"Classes file: {args.classes_file}")
    print(f"Random seed: {args.seed}")
    print(f"Device: {device}")
    print("=" * 60)

    # Create frozen baseline using the convenience function
    print("\nCreating frozen ImageNet baseline...")
    model = create_frozen_baseline(device=device)
    print("Model created: Frozen ResNet18 backbone (512-D embeddings)")
    
    # Prepare novel class data
    print("\nPreparing novel class data...")
    novel_dataset = prepare_novel_data(
        data_root=args.data_root,
        classes_file=args.classes_file,
        seed=args.seed
    )
    
    # Evaluation config
    eval_config = FewShotConfig()
    eval_config.device = device
    eval_config.n_way = min(4, len(novel_dataset.classes))  # Use number of available novel classes (max 4)
    
    # Run evaluation
    print("\nStarting few-shot evaluation...")
    results = run_few_shot_evaluation(
        model=model,
        novel_dataset=novel_dataset,
        config=eval_config
    )

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()