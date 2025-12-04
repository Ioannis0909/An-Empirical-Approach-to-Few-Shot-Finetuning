#!/bin/bash

################################################################################
# run_all_models.sh - Sequential Training and Evaluation of All Model Variants
################################################################################
# Usage: ./run_all_models.sh <dataset_name>
# Example: ./run_all_models.sh Bird_Dataset
#
# NOTE: This script does not work for Base_Model variants.
#
# This script runs all 6 trainable model variants:
# - One_Layer, Two_Layer, Fully_Tuned (CNN & Transformers): Training + Evaluation
#
# Requirements:
# - Dataset structure: <dataset_name>/images/ and <dataset_name>/classes.txt
# - CUDA-enabled GPU
# - Python environment with PyTorch and dependencies
################################################################################

set -e  # Exit on error

# Check if dataset name is provided
if [ $# -eq 0 ]; then
    echo "Error: No dataset name provided"
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 CUB_200_2011"
    exit 1
fi

DATASET_NAME=$1
SEED=99

# Construct dataset paths
DATA_ROOT="${DATASET_NAME}/images"
CLASSES_FILE="${DATASET_NAME}/classes.txt"

# Verify dataset exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Dataset images directory not found: $DATA_ROOT"
    exit 1
fi

if [ ! -f "$CLASSES_FILE" ]; then
    echo "Error: Classes file not found: $CLASSES_FILE"
    exit 1
fi

echo "================================================================================"
echo "FEW-SHOT LEARNING - SEQUENTIAL MODEL TRAINING"
echo "================================================================================"
echo "Dataset: $DATASET_NAME"
echo "Images directory: $DATA_ROOT"
echo "Classes file: $CLASSES_FILE"
echo "Random seed: $SEED"
echo "================================================================================"
echo ""

# Function to run a model
run_model() {
    local model_type=$1      # CNN or Transformers
    local variant=$2         # One_Layer, Two_Layer, Fully_Tuned

    local model_dir="${model_type}/${variant}"

    echo ""
    echo "================================================================================"
    echo "Running: ${model_type}/${variant}"
    echo "================================================================================"
    echo "Start time: $(date)"
    echo ""

    cd "$model_dir"

    echo "Running training and evaluation..."
    python main.py \
        --data_root "../../${DATA_ROOT}" \
        --classes_file "../../${CLASSES_FILE}" \
        --seed $SEED \
        --mode both

    local exit_code=$?
    cd - > /dev/null

    if [ $exit_code -ne 0 ]; then
        echo ""
        echo "ERROR: ${model_type}/${variant} failed with exit code $exit_code"
        echo "Stopping execution."
        exit $exit_code
    fi

    echo ""
    echo "Completed: ${model_type}/${variant}"
    echo "End time: $(date)"
    echo "================================================================================"
    echo ""
}

# Track overall start time
OVERALL_START=$(date +%s)

################################################################################
# CNN MODELS
################################################################################

echo ""
echo "################################################################################"
echo "# PART 1/2: CNN-BASED MODELS (ResNet50 Backbone)"
echo "################################################################################"
echo ""

# CNN/One_Layer - Frozen backbone + single projection layer
run_model "CNN" "One_Layer"

# CNN/Two_Layer - Frozen backbone + two-layer projection
run_model "CNN" "Two_Layer"

# CNN/Fully_Tuned - Fine-tuned ResNet50
run_model "CNN" "Fully_Tuned"

################################################################################
# TRANSFORMER MODELS
################################################################################

echo ""
echo "################################################################################"
echo "# PART 2/2: TRANSFORMER-BASED MODELS (ViT-B/16 Backbone)"
echo "################################################################################"
echo ""

# Transformers/One_Layer - Frozen backbone + single projection layer
run_model "Transformers" "One_Layer"

# Transformers/Two_Layer - Frozen backbone + two-layer projection
run_model "Transformers" "Two_Layer"

# Transformers/Fully_Tuned - Fine-tuned ViT-B/16
run_model "Transformers" "Fully_Tuned"

################################################################################
# COMPLETION
################################################################################

OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))
HOURS=$((OVERALL_DURATION / 3600))
MINUTES=$(((OVERALL_DURATION % 3600) / 60))
SECONDS=$((OVERALL_DURATION % 60))

echo ""
echo "================================================================================"
echo "ALL MODELS COMPLETED SUCCESSFULLY!"
echo "================================================================================"
echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Completion time: $(date)"
echo ""
echo "Model checkpoints and results saved in respective ./outputs/ directories"
echo "================================================================================"
echo ""

exit 0
