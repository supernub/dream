#!/bin/bash
# -*- coding: utf-8 -*-
"""
DREAM Pipeline - Test Case Execution Script
Runs the complete pipeline on the 8-donor test case dataset (4 train + 4 test donors).

GPU Configuration Options:
  --gpu-ids GPU_IDS     Specify which GPU IDs to use (e.g., '0,1,2,3')
  --num-gpus NUM        Use first N GPUs (alternative to --gpu-ids)
  --help                Show usage information

Examples:
  ./run_test_case_pipeline.sh                    # Use all available GPUs
  ./run_test_case_pipeline.sh --gpu-ids 0,1     # Use GPUs 0 and 1
  ./run_test_case_pipeline.sh --num-gpus 2      # Use first 2 GPUs
"""

set -e  # Exit on any error

# Function to display usage information
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -g, --gpu-ids GPU_IDS     Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')"
    echo "  -n, --num-gpus NUM        Number of GPUs to use (alternative to --gpu-ids)"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use all available GPUs"
    echo "  $0 --gpu-ids 0,1                     # Use GPUs 0 and 1"
    echo "  $0 --num-gpus 2                      # Use first 2 GPUs"
    echo "  $0 --gpu-ids 2,3,4,5                # Use GPUs 2, 3, 4, and 5"
    echo ""
}

# Default values
GPU_IDS=""
NUM_GPUS=""
USE_ALL_GPUS=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpu-ids)
            GPU_IDS="$2"
            USE_ALL_GPUS=false
            shift 2
            ;;
        -n|--num-gpus)
            NUM_GPUS="$2"
            USE_ALL_GPUS=false
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set CUDA_VISIBLE_DEVICES based on parameters
if [ "$USE_ALL_GPUS" = true ]; then
    echo "Using all available GPUs"
    # Don't set CUDA_VISIBLE_DEVICES to use all GPUs
    ACCELERATE_GPU_IDS=""
elif [ -n "$GPU_IDS" ]; then
    echo "Using specified GPU IDs: $GPU_IDS"
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    ACCELERATE_GPU_IDS="$GPU_IDS"
elif [ -n "$NUM_GPUS" ]; then
    echo "Using first $NUM_GPUS GPUs"
    # Generate GPU IDs from 0 to (NUM_GPUS-1)
    GPU_LIST=""
    for ((i=0; i<NUM_GPUS; i++)); do
        if [ $i -eq 0 ]; then
            GPU_LIST="$i"
        else
            GPU_LIST="$GPU_LIST,$i"
        fi
    done
    export CUDA_VISIBLE_DEVICES="$GPU_LIST"
    ACCELERATE_GPU_IDS="$GPU_LIST"
fi

echo "================================================================"
echo "DREAM PIPELINE - TEST CASE EXECUTION"
echo "================================================================"

# Display current GPU configuration
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is set to: $CUDA_VISIBLE_DEVICES"
else
    echo "CUDA_VISIBLE_DEVICES is not set (using all available GPUs)"
fi
echo ""

# Configuration
H5AD_PATH="/home/spark/xinze-project/test_data/SEAAD_A9_testcase_8donors.h5ad"
OUTPUT_DIR="/home/spark/xinze-project/dream/outputs/test_case_pipeline_8donors"
SCRIPT_DIR="/home/spark/xinze-project/dream/scripts"

# Check if test case dataset exists
if [ ! -f "$H5AD_PATH" ]; then
    echo "Error: Test case dataset not found at $H5AD_PATH"
    echo "Please run the generate_testcase_h5ad.py script first to create the test dataset."
    exit 1
fi

echo "Using 8-donor test case dataset: $H5AD_PATH"
echo "Output directory: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Change to dream directory
cd /home/spark/xinze-project/dream

echo ""
echo "================================================================"
echo "STEP 1: DATA SPLITTING"
echo "================================================================"

python scripts/data_split.py \
    --h5ad_path "$H5AD_PATH" \
    --output_path "$OUTPUT_DIR/donor_split.json" \
    --split_type donor \
    --test_size 0.5 \
    --donor_col "Donor_space_ID" \
    --adnc_col "ADNC"

if [ $? -eq 0 ]; then
    echo "✓ Data splitting completed successfully"
else
    echo "✗ Data splitting failed"
    exit 1
fi

echo ""
echo "================================================================"
echo "STEP 2: TRANSFORMER TRAINING (MULTI-GPU)"
echo "================================================================"

# Determine number of processes for transformer training
if [ -n "$ACCELERATE_GPU_IDS" ]; then
    # Count the number of GPUs specified
    NUM_GPUS_TRAIN=$(echo "$ACCELERATE_GPU_IDS" | tr ',' '\n' | wc -l)
    echo "Using $NUM_GPUS_TRAIN GPU(s) for transformer training (GPUs: $ACCELERATE_GPU_IDS)"
    
    # Use accelerate launch with explicit GPU configuration
    accelerate launch --num_processes $NUM_GPUS_TRAIN --gpu_ids $ACCELERATE_GPU_IDS \
        scripts/train_transformer.py \
        --h5ad_path "$H5AD_PATH" \
        --split_json "$OUTPUT_DIR/donor_split.json" \
        --output_dir "$OUTPUT_DIR/transformer_model" \
        --embedding_dim 16 \
        --depth 1 \
        --epochs 1 \
        --batch_size 64 \
        --max_seq_len 1024
else
    # Use all available GPUs
    NUM_GPUS_TRAIN=$(nvidia-smi --list-gpus | wc -l)
    echo "Using $NUM_GPUS_TRAIN GPU(s) for transformer training (all available GPUs)"
    
    accelerate launch --num_processes $NUM_GPUS_TRAIN \
        scripts/train_transformer.py \
        --h5ad_path "$H5AD_PATH" \
        --split_json "$OUTPUT_DIR/donor_split.json" \
        --output_dir "$OUTPUT_DIR/transformer_model" \
        --embedding_dim 16 \
        --depth 2 \
        --epochs 1 \
        --batch_size 16 \
        --max_seq_len 1024
fi

if [ $? -eq 0 ]; then
    echo "✓ Transformer training completed successfully"
else
    echo "✗ Transformer training failed"
    exit 1
fi

echo ""
echo "================================================================"
echo "STEP 3: EMBEDDING EXTRACTION (MULTI-GPU)"
echo "================================================================"

# Determine number of processes based on GPU configuration
if [ -n "$ACCELERATE_GPU_IDS" ]; then
    # Count the number of GPUs specified
    NUM_GPUS=$(echo "$ACCELERATE_GPU_IDS" | tr ',' '\n' | wc -l)
    echo "Using $NUM_GPUS GPU(s) for embedding extraction (GPUs: $ACCELERATE_GPU_IDS)"
    
    # Use accelerate launch with explicit GPU configuration
    accelerate launch --num_processes $NUM_GPUS --gpu_ids $ACCELERATE_GPU_IDS \
        scripts/extract_embeddings_optimized.py \
        --h5ad_path "$H5AD_PATH" \
        --split_json "$OUTPUT_DIR/donor_split.json" \
        --checkpoint_path "$OUTPUT_DIR/transformer_model/best_model.pt" \
        --output_dir "$OUTPUT_DIR/embeddings" \
        --k_samples 50 \
        --num_repetitions 3
else
    # Use all available GPUs
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Using $NUM_GPUS GPU(s) for embedding extraction (all available GPUs)"
    
    accelerate launch --num_processes $NUM_GPUS \
        scripts/extract_embeddings_optimized.py \
        --h5ad_path "$H5AD_PATH" \
        --split_json "$OUTPUT_DIR/donor_split.json" \
        --checkpoint_path "$OUTPUT_DIR/transformer_model/best_model.pt" \
        --output_dir "$OUTPUT_DIR/embeddings" \
        --k_samples 50 \
        --num_repetitions 3
fi

if [ $? -eq 0 ]; then
    echo "✓ Embedding extraction completed successfully"
else
    echo "✗ Embedding extraction failed"
    exit 1
fi

echo ""
echo "================================================================"
echo "STEP 4: DONOR CLASSIFICATION"
echo "================================================================"

python scripts/donor_classifier.py \
    --h5ad_path "$H5AD_PATH" \
    --split_json "$OUTPUT_DIR/donor_split.json" \
    --embeddings_path "$OUTPUT_DIR/embeddings/donor_embeddings.npy" \
    --predictions_path "$OUTPUT_DIR/embeddings/donor_labels.npy" \
    --labels_path "$OUTPUT_DIR/embeddings/cell_labels.npy" \
    --output_dir "$OUTPUT_DIR/donor_classifier" \
    --k_samples 50 \
    --num_repetitions 3

if [ $? -eq 0 ]; then
    echo "✓ Donor classifier training completed successfully"
else
    echo "✗ Donor classifier training failed"
    exit 1
fi

echo ""
echo "================================================================"
echo "PIPELINE COMPLETED SUCCESSFULLY!"
echo "================================================================"

# Print summary
echo ""
echo "Generated files:"
echo "  ✓ $OUTPUT_DIR/donor_split.json"
echo "  ✓ $OUTPUT_DIR/transformer_model/best_model.pt"
echo "  ✓ $OUTPUT_DIR/embeddings/cell_embeddings.npy"
echo "  ✓ $OUTPUT_DIR/embeddings/cell_predictions.npy"
echo "  ✓ $OUTPUT_DIR/embeddings/cell_labels.npy"
echo "  ✓ $OUTPUT_DIR/donor_classifier/donor_classifier.pt"
echo "  ✓ $OUTPUT_DIR/donor_classifier/donor_classifier_results.json"

# Print results if available
RESULTS_FILE="$OUTPUT_DIR/donor_classifier/donor_classifier_results.json"
if [ -f "$RESULTS_FILE" ]; then
    echo ""
    echo "Results from $RESULTS_FILE:"
    python -c "
import json
try:
    with open('$RESULTS_FILE', 'r') as f:
        results = json.load(f)
    print(f'  Test Accuracy: {results.get(\"test_accuracy\", \"N/A\"):.4f}')
    print(f'  Test F1 Score: {results.get(\"test_f1\", \"N/A\"):.4f}')
    print(f'  Test QWK: {results.get(\"test_qwk\", \"N/A\"):.4f}')
    print(f'  Train Donors: {results.get(\"n_train_donors\", \"N/A\")}')
    print(f'  Test Donors: {results.get(\"n_test_donors\", \"N/A\")}')
except Exception as e:
    print(f'  Error reading results: {e}')
"
fi

echo ""
echo "Test case pipeline execution completed!"
echo "Output directory: $OUTPUT_DIR"
