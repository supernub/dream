#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete pipeline for training Transformer models and building donor classifiers.
Handles large single-cell datasets with memory-efficient processing.
"""

import os
import json
import argparse
import subprocess
import time
import torch
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {cmd}")
    print(f"{'='*60}")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    end_time = time.time()
    
    print(f"Duration: {end_time - start_time:.2f} seconds")
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    else:
        print(f"SUCCESS: {description} completed")
        if result.stdout:
            print(f"OUTPUT: {result.stdout}")
    
    return result

def create_data_split(h5ad_path, output_dir, split_type="donor", test_size=0.2):
    """Create data split for training."""
    split_json = os.path.join(output_dir, f"{split_type}_split.json")
    
    cmd = f"""
    python scripts/data_split.py \
        --h5ad_path {h5ad_path} \
        --output_path {split_json} \
        --split_type {split_type} \
        --test_size {test_size}
    """
    
    run_command(cmd, f"Creating {split_type}-level data split")
    return split_json

def train_transformer(h5ad_path, split_json, output_dir, memmap_dir="", 
                    embedding_dim=128, depth=4, epochs=10, batch_size=64, 
                    gradient_accumulation_steps=1):
    """Train Transformer model with multi-GPU support."""
    model_dir = os.path.join(output_dir, "transformer_model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Use accelerate launch for multi-GPU training
    cmd = f"""
    accelerate launch --num_processes {torch.cuda.device_count() if torch.cuda.is_available() else 1} \
        scripts/train_transformer.py \
        --h5ad_path {h5ad_path} \
        --split_json {split_json} \
        --output_dir {model_dir} \
        --memmap_dir {memmap_dir} \
        --embedding_dim {embedding_dim} \
        --depth {depth} \
        --epochs {epochs} \
        --batch_size {batch_size} \
        --gradient_accumulation_steps {gradient_accumulation_steps} \
        --amp
    """
    
    run_command(cmd, "Training Transformer model with multi-GPU")
    return model_dir

def extract_embeddings(h5ad_path, split_json, checkpoint_path, output_dir, 
                      memmap_dir="", k_samples=1000, num_repetitions=1):
    """Extract cell embeddings and predictions with optimized sampling."""
    embeddings_dir = os.path.join(output_dir, "embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    
    cmd = f"""
    python scripts/extract_embeddings_optimized.py \
        --h5ad_path {h5ad_path} \
        --split_json {split_json} \
        --checkpoint_path {checkpoint_path} \
        --output_dir {embeddings_dir} \
        --memmap_dir {memmap_dir} \
        --k_samples {k_samples} \
        --num_repetitions {num_repetitions}
    """
    
    run_command(cmd, "Extracting cell embeddings with optimized sampling")
    return embeddings_dir

def train_donor_classifier(h5ad_path, split_json, embeddings_dir, output_dir, k_samples=1000, num_repetitions=1):
    """Train donor-level classifier."""
    classifier_dir = os.path.join(output_dir, "donor_classifier")
    os.makedirs(classifier_dir, exist_ok=True)
    
    embeddings_path = os.path.join(embeddings_dir, "cell_embeddings.npy")
    predictions_path = os.path.join(embeddings_dir, "cell_predictions.npy")
    labels_path = os.path.join(embeddings_dir, "cell_labels.npy")
    
    cmd = f"""
    python scripts/donor_classifier.py \
        --h5ad_path {h5ad_path} \
        --split_json {split_json} \
        --embeddings_path {embeddings_path} \
        --predictions_path {predictions_path} \
        --labels_path {labels_path} \
        --output_dir {classifier_dir} \
        --k_samples {k_samples} \
        --num_repetitions {num_repetitions}
    """
    
    run_command(cmd, "Training donor-level classifier with QWK evaluation")
    return classifier_dir

def main():
    parser = argparse.ArgumentParser(description="Run complete pipeline for ADNC prediction")
    
    # Data arguments
    parser.add_argument("--h5ad_path", required=True, help="Path to h5ad file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--memmap_dir", default="", help="Path to memmap directory (optional)")
    
    # Pipeline arguments
    parser.add_argument("--split_type", choices=["cell", "donor"], default="donor", 
                       help="Type of data split")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set fraction")
    parser.add_argument("--k_samples", type=int, default=50, 
                       help="Number of cells to sample per donor-celltype (reduced for test case)")
    parser.add_argument("--num_repetitions", type=int, default=3,
                       help="Number of independent sampling repetitions for training donors")
    
    # Model arguments
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension (reduced for test case)")
    parser.add_argument("--depth", type=int, default=2, help="Transformer depth (reduced for test case)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (reduced for test case)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (reduced for test case)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Pipeline steps
    parser.add_argument("--skip_split", action="store_true", help="Skip data splitting")
    parser.add_argument("--skip_transformer", action="store_true", help="Skip Transformer training")
    parser.add_argument("--skip_embeddings", action="store_true", help="Skip embedding extraction")
    parser.add_argument("--skip_classifier", action="store_true", help="Skip donor classifier")
    
    args = parser.parse_args()
    
    print("="*80)
    print("ADNC PREDICTION PIPELINE")
    print("="*80)
    print(f"Input data: {args.h5ad_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Split type: {args.split_type}")
    print(f"Test size: {args.test_size}")
    print(f"K samples: {args.k_samples}")
    print("="*80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Create data split
    if not args.skip_split:
        split_json = create_data_split(
            args.h5ad_path, args.output_dir, args.split_type, args.test_size
        )
    else:
        split_json = os.path.join(args.output_dir, f"{args.split_type}_split.json")
        if not os.path.exists(split_json):
            raise FileNotFoundError(f"Split file not found: {split_json}")
    
    # Step 2: Train Transformer model
    if not args.skip_transformer:
        model_dir = train_transformer(
            args.h5ad_path, split_json, args.output_dir, args.memmap_dir,
            args.embedding_dim, args.depth, args.epochs, args.batch_size,
            args.gradient_accumulation_steps
        )
        checkpoint_path = os.path.join(model_dir, "best_model.pt")
    else:
        # Look for existing model
        model_dir = os.path.join(args.output_dir, "transformer_model")
        checkpoint_path = os.path.join(model_dir, "best_model.pt")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    
    # Step 3: Extract embeddings with optimized sampling
    if not args.skip_embeddings:
        embeddings_dir = extract_embeddings(
            args.h5ad_path, split_json, checkpoint_path, args.output_dir,
            args.memmap_dir, args.k_samples, args.num_repetitions
        )
    else:
        embeddings_dir = os.path.join(args.output_dir, "embeddings")
        if not os.path.exists(embeddings_dir):
            raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
    
    # Step 4: Train donor classifier with QWK evaluation
    if not args.skip_classifier:
        classifier_dir = train_donor_classifier(
            args.h5ad_path, split_json, embeddings_dir, args.output_dir, args.k_samples, args.num_repetitions
        )
    else:
        classifier_dir = os.path.join(args.output_dir, "donor_classifier")
        if not os.path.exists(classifier_dir):
            raise FileNotFoundError(f"Classifier directory not found: {classifier_dir}")
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Output directory: {args.output_dir}")
    print(f"Data split: {split_json}")
    print(f"Transformer model: {model_dir}")
    print(f"Cell embeddings: {embeddings_dir}")
    print(f"Donor classifier: {classifier_dir}")
    print("="*80)
    
    # Save pipeline configuration
    config = {
        'h5ad_path': args.h5ad_path,
        'output_dir': args.output_dir,
        'split_type': args.split_type,
        'test_size': args.test_size,
        'k_samples': args.k_samples,
        'embedding_dim': args.embedding_dim,
        'depth': args.depth,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'split_json': split_json,
        'model_dir': model_dir,
        'embeddings_dir': embeddings_dir,
        'classifier_dir': classifier_dir
    }
    
    with open(os.path.join(args.output_dir, 'pipeline_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Pipeline configuration saved to: {os.path.join(args.output_dir, 'pipeline_config.json')}")

if __name__ == "__main__":
    main()
