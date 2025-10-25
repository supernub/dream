#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example usage script for the DREAM pipeline.
Demonstrates how to run the complete pipeline on A9 and MTG datasets.
"""

import os
import subprocess
import sys

def run_example(dataset_name, h5ad_path, output_dir):
    """Run the complete pipeline for a dataset."""
    print(f"\n{'='*80}")
    print(f"RUNNING PIPELINE FOR {dataset_name.upper()} DATASET")
    print(f"{'='*80}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the optimized pipeline with multi-GPU training
    cmd = f"""
    python scripts/run_full_pipeline.py \
        --h5ad_path {h5ad_path} \
        --output_dir {output_dir} \
        --split_type donor \
        --test_size 0.2 \
        --k_samples 1000 \
        --embedding_dim 128 \
        --depth 4 \
        --epochs 10 \
        --batch_size 64 \
        --gradient_accumulation_steps 2 \
        --amp
    """
    
    print(f"Running command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"SUCCESS: {dataset_name} pipeline completed")
        print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {dataset_name} pipeline failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Run examples for both A9 and MTG datasets."""
    print("DREAM Pipeline Examples")
    print("="*80)
    
    # Define dataset paths
    datasets = {
        "A9": "/home/spark/xinze-project/training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad",
        "MTG": "/home/spark/xinze-project/training_data/SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad"
    }
    
    # Check if datasets exist
    for name, path in datasets.items():
        if not os.path.exists(path):
            print(f"WARNING: {name} dataset not found at {path}")
            print("Please update the path in this script")
        else:
            print(f"✓ {name} dataset found at {path}")
    
    print("\nAvailable datasets:")
    for name, path in datasets.items():
        status = "✓" if os.path.exists(path) else "✗"
        print(f"  {status} {name}: {path}")
    
    # Ask user which dataset to run
    print("\nWhich dataset would you like to run?")
    print("1. A9 dataset")
    print("2. MTG dataset")
    print("3. Both datasets")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        if os.path.exists(datasets["A9"]):
            success = run_example("A9", datasets["A9"], "outputs/a9_example")
            if success:
                print("\n✓ A9 pipeline completed successfully!")
            else:
                print("\n✗ A9 pipeline failed")
        else:
            print("A9 dataset not found")
    
    elif choice == "2":
        if os.path.exists(datasets["MTG"]):
            success = run_example("MTG", datasets["MTG"], "outputs/mtg_example")
            if success:
                print("\n✓ MTG pipeline completed successfully!")
            else:
                print("\n✗ MTG pipeline failed")
        else:
            print("MTG dataset not found")
    
    elif choice == "3":
        print("\nRunning both datasets...")
        results = {}
        
        if os.path.exists(datasets["A9"]):
            results["A9"] = run_example("A9", datasets["A9"], "outputs/a9_example")
        else:
            print("A9 dataset not found, skipping...")
        
        if os.path.exists(datasets["MTG"]):
            results["MTG"] = run_example("MTG", datasets["MTG"], "outputs/mtg_example")
        else:
            print("MTG dataset not found, skipping...")
        
        print("\nResults summary:")
        for dataset, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {dataset}: {'Success' if success else 'Failed'}")
    
    elif choice == "4":
        print("Exiting...")
        return
    
    else:
        print("Invalid choice. Please run the script again.")
        return
    
    print("\n" + "="*80)
    print("EXAMPLE COMPLETED")
    print("="*80)
    print("Check the outputs/ directory for results")
    print("Each experiment will have:")
    print("  - transformer_model/: Trained Transformer model")
    print("  - embeddings/: Cell embeddings and predictions")
    print("  - donor_classifier/: Donor-level classifier")
    print("  - pipeline_config.json: Configuration used")

def run_multi_gpu_training():
    """Example of running multi-GPU training directly."""
    print("="*80)
    print("MULTI-GPU TRAINING EXAMPLE")
    print("="*80)
    
    h5ad_path = "/home/spark/xinze-project/training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad"
    split_json = "outputs/a9_donor_split.json"
    output_dir = "outputs/a9_multi_gpu"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run multi-GPU training
    cmd = f"""
    accelerate launch --config_file accelerate_config.yaml \
        scripts/train_transformer.py \
        --h5ad_path {h5ad_path} \
        --split_json {split_json} \
        --output_dir {output_dir} \
        --embedding_dim 256 \
        --depth 6 \
        --epochs 20 \
        --batch_size 128 \
        --gradient_accumulation_steps 4 \
        --amp
    """
    
    print(f"Running command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("Multi-GPU training completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Multi-GPU training failed: {e}")
        print(f"Error output: {e.stderr}")

if __name__ == "__main__":
    main()
    run_multi_gpu_training()
