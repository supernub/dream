#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test case example for DREAM pipeline using 4-donor test case dataset.
Demonstrates the complete pipeline with optimized parameters for small datasets.
"""

import os
import subprocess
import sys

def run_test_case_pipeline():
    """Run the complete DREAM pipeline on the test case dataset."""
    print("="*80)
    print("DREAM PIPELINE - TEST CASE EXECUTION")
    print("="*80)
    
    # Configuration for test case
    h5ad_path = "/home/spark/xinze-project/test_data/SEAAD_A9_testcase_4donors.h5ad"
    output_dir = "/home/spark/xinze-project/dream/outputs/test_case_experiment"
    
    # Check if test case dataset exists
    if not os.path.exists(h5ad_path):
        print(f"Error: Test case dataset not found at {h5ad_path}")
        print("Please run the generate_testcase_h5ad.py script first to create the test dataset.")
        return False
    
    print(f"Using test case dataset: {h5ad_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the complete pipeline with test case optimized parameters
    cmd = f"""
    python scripts/run_full_pipeline.py \
        --h5ad_path {h5ad_path} \
        --output_dir {output_dir} \
        --split_type donor \
        --test_size 0.5 \
        --k_samples 50 \
        --embedding_dim 64 \
        --depth 2 \
        --epochs 3 \
        --batch_size 16 \
        --gradient_accumulation_steps 1
    """
    
    print(f"Running command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("Test case pipeline completed successfully!")
        print(result.stdout)
        
        # Print summary
        print_summary(output_dir)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Test case pipeline failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_individual_steps():
    """Run individual pipeline steps for testing."""
    print("="*80)
    print("DREAM PIPELINE - INDIVIDUAL STEPS")
    print("="*80)
    
    h5ad_path = "/home/spark/xinze-project/test_data/SEAAD_A9_testcase_4donors.h5ad"
    output_dir = "/home/spark/xinze-project/dream/outputs/test_case_individual"
    
    if not os.path.exists(h5ad_path):
        print(f"Error: Test case dataset not found at {h5ad_path}")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Data splitting
    print("\n1. Creating data split...")
    split_cmd = f"""
    python scripts/data_split.py \
        --h5ad_path {h5ad_path} \
        --output_path {output_dir}/donor_split.json \
        --split_type donor \
        --test_size 0.5
    """
    
    try:
        subprocess.run(split_cmd, shell=True, check=True)
        print("✓ Data split completed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Data split failed: {e}")
        return False
    
    # Step 2: Transformer training
    print("\n2. Training Transformer...")
    train_cmd = f"""
    python scripts/train_transformer.py \
        --h5ad_path {h5ad_path} \
        --split_json {output_dir}/donor_split.json \
        --output_dir {output_dir}/transformer_model \
        --embedding_dim 64 \
        --depth 2 \
        --epochs 3 \
        --batch_size 16 \
        --amp
    """
    
    try:
        subprocess.run(train_cmd, shell=True, check=True)
        print("✓ Transformer training completed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Transformer training failed: {e}")
        return False
    
    # Step 3: Embedding extraction
    print("\n3. Extracting embeddings...")
    extract_cmd = f"""
    python scripts/extract_embeddings_optimized.py \
        --h5ad_path {h5ad_path} \
        --split_json {output_dir}/donor_split.json \
        --checkpoint_path {output_dir}/transformer_model/best_model.pt \
        --output_dir {output_dir}/embeddings \
        --k_samples 50
    """
    
    try:
        subprocess.run(extract_cmd, shell=True, check=True)
        print("✓ Embedding extraction completed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Embedding extraction failed: {e}")
        return False
    
    # Step 4: Donor classification
    print("\n4. Training donor classifier...")
    classifier_cmd = f"""
    python scripts/donor_classifier.py \
        --h5ad_path {h5ad_path} \
        --split_json {output_dir}/donor_split.json \
        --embeddings_path {output_dir}/embeddings/cell_embeddings.npy \
        --predictions_path {output_dir}/embeddings/cell_predictions.npy \
        --labels_path {output_dir}/embeddings/cell_labels.npy \
        --output_dir {output_dir}/donor_classifier \
        --k_samples 50
    """
    
    try:
        subprocess.run(classifier_cmd, shell=True, check=True)
        print("✓ Donor classifier training completed")
    except subprocess.CalledProcessError as e:
        print(f"✗ Donor classifier training failed: {e}")
        return False
    
    print("\n✓ All individual steps completed successfully!")
    print_summary(output_dir)
    return True

def print_summary(output_dir):
    """Print summary of results."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    # Check if files exist
    files_to_check = [
        "donor_split.json",
        "transformer_model/best_model.pt",
        "embeddings/cell_embeddings.npy",
        "donor_classifier/donor_classifier_results.json"
    ]
    
    print("Generated files:")
    for file_path in files_to_check:
        full_path = os.path.join(output_dir, file_path)
        if os.path.exists(full_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
    
    # Print results if available
    results_file = os.path.join(output_dir, "donor_classifier/donor_classifier_results.json")
    if os.path.exists(results_file):
        print(f"\nResults from {results_file}:")
        try:
            import json
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"  Test Accuracy: {results.get('test_accuracy', 'N/A'):.4f}")
            print(f"  Test F1 Score: {results.get('test_f1', 'N/A'):.4f}")
            print(f"  Test QWK: {results.get('test_qwk', 'N/A'):.4f}")
            print(f"  Train Donors: {results.get('n_train_donors', 'N/A')}")
            print(f"  Test Donors: {results.get('n_test_donors', 'N/A')}")
            
        except Exception as e:
            print(f"  Error reading results: {e}")

def main():
    """Main function to run test case examples."""
    print("DREAM Test Case Examples")
    print("Choose an option:")
    print("1. Run complete pipeline")
    print("2. Run individual steps")
    print("3. Exit")
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        run_test_case_pipeline()
    elif choice == "2":
        run_individual_steps()
    elif choice == "3":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
