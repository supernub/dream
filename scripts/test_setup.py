#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify the DREAM setup and dependencies.
"""

import os
import sys
import importlib
import torch
import numpy as np
import pandas as pd
import scanpy as sc

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    required_packages = [
        'torch', 'numpy', 'pandas', 'scipy',
        'scanpy', 'anndata', 'sklearn',
        'h5py', 'tqdm', 'json'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        print("Please install missing packages: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All imports successful")
        return True

def test_torch():
    """Test PyTorch functionality."""
    print("\nTesting PyTorch...")
    
    try:
        # Test basic tensor operations
        x = torch.randn(10, 5)
        y = torch.randn(5, 3)
        z = torch.mm(x, y)
        print(f"✓ Tensor operations: {z.shape}")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            x_gpu = x.cuda()
            print(f"✓ GPU tensor: {x_gpu.device}")
        else:
            print("⚠ CUDA not available (CPU only)")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        return False

def test_data_paths():
    """Test if data paths exist."""
    print("\nTesting data paths...")
    
    data_paths = [
        "/home/spark/xinze-project/training_data/SEAAD_A9_RNAseq_DREAM.2025-07-15.h5ad",
        "/home/spark/xinze-project/training_data/SEAAD_MTG_RNAseq_DREAM.2025-07-15.h5ad"
    ]
    
    for path in data_paths:
        if os.path.exists(path):
            print(f"✓ {path}")
        else:
            print(f"✗ {path} (not found)")
    
    return True

def test_scanpy():
    """Test scanpy functionality."""
    print("\nTesting scanpy...")
    
    try:
        # Create a small test dataset
        n_obs, n_vars = 100, 50
        X = np.random.randn(n_obs, n_vars)
        obs = pd.DataFrame({
            'cell_id': [f'cell_{i}' for i in range(n_obs)],
            'donor': np.random.choice(['A', 'B', 'C'], n_obs),
            'celltype': np.random.choice(['T', 'B', 'NK'], n_obs)
        })
        var = pd.DataFrame({
            'gene_id': [f'gene_{i}' for i in range(n_vars)]
        })
        
        adata = sc.AnnData(X=X, obs=obs, var=var)
        print(f"✓ AnnData creation: {adata.shape}")
        
        # Test basic operations
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        print(f"✓ Normalization and log transform")
        
        return True
    except Exception as e:
        print(f"✗ Scanpy test failed: {e}")
        return False

def test_scripts():
    """Test if all scripts are executable."""
    print("\nTesting scripts...")
    
    script_dir = "/home/spark/xinze-project/dream/scripts"
    scripts = [
        "data_split.py",
        "train_transformer.py", 
        "extract_embeddings.py",
        "donor_classifier.py",
        "run_full_pipeline.py"
    ]
    
    for script in scripts:
        script_path = os.path.join(script_dir, script)
        if os.path.exists(script_path):
            if os.access(script_path, os.X_OK):
                print(f"✓ {script} (executable)")
            else:
                print(f"⚠ {script} (not executable)")
        else:
            print(f"✗ {script} (not found)")
    
    return True

def main():
    """Run all tests."""
    print("DREAM Setup Test")
    print("="*50)
    
    tests = [
        ("Imports", test_imports),
        ("PyTorch", test_torch),
        ("Data Paths", test_data_paths),
        ("Scanpy", test_scanpy),
        ("Scripts", test_scripts)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! DREAM is ready to use.")
        print("\nNext steps:")
        print("1. Run: python scripts/example_usage.py")
        print("2. Or run: python scripts/run_full_pipeline.py --help")
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        print("Common fixes:")
        print("- Install missing packages: pip install -r requirements.txt")
        print("- Make scripts executable: chmod +x scripts/*.py")
        print("- Check data paths in the script")

if __name__ == "__main__":
    main()
