#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract cell embeddings and predictions from trained Transformer model.
Uses the last layer hidden states averaged as cell embeddings.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

# Import from utils and models
import sys
sys.path.append('/home/spark/xinze-project/dream')
from utils.dataset_varlen import VariableLengthSequenceDataset, CSRMemmapDataset, pad_collate, load_labels_and_split_only
from models.model_ordinal_transformer import GeneTransformerOrdinal, coral_predict

class EmbeddingExtractor:
    """Extract embeddings and predictions from Transformer model."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.embeddings = []
        self.predictions = []
        self.global_indices = []
        self.labels = []
        
        # Register hook to capture last layer hidden states
        self.hidden_states = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture hidden states."""
        def hook_fn(module, input, output):
            # output is (batch_size, seq_len, hidden_dim)
            # Take mean over sequence length to get cell embedding
            if isinstance(output, torch.Tensor):
                cell_embeddings = output.mean(dim=1)  # (batch_size, hidden_dim)
                self.hidden_states.append(cell_embeddings.detach().cpu())
        
        # Hook the last transformer layer
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
            last_layer = self.model.encoder.layers[-1]
            self.hook = last_layer.register_forward_hook(hook_fn)
        else:
            raise ValueError("Could not find encoder layers to hook")
    
    def extract_batch(self, batch):
        """Extract embeddings and predictions for a batch."""
        seqs, vals, labels, attn_mask = batch
        seqs = seqs.to(self.device, non_blocking=True)
        vals = vals.to(self.device, non_blocking=True)
        attn_mask = attn_mask.to(self.device, non_blocking=True)
        
        with torch.no_grad():
            # Forward pass to get predictions and trigger hooks
            logits = self.model(seqs, vals, attn_mask=attn_mask)
            predictions = coral_predict(logits, threshold=0.5)
            
            # Get embeddings from hooks
            if self.hidden_states:
                batch_embeddings = self.hidden_states.pop(0)
                self.embeddings.append(batch_embeddings.numpy())
            
            self.predictions.append(predictions.cpu().numpy())
            self.labels.append(labels.numpy())
    
    def get_results(self):
        """Get concatenated results."""
        if not self.embeddings:
            raise ValueError("No embeddings extracted. Make sure to run extract_batch first.")
        
        embeddings = np.concatenate(self.embeddings, axis=0)
        predictions = np.concatenate(self.predictions, axis=0)
        labels = np.concatenate(self.labels, axis=0)
        
        return embeddings, predictions, labels
    
    def cleanup(self):
        """Remove hooks."""
        if hasattr(self, 'hook'):
            self.hook.remove()

def load_model(checkpoint_path, num_genes, num_classes, device):
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Create model
    model = GeneTransformerOrdinal(
        num_genes=num_genes,
        num_classes=num_classes,
        embedding_dim=128,  # Default, will be updated from checkpoint
        dim_feedforward=512,
        nhead=8,
        depth=4,
        dropout=0.1,
        pad_idx=0
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    return model

def extract_embeddings_and_predictions(model, data_loader, device, output_dir, 
                                     partition="all", k_samples=1000):
    """Extract embeddings and predictions with sampling strategy."""
    print(f"Extracting embeddings and predictions for {partition} partition...")
    
    extractor = EmbeddingExtractor(model, device)
    
    # Process all batches
    for batch_idx, batch in enumerate(tqdm(data_loader, desc="Extracting")):
        extractor.extract_batch(batch)
        
        # Save intermediate results periodically
        if batch_idx % 100 == 0 and batch_idx > 0:
            print(f"Processed {batch_idx} batches")
    
    # Get final results
    embeddings, predictions, labels = extractor.get_results()
    extractor.cleanup()
    
    print(f"Extracted {len(embeddings)} cell embeddings")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Prediction distribution: {np.bincount(predictions)}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save embeddings
    np.save(os.path.join(output_dir, 'cell_embeddings.npy'), embeddings)
    np.save(os.path.join(output_dir, 'cell_predictions.npy'), predictions)
    np.save(os.path.join(output_dir, 'cell_labels.npy'), labels)
    
    # Save metadata
    metadata = {
        'n_cells': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'n_classes': len(np.unique(labels)),
        'prediction_distribution': np.bincount(predictions).tolist(),
        'label_distribution': np.bincount(labels).tolist(),
        'partition': partition,
        'k_samples': k_samples if partition == "train" else None
    }
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Results saved to {output_dir}")
    
    return embeddings, predictions, labels

def main():
    parser = argparse.ArgumentParser(description="Extract cell embeddings and predictions")
    
    # Data arguments
    parser.add_argument("--h5ad_path", required=True, help="Path to h5ad file")
    parser.add_argument("--split_json", required=True, help="Path to split JSON file")
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--memmap_dir", default="", help="Path to memmap directory (optional)")
    parser.add_argument("--label_col", default="ADNC", help="ADNC column name")
    
    # Inference arguments
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--partition", choices=["train", "test", "all"], default="all", 
                       help="Which partition to process")
    
    args = parser.parse_args()
    
    print("="*60)
    print("EXTRACTING CELL EMBEDDINGS AND PREDICTIONS")
    print("="*60)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load split data
    y, train_idx, val_idx, num_genes, num_classes, mapping = load_labels_and_split_only(
        args.h5ad_path, args.split_json, args.label_col, None
    )
    
    print(f"Number of genes: {num_genes}")
    print(f"Number of classes: {num_classes}")
    
    # Determine which indices to use
    if args.partition == "train":
        indices = train_idx
    elif args.partition == "test":
        indices = val_idx
    else:  # all
        indices = list(train_idx) + list(val_idx)
    
    print(f"Processing {len(indices)} cells from {args.partition} partition")
    
    # Create dataset
    if args.memmap_dir and os.path.exists(args.memmap_dir):
        dataset = CSRMemmapDataset(args.memmap_dir, y, indices)
    else:
        # Load the actual data for non-memmap case
        import scanpy as sc
        adata = sc.read_h5ad(args.h5ad_path, backed='r')
        # Convert the lazy dataset to a proper sparse matrix
        X = adata.X[:].tocsr()
        dataset = VariableLengthSequenceDataset(X, y, indices)
        
        # Close the file to free memory
        try:
            adata.file.close()
        except:
            pass
    
    # Create data loader
    collate_fn = lambda batch: pad_collate(batch, pad_idx=0)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Load model
    model = load_model(args.checkpoint_path, num_genes, num_classes, device)
    
    # Extract embeddings and predictions
    embeddings, predictions, labels = extract_embeddings_and_predictions(
        model, data_loader, device, args.output_dir
    )
    
    print("Extraction completed successfully!")

if __name__ == "__main__":
    main()
