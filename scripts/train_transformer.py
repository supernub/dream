#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Transformer model for ADNC prediction on large single-cell datasets.
Uses memory-efficient data loading and distributed training.
"""

import os
import json
import argparse
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed

# Import from utils
import sys
sys.path.append('/home/spark/xinze-project/dream')
from utils.dataset_varlen import VariableLengthSequenceDataset, CSRMemmapDataset, pad_collate, load_labels_and_split_only
from torch.utils.data import Dataset
from models.model_ordinal_transformer import GeneTransformerOrdinal, coral_loss, coral_predict

class TruncatedSequenceDataset(Dataset):
    """Dataset that truncates sequences to a maximum length to prevent OOM."""
    
    def __init__(self, X, labels, cell_indices, max_seq_len=2048):
        self.X = X.tocsr(copy=False)
        self.labels_all = labels
        self.ids = np.asarray(cell_indices, dtype=np.int64)
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return self.ids.shape[0]
    
    def __getitem__(self, i):
        ridx = int(self.ids[i])
        row = self.X.getrow(ridx)
        
        # Get indices and values
        indices = row.indices.astype(np.int64) + 1  # +1 for 1-based indexing
        values = row.data.astype(np.float32)
        
        # Truncate if sequence is too long
        if len(indices) > self.max_seq_len:
            # Keep the most highly expressed genes
            top_indices = np.argsort(values)[-self.max_seq_len:]
            indices = indices[top_indices]
            values = values[top_indices]
        
        seq = torch.from_numpy(indices)
        vals = torch.from_numpy(values)
        label = torch.as_tensor(self.labels_all[ridx], dtype=torch.long)
        return seq, vals, label

def setup_training(args):
    """Setup training environment and data."""
    print("Setting up training environment...")
    
    # Set random seeds
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load split
    with open(args.split_json, 'r') as f:
        split_data = json.load(f)
    
    print(f"Split info: {split_data.get('train_cells', 'unknown')} train, {split_data.get('test_cells', 'unknown')} test cells")
    
    return split_data

def load_data(args, split_data, accelerator):
    """Load training and validation data."""
    print("Loading data...")
    
    # Load labels and split indices
    y, train_idx, val_idx, num_genes, num_classes, mapping = load_labels_and_split_only(
        args.h5ad_path, args.split_json, args.label_col, None
    )
    
    print(f"Number of genes: {num_genes}")
    print(f"Number of classes: {num_classes}")
    print(f"Train cells: {len(train_idx)}")
    print(f"Validation cells: {len(val_idx)}")
    
    # Create datasets
    if args.memmap_dir and os.path.exists(args.memmap_dir):
        print("Using memmap dataset for memory efficiency")
        train_ds = CSRMemmapDataset(args.memmap_dir, y, train_idx)
        val_ds = CSRMemmapDataset(args.memmap_dir, y, val_idx)
    else:
        print("Loading data for regular dataset...")
        # Load the actual data for non-memmap case
        import scanpy as sc
        adata = sc.read_h5ad(args.h5ad_path, backed='r')
        # Convert the lazy dataset to a proper sparse matrix
        X = adata.X[:].tocsr()
        # Create datasets with sequence length limiting
        train_ds = TruncatedSequenceDataset(X, y, train_idx, max_seq_len=args.max_seq_len)
        val_ds = TruncatedSequenceDataset(X, y, val_idx, max_seq_len=args.max_seq_len)
        
        # Close the file to free memory
        try:
            adata.file.close()
        except:
            pass
    
    # Create data loaders
    collate_fn = lambda batch: pad_collate(batch, pad_idx=0)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers, 
        collate_fn=collate_fn, 
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers, 
        collate_fn=collate_fn, 
        pin_memory=True
    )
    
    return train_loader, val_loader, num_genes, num_classes, mapping

def create_model(num_genes: int, num_classes: int, args):
    """Create Transformer model."""
    print("Creating model...")
    
    model = GeneTransformerOrdinal(
        num_genes=num_genes,
        num_classes=num_classes,
        embedding_dim=args.embedding_dim,
        dim_feedforward=args.ffn_dim,
        nhead=args.nhead,
        depth=args.depth,
        dropout=args.dropout,
        pad_idx=0
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model

def train_epoch(model, train_loader, optimizer, accelerator, args):
    """Train for one epoch using Accelerate."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Only show progress bar on main process
    if accelerator.is_main_process:
        pbar = tqdm(train_loader, desc="Training")
    else:
        pbar = train_loader
    
    for batch_idx, (seqs, vals, labels, attn_mask) in enumerate(pbar):
        with accelerator.accumulate(model):
            # Forward pass
            if args.amp:
                with accelerator.autocast():
                    logits = model(seqs, vals, attn_mask=attn_mask)
                    loss = coral_loss(logits, labels, reduction="mean")
            else:
                logits = model(seqs, vals, attn_mask=attn_mask)
                loss = coral_loss(logits, labels, reduction="mean")
            
            # Backward pass
            accelerator.backward(loss)
            
            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item()
        num_batches += 1
        
        if accelerator.is_main_process and batch_idx % args.log_interval == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def evaluate(model, val_loader, accelerator, args):
    """Evaluate model on validation set using Accelerate."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        # Only show progress bar on main process
        if accelerator.is_main_process:
            pbar = tqdm(val_loader, desc="Validation")
        else:
            pbar = val_loader
        
        for seqs, vals, labels, attn_mask in pbar:
            # Forward pass
            if args.amp:
                with accelerator.autocast():
                    logits = model(seqs, vals, attn_mask=attn_mask)
                    loss = coral_loss(logits, labels, reduction="mean")
            else:
                logits = model(seqs, vals, attn_mask=attn_mask)
                loss = coral_loss(logits, labels, reduction="mean")
            
            total_loss += loss.item()
            
            # Get predictions
            preds = coral_predict(logits, threshold=0.5)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Gather predictions from all processes
    all_preds_tensor = torch.tensor(all_preds, device=accelerator.device)
    all_labels_tensor = torch.tensor(all_labels, device=accelerator.device)
    all_preds = accelerator.gather(all_preds_tensor).cpu().numpy()
    all_labels = accelerator.gather(all_labels_tensor).cpu().numpy()
    
    # Calculate metrics (only on main process)
    if accelerator.is_main_process:
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted")
        mae = mean_absolute_error(all_labels, all_preds)
        
        return avg_loss, accuracy, f1, mae
    else:
        return 0.0, 0.0, 0.0, 0.0

def save_checkpoint(model, optimizer, epoch, metrics, output_dir, args, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'model_params': {
            'embedding_dim': args.embedding_dim,
            'dim_feedforward': args.ffn_dim,
            'nhead': args.nhead,
            'depth': args.depth,
            'dropout': args.dropout
        }
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pt')
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")
    
    return checkpoint_path

def main():
    parser = argparse.ArgumentParser(description="Train Transformer for ADNC prediction with Accelerate")
    
    # Data arguments
    parser.add_argument("--h5ad_path", required=True, help="Path to h5ad file")
    parser.add_argument("--split_json", required=True, help="Path to split JSON file")
    parser.add_argument("--memmap_dir", default="", help="Path to memmap directory (optional)")
    parser.add_argument("--label_col", default="ADNC", help="ADNC column name")
    
    # Model arguments
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--ffn_dim", type=int, default=512, help="Feedforward dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--depth", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length (to prevent OOM)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (reduced for test case)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs (reduced for test case)")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers (reduced for test case)")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval (reduced for test case)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Output arguments
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if args.amp else "no"
    )
    
    if accelerator.is_main_process:
        print("="*60)
        print("TRAINING TRANSFORMER FOR ADNC PREDICTION (MULTI-GPU)")
        print("="*60)
        print(f"Device: {accelerator.device}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
    
    # Setup
    split_data = setup_training(args)
    
    # Load data
    train_loader, val_loader, num_genes, num_classes, mapping = load_data(args, split_data, accelerator)
    
    # Create model
    model = create_model(num_genes, num_classes, args)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Prepare model, optimizer, and data loaders with Accelerate
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Training loop
    best_mae = float('inf')
    train_losses = []
    val_metrics = []
    
    if accelerator.is_main_process:
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Effective batch size: {args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}")
    
    for epoch in range(1, args.epochs + 1):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, accelerator, args)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_acc, val_f1, val_mae = evaluate(model, val_loader, accelerator, args)
        
        if accelerator.is_main_process:
            val_metrics.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1': val_f1,
                'val_mae': val_mae
            })
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, MAE: {val_mae:.4f}")
            
            # Save checkpoint
            is_best = val_mae < best_mae
            if is_best:
                best_mae = val_mae
            
            # Unwrap model for saving
            unwrapped_model = accelerator.unwrap_model(model)
            save_checkpoint(unwrapped_model, optimizer, epoch, val_metrics[-1], args.output_dir, args, is_best)
    
    # Save final results (only on main process)
    if accelerator.is_main_process:
        results = {
            'train_losses': train_losses,
            'val_metrics': val_metrics,
            'best_mae': best_mae,
            'args': vars(args),
            'accelerator_config': {
                'num_processes': accelerator.num_processes,
                'mixed_precision': accelerator.mixed_precision,
                'gradient_accumulation_steps': args.gradient_accumulation_steps
            }
        }
        
        with open(os.path.join(args.output_dir, 'training_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nTraining completed!")
        print(f"Best MAE: {best_mae:.4f}")
        print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
