#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build donor-level classifier using cell embeddings.
For each donor, aggregates cell embeddings by cell type and trains MLP classifier.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, cohen_kappa_score
from collections import defaultdict, Counter
from tqdm import tqdm

class DonorEmbeddingDataset(Dataset):
    """Dataset for donor-level embeddings."""
    
    def __init__(self, donor_embeddings, donor_labels):
        self.donor_embeddings = donor_embeddings
        self.donor_labels = donor_labels
        self.donors = list(donor_embeddings.keys())
    
    def __len__(self):
        return len(self.donors)
    
    def __getitem__(self, idx):
        donor = self.donors[idx]
        embedding = self.donor_embeddings[donor]
        label = self.donor_labels[donor]
        return torch.FloatTensor(embedding), torch.LongTensor([label])

class DonorMLP(nn.Module):
    """MLP classifier for donor-level predictions."""
    
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.1):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def load_cell_data(h5ad_path, embeddings_path, predictions_path, labels_path):
    """Load cell embeddings, predictions, and metadata."""
    print("Loading cell data...")
    
    # Load embeddings and predictions
    embeddings = np.load(embeddings_path)
    predictions = np.load(predictions_path)
    labels = np.load(labels_path)
    
    print(f"Loaded {len(embeddings)} cells")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Load metadata from h5ad
    adata = sc.read_h5ad(h5ad_path, backed='r')
    obs = adata.obs.copy()
    
    # Extract donor and cell type information
    # Handle different column names for test case vs full dataset
    if 'Donor_space_ID' in obs.columns:
        donors = obs['Donor_space_ID'].astype(str).values
    elif 'Donor ID' in obs.columns:
        donors = obs['Donor ID'].astype(str).values
    else:
        raise KeyError(f"Neither 'Donor_space_ID' nor 'Donor ID' found in obs. Available columns: {list(obs.columns)}")
    
    celltypes = obs['Subclass'].astype(str).values
    adnc_labels = obs['ADNC'].astype(str).values
    
    # Close file
    try:
        adata.file.close()
    except:
        pass
    
    return embeddings, predictions, labels, donors, celltypes, adnc_labels

def aggregate_by_donor_celltype(embeddings, predictions, labels, donors, celltypes, adnc_labels, 
                               global_indices, split_json, k_samples=1000, num_repetitions=1):
    """
    Aggregate cell embeddings by donor and cell type with train/test split awareness.
    Handles multiple repetitions for training donors.
    
    Args:
        embeddings: Cell embeddings (N, D)
        predictions: Cell predictions (N,)
        labels: Cell labels (N,)
        donors: Donor IDs (N,)
        celltypes: Cell type IDs (N,)
        adnc_labels: ADNC labels (N,)
        global_indices: Global cell indices (N,)
        split_json: Path to split JSON file
        k_samples: Number of cells to sample per donor-celltype combination for training donors
        num_repetitions: Number of independent sampling repetitions for training donors
    """
    print("Aggregating by donor and cell type with train/test split awareness...")
    
    # Load split information
    with open(split_json, 'r') as f:
        split_data = json.load(f)
    
    train_donors = set(split_data.get('train_donors', []))
    test_donors = set(split_data.get('test_donors', []))
    
    print(f"Train donors: {len(train_donors)}")
    print(f"Test donors: {len(test_donors)}")
    
    # Create mapping from global indices to metadata
    donor_map = {}
    celltype_map = {}
    adnc_map = {}
    
    for i, gidx in enumerate(global_indices):
        if gidx < len(donors):
            donor_map[i] = donors[gidx]
            celltype_map[i] = celltypes[gidx]
            adnc_map[i] = adnc_labels[gidx]
    
    # Group cells by donor and cell type
    donor_celltype_groups = defaultdict(lambda: defaultdict(list))
    
    for i in range(len(embeddings)):
        if i in donor_map:
            donor = donor_map[i]
            celltype = celltype_map[i]
            adnc = adnc_map[i]
            
            donor_celltype_groups[donor][celltype].append({
                'embedding': embeddings[i],
                'prediction': predictions[i],
                'label': labels[i],
                'adnc': adnc
            })
    
    print(f"Found {len(donor_celltype_groups)} donors")
    
    # Get unique cell types
    all_celltypes = set()
    for donor_data in donor_celltype_groups.values():
        all_celltypes.update(donor_data.keys())
    all_celltypes = sorted(list(all_celltypes))
    
    print(f"Found {len(all_celltypes)} unique cell types")
    
    # Create donor embeddings with different strategies for train/test
    donor_embeddings = {}
    donor_labels = {}
    donor_split_info = {}
    
    for donor, celltype_data in donor_celltype_groups.items():
        is_train_donor = donor in train_donors
        celltype_embeddings = []
        
        for celltype in all_celltypes:
            if celltype in celltype_data:
                cells = celltype_data[celltype]
                
                if is_train_donor and len(cells) > k_samples:
                    # For training donors, sample k cells
                    sampled_cells = np.random.choice(len(cells), k_samples, replace=False)
                    cells = [cells[i] for i in sampled_cells]
                # For test donors, use all cells (no sampling)
                
                # Compute mean embedding for this cell type
                cell_embeddings = np.array([cell['embedding'] for cell in cells])
                mean_embedding = np.mean(cell_embeddings, axis=0)
                celltype_embeddings.append(mean_embedding)
            else:
                # No cells of this type for this donor, use zero embedding
                celltype_embeddings.append(np.zeros(embeddings.shape[1]))
        
        # Concatenate all cell type embeddings
        donor_embedding = np.concatenate(celltype_embeddings)
        donor_embeddings[donor] = donor_embedding
        
        # Get donor label (most common ADNC label)
        all_adnc_labels = []
        for celltype_cells in celltype_data.values():
            all_adnc_labels.extend([cell['adnc'] for cell in celltype_cells])
        
        if all_adnc_labels:
            donor_label = Counter(all_adnc_labels).most_common(1)[0][0]
            # Convert to numeric label
            adnc_mapping = {'Not AD': 0, 'Low': 1, 'Intermediate': 2, 'High': 3}
            donor_labels[donor] = adnc_mapping.get(donor_label, 0)
        else:
            donor_labels[donor] = 0
        
        # Record split information
        donor_split_info[donor] = {
            'is_train': is_train_donor,
            'n_cells': sum(len(cells) for cells in celltype_data.values()),
            'sampling_applied': is_train_donor
        }
    
    print(f"Created embeddings for {len(donor_embeddings)} donors")
    print(f"Donor embedding dimension: {list(donor_embeddings.values())[0].shape[0]}")
    
    return donor_embeddings, donor_labels, all_celltypes, donor_split_info

def train_donor_classifier(donor_embeddings, donor_labels, donor_split_info, args, num_repetitions=3):
    """Train MLP classifier on donor embeddings using existing split with repetitions for training donors."""
    print("Training donor classifier...")
    
    # Prepare data
    donors = list(donor_embeddings.keys())
    
    print(f"Total donors: {len(donors)}")
    print(f"Embedding dimension: {list(donor_embeddings.values())[0].shape[0]}")
    
    # Use existing split from donor_split_info
    train_donors = [donor for donor in donors if donor_split_info[donor]['is_train']]
    test_donors = [donor for donor in donors if not donor_split_info[donor]['is_train']]
    
    print(f"Train donors: {len(train_donors)}")
    print(f"Test donors: {len(test_donors)}")
    
    # Create repetitions for training donors
    train_embeddings_list = []
    train_labels_list = []
    
    for donor in train_donors:
        for _ in range(num_repetitions):
            train_embeddings_list.append(donor_embeddings[donor])
            train_labels_list.append(donor_labels[donor])
    
    # Test donors (no repetitions)
    test_embeddings = np.array([donor_embeddings[d] for d in test_donors])
    test_labels = np.array([donor_labels[d] for d in test_donors])
    
    # Convert to arrays
    train_embeddings = np.array(train_embeddings_list)
    train_labels = np.array(train_labels_list)
    
    print(f"Training samples: {len(train_embeddings)} (2 donors × 3 repetitions = 6 samples)")
    print(f"Test samples: {len(test_embeddings)} (2 donors × 1 sample each = 2 samples)")
    print(f"Label distribution: {np.bincount(train_labels)}")
    print(f"Input feature dimension: {train_embeddings.shape[1]} (24 cell types × 16 embedding dim = 384)")
    
    print(f"Train donors: {len(train_donors)}")
    print(f"Test donors: {len(test_donors)}")
    
    
    # Create datasets
    train_dataset = DonorEmbeddingDataset(
        {d: donor_embeddings[d] for d in train_donors},
        {d: donor_labels[d] for d in train_donors}
    )
    test_dataset = DonorEmbeddingDataset(
        {d: donor_embeddings[d] for d in test_donors},
        {d: donor_labels[d] for d in test_donors}
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = DonorMLP(
        input_dim=train_embeddings.shape[1],
        hidden_dims=args.hidden_dims,
        num_classes=4,  # ADNC classes
        dropout=args.dropout
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_embeddings, batch_labels in train_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_embeddings)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}, Acc: {100*correct/total:.2f}%")
    
    # Evaluate
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_embeddings, batch_labels in test_loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_labels = batch_labels.squeeze().to(device)
            
            outputs = model(batch_embeddings)
            _, predicted = torch.max(outputs.data, 1)
            
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # Calculate metrics
    test_accuracy = test_correct / test_total
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    test_qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    print(f"\nTest Results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Quadratic Weighted Kappa (QWK): {test_qwk:.4f}")
    # Create dynamic target names based on actual classes
    unique_labels = sorted(np.unique(all_labels))
    adnc_mapping = {0: 'Not AD', 1: 'Low', 2: 'Intermediate', 3: 'High'}
    target_names = [adnc_mapping.get(label, f'Class_{label}') for label in unique_labels]
    
    print(f"Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, labels=unique_labels))
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'donor_classifier.pt'))
    
    # Save results
    results = {
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_qwk': test_qwk,
        'n_train_donors': len(train_donors),
        'n_test_donors': len(test_donors),
        'embedding_dim': train_embeddings.shape[1],
        'model_config': {
            'hidden_dims': args.hidden_dims,
            'dropout': args.dropout
        },
        'sampling_strategy': 'training_donors_sampled_test_donors_all'
    }
    
    with open(os.path.join(args.output_dir, 'donor_classifier_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Model and results saved to {args.output_dir}")
    
    return model, results

def main():
    parser = argparse.ArgumentParser(description="Train donor-level classifier")
    
    # Data arguments
    parser.add_argument("--h5ad_path", required=True, help="Path to h5ad file")
    parser.add_argument("--split_json", required=True, help="Path to split JSON file")
    parser.add_argument("--embeddings_path", required=True, help="Path to cell embeddings")
    parser.add_argument("--predictions_path", required=True, help="Path to cell predictions")
    parser.add_argument("--labels_path", required=True, help="Path to cell labels")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    # Sampling arguments
    parser.add_argument("--k_samples", type=int, default=1000, 
                       help="Number of cells to sample per donor-celltype combination")
    parser.add_argument("--num_repetitions", type=int, default=1,
                       help="Number of independent sampling repetitions for training donors")
    
    # Model arguments
    parser.add_argument("--hidden_dims", nargs='+', type=int, default=[512, 256], 
                       help="Hidden layer dimensions")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*60)
    print("TRAINING DONOR-LEVEL CLASSIFIER")
    print("="*60)
    
    # Load donor embeddings from Step 3
    print("Loading donor embeddings from Step 3...")
    
    # Load donor embeddings and labels
    donor_embeddings = np.load(args.embeddings_path, allow_pickle=True).item()
    donor_labels = np.load(args.predictions_path, allow_pickle=True).item()
    
    # Load metadata to get split info and num_repetitions
    metadata_path = os.path.join(os.path.dirname(args.embeddings_path), 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    donor_split_info = metadata['donor_split_info']
    num_repetitions = metadata.get('num_repetitions', 1)
    
    print(f"Loaded {len(donor_embeddings)} donor embeddings")
    print(f"Will create {num_repetitions} repetitions for training donors")
    
    # Train donor classifier
    model, results = train_donor_classifier(donor_embeddings, donor_labels, donor_split_info, args, num_repetitions)
    
    print("Donor classifier training completed!")

if __name__ == "__main__":
    main()
