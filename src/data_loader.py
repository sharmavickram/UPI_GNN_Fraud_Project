# data_loader.py is the bridge that converts your preprocessed DataFrame into a mathematical graph format.
# It maps unique UPI IDs (VPAs) to numerical indices and structures the relationships into an edge_index
# tensor, which is the standard input for PyTorch Geometric ($PyG$).

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split


def build_graph(df, feature_list, add_edge_weights=True, train_mask=None, test_mask=None):
    """
    Build a graph from DataFrame with optional edge weights and train/test split.

    Args:
        df: DataFrame with transaction data
        feature_list: List of feature column names
        add_edge_weights: If True, add transaction amount as edge weight
        train_mask: Boolean mask for training edges (optional)
        test_mask: Boolean mask for test edges (optional)

    Returns:
        data: PyTorch Geometric Data object
        vpa_map: Dictionary mapping VPA to node index
    """
    # 1. Map VPAs (Nodes)
    all_vpas = pd.concat([df['sender_vpa'], df['receiver_vpa']]).unique()
    vpa_map = {vpa: i for i, vpa in enumerate(all_vpas)}

    num_nodes = len(all_vpas)
    print(f"DEBUG: Found {num_nodes} unique VPAs (Nodes)")

    # 2. Build Edge Index
    src = df['sender_vpa'].map(vpa_map).values
    dst = df['receiver_vpa'].map(vpa_map).values
    edge_index = torch.from_numpy(np.vstack([src, dst])).long()

    # 3. Create Feature Matrix (X)
    x_df = df[feature_list].copy()

    for col in feature_list:
        x_df[col] = pd.to_numeric(x_df[col], errors='coerce').fillna(0)

    x_np = x_df.values.astype(np.float32)
    x = torch.from_numpy(x_np)

    # 4. Labels
    label_col = 'fraud_flag' if 'fraud_flag' in df.columns else 'is_fraud'
    y = torch.tensor(df[label_col].values, dtype=torch.long)

    # 5. Edge Weights (transaction amount)
    if add_edge_weights and 'amount (inr)' in df.columns:
        edge_attr = torch.tensor(
            df['amount (inr)'].values.astype(np.float32),
            dtype=torch.float
        ).unsqueeze(1)  # Shape: [num_edges, 1]
    else:
        edge_attr = None

    # 6. Train/Test Edge Masks
    num_edges = len(df)
    if train_mask is None:
        train_mask = torch.ones(num_edges, dtype=torch.bool)
    if test_mask is None:
        test_mask = torch.zeros(num_edges, dtype=torch.bool)

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        edge_attr=edge_attr,
        train_mask=train_mask,
        test_mask=test_mask
    )
    return data, vpa_map


def build_graph_with_split(df, feature_list, test_size=0.2, random_state=42):
    """
    Build a graph with edge-level train/test split.

    This approach keeps all nodes in the graph but splits edges for train/test.
    This allows the model to see all node features but only train on subset of edges.

    Args:
        df: DataFrame with transaction data
        feature_list: List of feature column names
        test_size: Fraction of edges for test set
        random_state: Random seed for reproducibility

    Returns:
        data: PyTorch Geometric Data object with train/test masks
        vpa_map: Dictionary mapping VPA to node index
    """
    label_col = 'fraud_flag' if 'fraud_flag' in df.columns else 'is_fraud'

    # Stratified split on labels to preserve fraud ratio
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        stratify=df[label_col],
        random_state=random_state
    )

    # Create boolean masks
    train_mask = torch.zeros(len(df), dtype=torch.bool)
    test_mask = torch.zeros(len(df), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    print(f"Edge split: {train_mask.sum()} train, {test_mask.sum()} test")
    print(f"Fraud ratio - Train: {df.iloc[train_idx][label_col].mean():.4f}, "
          f"Test: {df.iloc[test_idx][label_col].mean():.4f}")

    return build_graph(df, feature_list, add_edge_weights=True,
                       train_mask=train_mask, test_mask=test_mask)
