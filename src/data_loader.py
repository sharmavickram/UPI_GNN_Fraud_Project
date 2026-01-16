# data_loader.py is the bridge that converts your preprocessed DataFrame into a mathematical graph format. 
# It maps unique UPI IDs (VPAs) to numerical indices and structures the relationships into an edge_index 
# tensor, which is the standard input for PyTorch Geometric ($PyG$).

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def build_graph(df, feature_list):
    # 1. Map VPAs (Nodes)
    all_vpas = pd.concat([df['sender_vpa'], df['receiver_vpa']]).unique()
    vpa_map = {vpa: i for i, vpa in enumerate(all_vpas)}
    
    num_nodes = len(all_vpas)
    print(f"DEBUG: Found {num_nodes} unique VPAs (Nodes)")

    # 2. Build Edge Index (Optimized)
    # Convert series to numpy array first to avoid the PyTorch UserWarning
    src = df['sender_vpa'].map(vpa_map).values
    dst = df['receiver_vpa'].map(vpa_map).values
    edge_index = torch.from_numpy(np.vstack([src, dst])).long()

    # 3. Create Feature Matrix (X) - THE CRITICAL FIX
    # Convert everything to float and handle any non-numeric leftovers
    x_df = df[feature_list].copy()
    
    # Force all columns to numeric, turning errors (strings) into NaN, then filling with 0
    for col in feature_list:
        x_df[col] = pd.to_numeric(x_df[col], errors='coerce').fillna(0)
    
    # Now convert to a float32 numpy array, then to a tensor
    x_np = x_df.values.astype(np.float32)
    x = torch.from_numpy(x_np)

    # 4. Labels (Handle potential naming issues)
    label_col = 'fraud_flag' if 'fraud_flag' in df.columns else 'is_fraud'
    y = torch.tensor(df[label_col].values, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    return data, vpa_map