# data_loader.py is the bridge that converts your preprocessed DataFrame into a mathematical graph format. 
# It maps unique UPI IDs (VPAs) to numerical indices and structures the relationships into an edge_index 
# tensor, which is the standard input for PyTorch Geometric ($PyG$).

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

def build_graph(df, feature_cols):
    """
    Constructs the graph using banks as nodes and transactions as edges.
    feature_cols: List of 17 features (numeric + one-hot encoded)
    """
    # 1. Column Definitions
    sender_col = 'sender_bank'
    receiver_col = 'receiver_bank'
    label_col = 'fraud_flag'
    
    # 2. Map Banks to Unique IDs
    all_entities = pd.concat([df[sender_col], df[receiver_col]]).unique()
    entity_to_id = {entity: i for i, entity in enumerate(all_entities)}
    
    # 3. Build Edge Index (Source -> Target)
    # Map bank names to the numeric IDs we just created
    edge_index = torch.tensor(
        np.array([
            df[sender_col].map(entity_to_id).values,
            df[receiver_col].map(entity_to_id).values
        ]), 
        dtype=torch.long
    )
    
    # 4. Create Node Features (x)
    # We aggregate the 17 features by bank to give each node a behavioral profile
    node_features_df = df.groupby(sender_col)[feature_cols].mean()
    
    # Ensure every bank (sender or receiver) has a feature row
    node_features_df = node_features_df.reindex(all_entities).fillna(0)
    
    # Convert to tensor: Shape will be [Num_Nodes, 17]
    x = torch.tensor(node_features_df.values, dtype=torch.float)
    
    # 5. Labels (y)
    y = torch.tensor(df[label_col].values, dtype=torch.long)
    
    # 6. Build the PyG Data object
    data = Data(x=x, edge_index=edge_index, y=y)
    
    return data, entity_to_id