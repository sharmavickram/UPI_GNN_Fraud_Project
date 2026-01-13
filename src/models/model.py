# model.py defines the neural architecture. For UPI fraud, GraphSAGE (Sample and Aggregate) is 
# the superior choice over standard GCNs because it is inductive. This means it can generate 
# embeddings for a new UPI ID (a new user or merchant) by simply looking at their transaction 
# neighbors, without needing to retrain the entire graph from scratch.

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, GraphNorm

class UPIGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(UPIGraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.norm1 = GraphNorm(hidden_channels) # Prevents features from "blurring"
        
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.norm2 = GraphNorm(hidden_channels)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels * 2 + in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index):
        # 1. Layer 1 with Residual Connection
        h1 = self.conv1(x, edge_index)
        h1 = self.norm1(h1).relu()
        
        # 2. Layer 2 
        h2 = self.conv2(h1, edge_index)
        h2 = self.norm2(h2).relu()
        
        # 3. Concatenate (Node Context + Raw Transaction Features)
        row, col = edge_index
        edge_rep = torch.cat([h2[row], h2[col]], dim=-1)
        final_input = torch.cat([edge_rep, x[row]], dim=-1)
        
        return self.classifier(final_input)