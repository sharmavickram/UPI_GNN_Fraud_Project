# trainer.py is the execution engine that manages the backpropagation process. For UPI fraud detection, 
# the trainer's most important job is to handle class imbalance (where 99.8% of transactions are legitimate) '
# 'so that the model doesn't just learn to ignore the few fraudulent cases.

import torch
import torch.nn as nn

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    # 1. Ensure all graph components are on the same device as the model
    # (x, edge_index, and y)
    device = next(model.parameters()).device
    data = data.to(device)
    
    # 2. Forward pass
    logits = model(data.x, data.edge_index)
    
    # 3. Compute loss
    # Our model predicts for every edge (transaction), so we compare against data.y
    loss = criterion(logits, data.y)
    
    # 4. Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def get_criterion(fraud_weight, device):
    """
    Creates a loss function that penalizes missing fraud more heavily.
    """
    # Weight for Legit (0) is 1.0, Weight for Fraud (1) is fraud_weight
    weights = torch.tensor([1.0, float(fraud_weight)]).to(device)
    return nn.CrossEntropyLoss(weight=weights)

def save_checkpoint(model, epoch, path='models/checkpoints/'):
    import os
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), f"{path}checkpoint_epoch_{epoch}.pth")