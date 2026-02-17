# trainer.py is the execution engine that manages the backpropagation process. For UPI fraud detection,
# the trainer's most important job is to handle class imbalance (where 99.8% of transactions are legitimate)
# so that the model doesn't just learn to ignore the few fraudulent cases.

import torch
import torch.nn as nn


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    device = next(model.parameters()).device
    data = data.to(device)

    logits = model(data.x, data.edge_index)

    loss = criterion(logits, data.y)

    loss.backward()
    optimizer.step()

    return loss.item()


def train_with_mask(model, data, optimizer, criterion, edge_mask):
    """
    Train on a subset of edges specified by edge_mask.

    Args:
        model: The GNN model
        data: PyTorch Geometric Data object
        optimizer: Optimizer
        criterion: Loss function
        edge_mask: Boolean mask indicating which edges to train on
    """
    model.train()
    optimizer.zero_grad()

    device = next(model.parameters()).device
    data = data.to(device)
    edge_mask = edge_mask.to(device)

    # Forward pass - get predictions for ALL edges
    logits = model(data.x, data.edge_index)

    # Compute loss only on the masked edges
    loss = criterion(logits[edge_mask], data.y[edge_mask])

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