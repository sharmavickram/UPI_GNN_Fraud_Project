# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Graph Neural Network (GNN) for UPI Fraud Detection** using PyTorch Geometric. The project builds a transaction graph where nodes are UPI Virtual Payment Addresses (VPAs) and edges represent transactions between senders and receivers.

## Running the Project

```bash
# Activate virtual environment
source venv/bin/activate

# Run the full pipeline
python src/main.py
```

The pipeline:
1. Loads raw transaction data from `data/raw/upi_transactions.csv`
2. Engineers fraud-relevant features (transaction velocity, time-based, categorical)
3. Builds a graph structure (sender_vpa → receiver_vpa edges)
4. Trains a GraphSAGE model with class imbalance handling
5. Saves best model to `models/saved_weights/upi_gnn_best.pth`
6. Generates visualizations in `outputs/` (feature importance, PR curve, t-SNE embeddings)

## Architecture

- **Model**: `src/models/model.py` - GraphSAGE with GraphNorm and residual connections
- **Config**: `src/models/config.yaml` - hyperparameters (hidden_channels, learning_rate, fraud_weight)
- **Feature Engineering**: `src/features.py` - preprocessing, scaling, one-hot encoding
- **Graph Construction**: `src/data_loader.py` - converts DataFrame to PyTorch Geometric Data object
- **Training**: `src/utils/trainer.py` - standard forward/backward pass with weighted CrossEntropyLoss
- **Evaluation**: `src/utils/metrics.py` - F1/precision/recall with dynamic threshold optimization, confusion matrix, t-SNE visualization

## Key Design Decisions

- **GraphSAGE over GCN**: Chosen for inductive capability - can generalize to new VPAs without retraining
- **Edge-level classification**: Model predicts fraud per transaction (edge), not per node
- **Class imbalance**: Fraud is ~0.2% of transactions; uses `fraud_weight` in loss function to prioritize recall
- **Synthetic VPA generation**: Missing VPAs are synthesized from bank_state combinations

## Data Format

Expected columns in `data/raw/upi_transactions.csv`:
- `sender_bank`, `sender_state`, `receiver_bank`, `merchant_category`
- `amount (inr)`, `timestamp`, `day_of_week`, `device_type`
- `fraud_flag` or `is_fraud` (label)

## Output Files

- `models/saved_weights/upi_gnn_best.pth` - Best model weights
- `outputs/feature_importance.png` - Feature importance bar chart
- `outputs/pr_curve.png` - Precision-Recall curve
- `outputs/node_embeddings.png` - t-SNE visualization of node embeddings
