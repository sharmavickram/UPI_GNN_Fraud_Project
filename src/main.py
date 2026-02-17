# main.py script serves as the orchestrator for your entire project.
# It imports your modular components to clean data, build the graph,
# train the GNN, and save the final results.

import os
import torch
import pandas as pd
import yaml
import numpy as np
import copy

# Import your custom modules
from features import preprocess_upi_data
from data_loader import build_graph_with_split
from models.model import UPIGraphSAGE, get_model
from utils.trainer import train, train_with_mask
from utils.metrics import evaluate_model, get_detailed_logs, plot_node_embeddings
from utils.metrics import plot_feature_importance, plot_pr_curve


def main():
    # 1. Load Hyperparameters from config
    config_path = 'src/models/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set seeds for reproducibility
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else
                          ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Starting UPI Fraud Detection Pipeline on {device}...")

    # 2. Data Acquisition
    raw_data_path = config.get('raw_data_path', 'data/raw/upi_transactions.csv')
    df_raw = pd.read_csv(raw_data_path)
    df_raw.columns = df_raw.columns.str.strip().str.lower()

    # --- SYNTHETIC VPA GENERATION ---
    # Create unique "Node IDs" based on Bank + State combinations
    print("Generating synthetic network topology...")
    df_raw['sender_vpa'] = df_raw['sender_bank'] + "_" + df_raw['sender_state']
    df_raw['receiver_vpa'] = df_raw['receiver_bank'] + "_Merchant_" + df_raw['merchant_category']

    # 3. Feature Engineering on ALL data (to get consistent features)
    print("Running feature engineering on full dataset...")
    df_processed, scaler, feature_list = preprocess_upi_data(df_raw)
    num_features = len(feature_list)

    # 4. Build graph with EDGE-LEVEL train/test split
    # This keeps all nodes visible but splits edges for evaluation
    print("Building graph with edge-level train/test split...")
    test_size = config.get('test_size', 0.2)

    data, vpa_map = build_graph_with_split(
        df_processed,
        feature_list,
        test_size=test_size,
        random_state=seed
    )

    print(f"Graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges")
    print(f"Train edges: {data.train_mask.sum()}, Test edges: {data.test_mask.sum()}")

    data = data.to(device)

    # 5. Initialize Model
    model_type = config.get('model_type', 'graphsage')
    print(f"Using model: {model_type}")

    model = get_model(
        model_type=model_type,
        in_channels=num_features,
        hidden_channels=config.get('hidden_channels', 64),
        out_channels=2,
        heads=config.get('gat_heads', 4)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    fraud_weight = config.get('fraud_weight', 75.0)
    weights = torch.tensor([1.0, float(fraud_weight)]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # 6. Training Loop
    print(f"Training for {config['epochs']} epochs...")

    patience = config.get('patience', 30)
    patience_counter = 0
    best_val_f1 = 0
    best_threshold = 0.5

    for epoch in range(1, config['epochs'] + 1):
        # Train on train_mask edges only
        loss = train_with_mask(model, data, optimizer, criterion, data.train_mask)

        if epoch % 10 == 0 or epoch == 1:
            # Evaluate on TEST set (not training set) for proper evaluation
            precision, recall, f1, auprc, epoch_threshold = evaluate_model(
                model, data, edge_mask=data.test_mask
            )
            scheduler.step(f1)

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_threshold = epoch_threshold
                patience_counter = 0
                torch.save(model.state_dict(), 'models/saved_weights/upi_gnn_best.pth')
            else:
                patience_counter += 1

            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val F1: {f1:.4f} | "
                  f"Recall: {recall:.4f} | Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # 7. Final Evaluation
    print("\n" + "="*50)
    print("FINAL PERFORMANCE REPORT")
    print("="*50)

    model.load_state_dict(torch.load('models/saved_weights/upi_gnn_best.pth'))
    model.eval()

    # Training set performance
    print("\n--- Training Set Performance ---")
    train_precision, train_recall, train_f1, train_auprc, train_threshold = evaluate_model(
        model, data, edge_mask=data.train_mask
    )
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall:    {train_recall:.4f}")
    print(f"F1:        {train_f1:.4f}")
    print(f"AUPRC:     {train_auprc:.4f}")

    tn_train, fp_train, fn_train, tp_train = get_detailed_logs(
        model, data, edge_mask=data.train_mask, threshold=best_threshold
    )
    print(f"TN: {tn_train} | TP: {tp_train} | FP: {fp_train} | FN: {fn_train}")

    # Test set performance (the true metric)
    print("\n--- Test Set Performance (FINAL METRIC) ---")
    test_precision, test_recall, test_f1, test_auprc, test_threshold = evaluate_model(
        model, data, edge_mask=data.test_mask
    )
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1:        {test_f1:.4f}")
    print(f"AUPRC:     {test_auprc:.4f}")

    tn_test, fp_test, fn_test, tp_test = get_detailed_logs(
        model, data, edge_mask=data.test_mask, threshold=best_threshold
    )
    print(f"TN: {tn_test} | TP: {tp_test} | FP: {fp_test} | FN: {fn_test}")

    # 8. Visualization
    print("\nGenerating visualizations...")
    plot_feature_importance(model, feature_list)
    plot_pr_curve(model, data, edge_mask=data.test_mask)
    plot_node_embeddings(model, data)

    print("\nAll tasks complete. Check the 'outputs/' folder for your PNG files.")


if __name__ == "__main__":
    main()
