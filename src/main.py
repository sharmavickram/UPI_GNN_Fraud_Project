# main.py script serves as the orchestrator for your entire project. It imports your modular 
# components to clean data, build the graph, train the GNN, and save the final results.

# main.py script serves as the orchestrator for your entire project. 
# It imports your modular components to clean data, build the graph, 
# train the GNN, and save the final results.

import os
import torch
import pandas as pd
import yaml
import numpy as np
import copy # Added for deepcopy

# Import your custom modules
from features import preprocess_upi_data
from data_loader import build_graph
from models.model import UPIGraphSAGE
from utils.trainer import train
from utils.metrics import evaluate_model, get_detailed_logs, plot_node_embeddings 
from utils.metrics import plot_feature_importance, plot_pr_curve

def main():
    # 1. Load Hyperparameters from config
    config_path = 'src/models/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 
                          ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"🚀 Starting UPI Fraud Detection Pipeline on {device}...")

   # 2. Data Acquisition
    raw_data_path = 'data/raw/upi_transactions.csv'
    df_raw = pd.read_csv(raw_data_path)
    df_raw.columns = df_raw.columns.str.strip().str.lower() 

    # --- SYNTHETIC VPA GENERATION ---
    # Since specific VPAs are missing, we create unique "Node IDs" 
    # based on Bank + State combinations to simulate a network.
    print("Generating synthetic network topology...")
    df_raw['sender_vpa'] = df_raw['sender_bank'] + "_" + df_raw['sender_state']
    df_raw['receiver_vpa'] = df_raw['receiver_bank'] + "_Merchant_" + df_raw['merchant_category']
    
    vpa_backup = df_raw[['sender_vpa', 'receiver_vpa']].copy()

    # --------------------------------

    # 3. Feature Engineering
    df_processed, scaler, feature_list = preprocess_upi_data(df_raw)
    num_features = len(feature_list) # Define this here!

    # 4. Graph Construction
    print("🕸️  Building transaction graph...")
    df_processed['sender_vpa'] = vpa_backup['sender_vpa']
    df_processed['receiver_vpa'] = vpa_backup['receiver_vpa']
    
    data, vpa_map = build_graph(df_processed, feature_list)
    print(f"📊 Graph Created with {data.num_nodes} nodes and {data.edge_index.shape[1]} edges.")
    
    # CRITICAL FIX: Save a deep copy of the full graph before training starts
    # This ensures 'full_graph_storage' always has all 250k nodes
    full_graph_storage = copy.deepcopy(data).to(device)
    data = data.to(device)

    # 5. Initialize Model
    model = UPIGraphSAGE(
        in_channels=num_features, 
        hidden_channels=config.get('hidden_channels', 128), 
        out_channels=2
    ).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    fraud_weight = config.get('fraud_weight', 500.0)
    weights = torch.tensor([1.0, float(fraud_weight)]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

   # 6. Training Loop
    print(f"🏋️  Training for {config['epochs']} epochs...")
    best_f1 = 0
    best_threshold = 0.5 
    
    for epoch in range(1, config['epochs'] + 1):
        loss = train(model, data, optimizer, criterion)
        
        if epoch % 10 == 0 or epoch == 1:
            precision, recall, f1, auprc, epoch_threshold = evaluate_model(model, data)
            scheduler.step(f1)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = epoch_threshold 
                torch.save(model.state_dict(), 'models/saved_weights/upi_gnn_best.pth')
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | F1: {f1:.4f} | Recall: {recall:.4f}")

    # 7. Final Detailed Report
    print("\n📊 --- FINAL PERFORMANCE REPORT ---")
    model.load_state_dict(torch.load('models/saved_weights/upi_gnn_best.pth'))
    
    # Use full_graph_storage to ensure logs are calculated on the whole dataset
    tn, fp, fn, tp = get_detailed_logs(model, full_graph_storage, threshold=best_threshold)
    
    print(f"Using Optimal Threshold: {best_threshold:.4f}")
    print(f"Transactions Correctly Identified as Legitimate (TN): {tn}")
    print(f"Transactions Correctly Identified as FRAUD      (TP): {tp} ✅")
    print(f"Innocent Transactions Blocked            (FP): {fp} ❌")
    print(f"Fraudulent Transactions Missed           (FN): {fn} ⚠️")

    # 8. Explainability & Visualization
    print("\n🔍 FORCE-LOADING FULL DATA FOR VISUALIZATION...")
    
    # Reload the model and full data to avoid the "8 nodes" buffer issue
    model.load_state_dict(torch.load('models/saved_weights/upi_gnn_best.pth'))
    model.eval()

    # Re-build the full graph object fresh for the final plots
    full_data_final, _ = build_graph(df_processed, feature_list)
    full_data_final = full_data_final.to(device)

    print(f"📊 Verified Nodes for Plotting: {full_data_final.num_nodes}")
    
    plot_feature_importance(model, feature_list)
    plot_pr_curve(model, full_data_final)
    
    # This should now show 250,000+ nodes and generate the PNG
    plot_node_embeddings(model, full_data_final)

    print("\n✅ All tasks complete. Check the 'outputs/' folder for your PNG files.")

if __name__ == "__main__":
    main()