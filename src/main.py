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

# Import your custom modules
from features import preprocess_upi_data
from data_loader import build_graph
from models.model import UPIGraphSAGE
from utils.trainer import train
from utils.metrics import evaluate_model, get_detailed_logs 
from utils.metrics import plot_feature_importance, plot_pr_curve

def main():
    # 1. Load Hyperparameters from config
    config_path = 'src/models/config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set Device (Mac M1/M2 can use 'mps', otherwise 'cpu' or 'cuda')
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"🚀 Starting UPI Fraud Detection Pipeline on {device}...")

    # 2. Data Acquisition
    raw_data_path = 'data/raw/upi_transactions.csv'
    if not os.path.exists(raw_data_path):
        print(f"❌ Error: {raw_data_path} not found.")
        return

    df_raw = pd.read_csv(raw_data_path)
    
    # 3. Feature Engineering
    df_processed, scaler, feature_list = preprocess_upi_data(df_raw)
    num_features = len(feature_list)

    # 4. Graph Construction
    # Make sure you pass feature_list to build_graph
    print("🕸️  Building transaction graph...")
    data, vpa_map = build_graph(df_processed, feature_list)
    data = data.to(device)

    # 5. Initialize Model
    # Update model with dynamic feature count
    model = UPIGraphSAGE(
        in_channels=num_features, 
        hidden_channels=config.get('hidden_channels', 128), 
        out_channels=2
    ).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # This reduces LR when F1 stops improving, helping the model converge
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Handle Imbalance: Weight the Fraud class (1) much higher than Legitimate (0)
    fraud_weight = config.get('fraud_weight', 500.0)
    weights = torch.tensor([1.0, float(fraud_weight)]).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

   # 6. Training Loop
    print(f"🏋️  Training for {config['epochs']} epochs...")
    best_f1 = 0
    best_threshold = 0.5  # Initialize a default
    
    for epoch in range(1, config['epochs'] + 1):
        loss = train(model, data, optimizer, criterion)
        
        if epoch % 10 == 0 or epoch == 1:
            # Modify evaluate_model to return the threshold it found
            precision, recall, f1, auprc, epoch_threshold = evaluate_model(model, data)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | F1: {f1:.4f} | Recall: {recall:.4f} | LR: {current_lr:.6f}")
            
            # Update the scheduler based on the latest F1 score
            scheduler.step(f1)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = epoch_threshold # Store the threshold that gave the best F1
                torch.save(model.state_dict(), 'models/saved_weights/upi_gnn_best.pth')

    # 7. Final Detailed Report
    print("\n📊 --- FINAL PERFORMANCE REPORT ---")
    model.load_state_dict(torch.load('models/saved_weights/upi_gnn_best.pth'))
    
    # Pass the best_threshold we found during training
    tn, fp, fn, tp = get_detailed_logs(model, data, threshold=best_threshold)
    
    # ... (rest of your print statements)
    print(f"Using Optimal Threshold: {best_threshold:.4f}")
    print(f"Transactions Correctly Identified as Legitimate (TN): {tn}")
    
    # Calculate Final Stats
    final_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    final_rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"Transactions Correctly Identified as Legitimate (TN): {tn}")
    print(f"Transactions Correctly Identified as FRAUD      (TP): {tp} ✅")
    print(f"Innocent Transactions Blocked            (FP): {fp} ❌")
    print(f"Fraudulent Transactions Missed           (FN): {fn} ⚠️")
    print(f"\nFinal Calculated Precision: {final_prec:.4f}")
    print(f"Final Calculated Recall:    {final_rec:.4f}")
    print(f"Best Training F1-Score:     {best_f1:.4f}")

    print("\n✅ Training Complete. Model saved to models/saved_weights/upi_gnn_best.pth")


    # 8. Explainability & Visualization
    # Ensure these are imported at the top of main.py or here 
    print("\n🔍 Generating Analysis Visualizations...")
    
    # Generate the bar chart you just saw
    plot_feature_importance(model, feature_list)
    
    # Generate the PR Curve (Crucial for your Thesis results)
    plot_pr_curve(model, data)

    print("\n✅ All tasks complete. Check the 'outputs/' folder for your PNG files.")

if __name__ == "__main__":
    main()