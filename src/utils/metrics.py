# metrics.py is the most critical file for academic evaluation. Because your fraud rate is only 0.2%, 
# standard accuracy will always be 99.8% even if your model catches zero fraud.

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os

def evaluate_model(model, data):
    """
    Calculates essential metrics and returns the threshold that maximized F1.
    """
    model.eval()
    with torch.no_grad():
        # 1. Get Predictions
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_true = data.y.cpu().numpy()

        # 2. Dynamic Threshold Search
        best_f1 = 0
        final_precision, final_recall = 0, 0
        best_threshold = 0.1 # Default starting point
        
        # Test 50 different thresholds to find the "sweet spot"
        for threshold in np.linspace(0.01, 0.8, 50):
            preds = (probs > threshold).astype(int)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, preds, average='binary', zero_division=0
            )
            
            if f1 > best_f1:
                best_f1 = f1
                final_precision = precision
                final_recall = recall
                best_threshold = threshold

        # 3. Calculate AUPRC (Area Under Precision-Recall Curve)
        auprc = average_precision_score(y_true, probs)

        print(f"DEBUG: Best Threshold: {best_threshold:.3f}")
        
        # 🔥 CRITICAL: Return exactly 5 values to match main.py
        return final_precision, final_recall, best_f1, auprc, best_threshold

def get_detailed_logs(model, data, threshold=0.5):
    """
    Generates counts for the Confusion Matrix based on a specific threshold.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_true = data.y.cpu().numpy()
        
        preds = (probs > threshold).astype(int)
        # Handle cases where the model might predict only one class
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        except ValueError:
            # Fallback if confusion_matrix is not 2x2
            tp = np.sum((preds == 1) & (y_true == 1))
            tn = np.sum((preds == 0) & (y_true == 0))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))
            
        return tn, fp, fn, tp
    
def plot_feature_importance(model, feature_names, output_path='outputs/feature_importance.png'):
    model.eval()
    
    # --- ADD THIS LINE ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # ---------------------

    weights = model.classifier[0].weight.abs().mean(dim=0).detach().cpu().numpy()
    raw_importance = weights[-len(feature_names):]
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': raw_importance
    }).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='#3498db')
    plt.title('GNN Feature Importance: Which signals catch UPI Fraud?')
    plt.xlabel('Normalized Importance (Weights)')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"📊 Feature Importance chart saved to {output_path}")

from sklearn.metrics import precision_recall_curve, auc

def plot_pr_curve(model, data, output_path='outputs/pr_curve.png'):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_true = data.y.cpu().numpy()
    
    precision, recall, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'GNN (AUC = {pr_auc:.4f})')
    plt.axhline(y=sum(y_true)/len(y_true), color='r', linestyle='--', label='Random Baseline')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print(f"📈 PR Curve saved to {output_path}")