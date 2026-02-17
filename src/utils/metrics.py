# metrics.py is the most critical file for academic evaluation. Because your fraud rate is only 0.2%,
# standard accuracy will always be 99.8% even if your model catches zero fraud.

import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def evaluate_model(model, data, edge_mask=None):
    """
    Calculates essential metrics and returns the threshold that maximized F1.

    Args:
        model: The GNN model
        data: PyTorch Geometric Data object
        edge_mask: Optional boolean mask to evaluate only on specific edges
    """
    model.eval()
    with torch.no_grad():
        # 1. Get Predictions
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_true = data.y.cpu().numpy()

        # Apply edge mask if provided
        if edge_mask is not None:
            edge_mask = edge_mask.cpu().numpy() if isinstance(edge_mask, torch.Tensor) else edge_mask
            probs = probs[edge_mask]
            y_true = y_true[edge_mask]

        # 2. Dynamic Threshold Search
        best_f1 = 0
        final_precision, final_recall = 0, 0
        best_threshold = 0.1

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

        # 3. Calculate AUPRC
        auprc = average_precision_score(y_true, probs)

        return final_precision, final_recall, best_f1, auprc, best_threshold

def get_detailed_logs(model, data, threshold=0.5, edge_mask=None):
    """
    Generates counts for the Confusion Matrix based on a specific threshold.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_true = data.y.cpu().numpy()

        # Apply edge mask if provided
        if edge_mask is not None:
            edge_mask = edge_mask.cpu().numpy() if isinstance(edge_mask, torch.Tensor) else edge_mask
            probs = probs[edge_mask]
            y_true = y_true[edge_mask]

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

def plot_pr_curve(model, data, edge_mask=None, output_path='outputs/pr_curve.png'):
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        y_true = data.y.cpu().numpy()

        # Apply edge mask if provided
        if edge_mask is not None:
            edge_mask = edge_mask.cpu().numpy() if isinstance(edge_mask, torch.Tensor) else edge_mask
            probs = probs[edge_mask]
            y_true = y_true[edge_mask]

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

from sklearn.manifold import TSNE
import seaborn as sns

def plot_node_embeddings(model, data, output_path='outputs/node_embeddings.png'):
    model.eval()
    
    # FORCE: Use the actual shape of the features, not the .num_nodes property
    actual_node_count = data.x.shape[0]
    print(f"📊 Feature Matrix contains: {actual_node_count} nodes")
    
    with torch.no_grad():
        # Get embeddings from the second GraphSAGE layer
        x = model.conv1(data.x, data.edge_index).relu()
        all_embeddings = model.conv2(x, data.edge_index).cpu().numpy()
        y = data.y.cpu().numpy()

    # Sample 5,000 nodes for a clean visualization
    sample_size = min(5000, actual_node_count)
    
    if sample_size < 30:
        print(f"⚠️ Warning: Still only seeing {sample_size} nodes. Visualization skipped.")
        return

    print(f"🎨 Computing t-SNE for {sample_size} points... (This takes 1-2 minutes)")
    indices = np.random.choice(actual_node_count, sample_size, replace=False)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(all_embeddings[indices])

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=embeddings_2d[:, 0], 
        y=embeddings_2d[:, 1],
        hue=y[indices], 
        palette={0: '#3498db', 1: '#e74c3c'},
        alpha=0.6, 
        s=20,
        edgecolor='w',
        linewidth=0.5
    )
    plt.title('t-SNE Visualization: UPI VPA Latent Space')
    plt.legend(title='Status', labels=['Legitimate', 'Fraudulent'])
    plt.savefig(output_path, dpi=300)
    print(f"✅ FINAL SUCCESS! Embedding map saved to {output_path}")