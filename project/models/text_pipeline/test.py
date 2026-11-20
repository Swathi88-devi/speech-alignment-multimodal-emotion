"""
Text Pipeline Evaluation Script
===============================
Evaluates the trained text model on the test set:
- Computes Multi-class Accuracy, Macro-F1, Weighted-F1.
- Generates publication-quality Normalized Confusion Matrix.
- Extracts 128-d Contextual Modelling representations (ht) for t-SNE analysis.
- Demonstrates the Lexical Carrier Dilemma on TESS.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import get_dataloaders
from models.text_pipeline.model import TextEmotionModel

def main():
    parser = argparse.ArgumentParser(description="Evaluate Text Emotion Recognition Model")
    parser.add_argument("--checkpoint", type=str, default="best_text_model.pt", help="Checkpoint filename")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}. Train the model first!")

    _, _, test_loader, metadata = get_dataloaders(batch_size=args.batch_size)
    emotions = metadata["emotions"]
    vocab_size = len(metadata["vocab"])

    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TextEmotionModel(vocab_size=vocab_size, embed_dim=64, latent_dim=128, num_classes=len(emotions)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"=== Evaluating Text Pipeline on Test Set ({len(test_loader.dataset)} samples) ===")

    all_preds, all_labels = [], []
    all_latents = []

    with torch.no_grad():
        for batch in test_loader:
            tokens = batch["tokens"].to(device)
            labels = batch["label"].to(device)

            logits, ht = model(tokens)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_latents.append(ht.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_latents = np.concatenate(all_latents, axis=0)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\n--- Text-Only Performance Metrics ---")
    print(f"Test Accuracy:       {acc * 100:.2f}%")
    print(f"Macro F1-Score:      {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score:   {f1_weight * 100:.2f}%")
    print(f"Macro Precision:     {p_macro * 100:.2f}%")
    print(f"Macro Recall:        {r_macro * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=emotions, digits=4, zero_division=0))

    # Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    results_plot_dir = os.path.join(PROJECT_ROOT, "Results", "plots")
    os.makedirs(results_plot_dir, exist_ok=True)
    cm_plot_path = os.path.join(results_plot_dir, "confusion_matrix_text.png")

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=emotions, yticklabels=emotions, cbar=True)
    plt.title("Normalized Confusion Matrix: Text-Only Pipeline", fontweight="bold", fontsize=12)
    plt.xlabel("Predicted Emotion", fontweight="bold")
    plt.ylabel("True Emotion", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to: {cm_plot_path}")

    # Save Latents for t-SNE
    latents_save_path = os.path.join(PROJECT_ROOT, "Results", "latents_text.pt")
    torch.save({
        "latents": all_latents,
        "labels": all_labels,
        "emotions": emotions,
        "filenames": [it["filename"] for it in test_loader.dataset.items]
    }, latents_save_path)
    print(f"Saved contextual text representations (128-d) to: {latents_save_path}")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weight
    }

if __name__ == "__main__":
    main()
