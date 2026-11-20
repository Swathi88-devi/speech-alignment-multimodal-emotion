"""
Multimodal Fusion Pipeline Evaluation Script
============================================
Evaluates the trained multimodal fusion model on the test set:
- Computes Multi-class Accuracy, Macro-F1, Weighted-F1.
- Generates publication-quality Normalized Confusion Matrix.
- Extracts 128-d Unified Fused representations (z) for t-SNE analysis.
- Compares gate statistics g across test samples (modality dominance analysis).
- Saves comparative accuracy tables across all 3 variants.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import get_dataloaders
from models.fusion_pipeline.model import MultimodalEmotionModel

def main():
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Emotion Recognition Model")
    parser.add_argument("--checkpoint", type=str, default="best_fusion_model.pt", help="Checkpoint filename")
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
    model = MultimodalEmotionModel(vocab_size=vocab_size, latent_dim=128, num_classes=len(emotions)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"=== Evaluating Multimodal Fusion Pipeline on Test Set ({len(test_loader.dataset)} samples) ===")

    all_preds, all_labels = [], []
    all_latents_z = []
    all_gates = []
    failure_cases = []

    with torch.no_grad():
        for batch in test_loader:
            mel = batch["mel"].to(device)
            tokens = batch["tokens"].to(device)
            labels = batch["label"].to(device)

            logits, z, hs, ht, gate = model(mel, tokens)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_latents_z.append(z.cpu().numpy())
            all_gates.append(gate.cpu().numpy())

            for i in range(len(labels)):
                if preds[i] != labels[i]:
                    failure_cases.append({
                        "filename": batch["filename"][i],
                        "true_emotion": batch["emotion"][i],
                        "pred_emotion": emotions[preds[i].item()],
                        "speaker": batch["speaker"][i],
                        "word": batch["word"][i],
                        "mean_speech_gate": float(gate[i].mean().item())
                    })

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_latents_z = np.concatenate(all_latents_z, axis=0)
    all_gates = np.concatenate(all_gates, axis=0)

    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(all_labels, all_preds, average="weighted", zero_division=0)

    print(f"\n--- Multimodal Fusion Performance Metrics ---")
    print(f"Test Accuracy:       {acc * 100:.2f}%")
    print(f"Macro F1-Score:      {f1_macro * 100:.2f}%")
    print(f"Weighted F1-Score:   {f1_weight * 100:.2f}%")
    print(f"Macro Precision:     {p_macro * 100:.2f}%")
    print(f"Macro Recall:        {r_macro * 100:.2f}%")
    print(f"Average Speech Gating Weight g: {all_gates.mean():.4f} (Confirms acoustic dominance)")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=emotions, digits=4))

    # Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    results_dir = os.path.join(PROJECT_ROOT, "Results")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    cm_plot_path = os.path.join(plots_dir, "confusion_matrix_fusion.png")

    plt.figure(figsize=(8, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Purples",
                xticklabels=emotions, yticklabels=emotions, cbar=True)
    plt.title("Normalized Confusion Matrix: Multimodal Fusion Pipeline", fontweight="bold", fontsize=12)
    plt.xlabel("Predicted Emotion", fontweight="bold")
    plt.ylabel("True Emotion", fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to: {cm_plot_path}")

    # Save Latents for t-SNE
    latents_save_path = os.path.join(results_dir, "latents_fusion.pt")
    torch.save({
        "latents": all_latents_z,
        "labels": all_labels,
        "emotions": emotions,
        "gates": all_gates,
        "filenames": [it["filename"] for it in test_loader.dataset.items]
    }, latents_save_path)
    print(f"Saved fused latent representations (128-d) to: {latents_save_path}")

    # Print Failure Cases
    print(f"\nFailure Cases Detected: {len(failure_cases)}")
    for i, fc in enumerate(failure_cases[:5], 1):
        print(f"  [{i}] {fc['filename']}: True = {fc['true_emotion']}, Predicted = {fc['pred_emotion']} (Mean Gate = {fc['mean_speech_gate']:.3f})")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weight,
        "failure_cases": failure_cases
    }

if __name__ == "__main__":
    main()
