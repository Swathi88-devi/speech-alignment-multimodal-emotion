"""
Speech Pipeline Training Script
===============================
Usage:
    python train.py --epochs 15 --batch_size 32 --lr 1e-3
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset import get_dataloaders
from models.speech_pipeline.model import SpeechEmotionModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Train Speech Emotion Recognition Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint", type=str, default="best_speech_model.pt", help="Checkpoint filename")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Training Speech Pipeline on device: {device} ===")

    train_loader, val_loader, _, metadata = get_dataloaders(batch_size=args.batch_size)
    num_classes = len(metadata["emotions"])

    model = SpeechEmotionModel(n_mels=128, hidden_dim=64, latent_dim=128, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)

    for epoch in range(1, args.epochs + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            mel = batch["mel"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits, _ = model(mel)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        scheduler.step()
        epoch_train_loss = running_loss / total
        epoch_train_acc = correct / total

        # 2. Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                mel = batch["mel"].to(device)
                labels = batch["label"].to(device)

                logits, _ = model(mel)
                loss = criterion(logits, labels)

                val_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)

        print(f"Epoch [{epoch:2d}/{args.epochs:2d}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc*100:5.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc*100:5.2f}%")

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": best_val_acc,
                "metadata": metadata
            }, checkpoint_path)

    print(f"\nTraining Complete. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"Saved best model checkpoint to: {checkpoint_path}")

    # Plot Loss & Accuracy Curves
    results_plot_dir = os.path.join(PROJECT_ROOT, "Results", "plots")
    os.makedirs(results_plot_dir, exist_ok=True)
    plot_path = os.path.join(results_plot_dir, "speech_learning_curves.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, args.epochs + 1)
    ax1.plot(epochs_range, train_losses, label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs_range, val_losses, label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title("Speech Pipeline Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, [a * 100 for a in train_accs], label="Train Acc", color="#1f77b4", lw=2)
    ax2.plot(epochs_range, [a * 100 for a in val_accs], label="Val Acc", color="#ff7f0e", lw=2, linestyle="--")
    ax2.set_title("Speech Pipeline Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved learning curves plot to: {plot_path}")

if __name__ == "__main__":
    main()
