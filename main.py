import argparse
import random
from pathlib import Path
import numpy as np
import torch

from data import create_dataloaders
from model import SentimentClassifier
from trainer import Trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_comparison(rnn_hist, lstm_hist, save_path="rnn_vs_lstm_comparison.png"):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = range(1, len(rnn_hist["train_losses"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss comparison
    ax1.plot(epochs, rnn_hist["train_losses"], "r--", label="RNN Train Loss")
    ax1.plot(epochs, rnn_hist["val_losses"], "r-", label="RNN Val Loss")
    ax1.plot(epochs, lstm_hist["train_losses"], "b--", label="LSTM Train Loss")
    ax1.plot(epochs, lstm_hist["val_losses"], "b-", label="LSTM Val Loss")
    ax1.set_title("Loss Comparison: RNN vs LSTM")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy comparison
    ax2.plot(epochs, rnn_hist["train_accuracies"], "r--", label="RNN Train Acc")
    ax2.plot(epochs, rnn_hist["val_accuracies"], "r-", label="RNN Val Acc")
    ax2.plot(epochs, lstm_hist["train_accuracies"], "b--", label="LSTM Train Acc")
    ax2.plot(epochs, lstm_hist["val_accuracies"], "b-", label="LSTM Val Acc")
    ax2.set_title("Accuracy Comparison: RNN vs LSTM")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nComparison plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare RNN vs LSTM for IMDB Sentiment Analysis")
    parser.add_argument("--data_dir", type=str, default="./aclImdb", help="Path to aclImdb directory")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and evaluation")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for RNN/LSTM")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of recurrent layers")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--plot", action="store_true", default=True, help="Save comparison plot")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_dir = Path(args.data_dir)
    print(f"\n[1/3] Preparing dataset from {data_dir.resolve()}...")
    train_loader, test_loader, vocab = create_dataloaders(data_dir, batch_size=args.batch_size)
    vocab_size = len(vocab)
    print(f"Vocabulary successfully built: {vocab_size:,} unique words")

    print("\n" + "=" * 30 + " TRAINING RNN " + "=" * 30)
    set_seed(args.seed)
    rnn_model = SentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        cell_type="rnn",
        bidirectional=True
    ).to(device)
    print(f"RNN Trainable Parameters: {rnn_model.count_parameters():,}")

    rnn_trainer = Trainer(
        model=rnn_model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        learning_rate=args.lr
    )
    rnn_history = rnn_trainer.fit(epochs=args.epochs)

    print("\n" + "=" * 30 + " TRAINING LSTM " + "=" * 30)
    set_seed(args.seed)
    lstm_model = SentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        cell_type="lstm",
        bidirectional=True
    ).to(device)
    print(f"LSTM Trainable Parameters: {lstm_model.count_parameters():,}")

    lstm_trainer = Trainer(
        model=lstm_model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        learning_rate=args.lr
    )
    lstm_history = lstm_trainer.fit(epochs=args.epochs)

    print("\n======= FINAL RESULTS =======")
    print("                 -RNN-          -LSTM-")
    print(f"Parameters: {rnn_model.count_parameters():>10,} | {lstm_model.count_parameters():>10,}")
    print(f"Train Loss: {rnn_trainer.train_losses[-1]:>10.4f} | {lstm_trainer.train_losses[-1]:>10.4f}")
    print(f"Train Acc : {rnn_trainer.train_accuracies[-1]:>10.4f} | {lstm_trainer.train_accuracies[-1]:>10.4f}")
    print(f"Val Loss  : {rnn_trainer.val_losses[-1]:>10.4f} | {lstm_trainer.val_losses[-1]:>10.4f}")
    print(f"Val Acc   : {rnn_trainer.val_accuracies[-1]:>10.4f} | {lstm_trainer.val_accuracies[-1]:>10.4f}")
    print(f"Best Val  : {rnn_trainer.best_val_accuracy:>10.4f} | {lstm_trainer.best_val_accuracy:>10.4f}")

    if args.plot:
        plot_comparison(rnn_history, lstm_history)


if __name__ == "__main__":
    main()