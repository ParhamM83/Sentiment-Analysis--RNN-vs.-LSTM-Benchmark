import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=1e-3,
        grad_clip=5.0
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.grad_clip = grad_clip
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_accuracy = 0.0
        self.best_epoch = 0

    def train_epoch(self):
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=True)
        for texts, lengths, labels in progress_bar:
            texts = texts.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(texts, lengths)
            loss = self.criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels).sum().item()
            total_correct += correct
            total_examples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

            running_loss = total_loss / total_examples
            running_acc = total_correct / total_examples
            progress_bar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

        epoch_loss = total_loss / total_examples
        epoch_accuracy = total_correct / total_examples

        return epoch_loss, epoch_accuracy

    def validate_epoch(self):
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation", leave=True)
            for texts, lengths, labels in progress_bar:
                texts = texts.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(texts, lengths)
                loss = self.criterion(logits, labels)

                predictions = torch.argmax(logits, dim=1)
                correct = (predictions == labels).sum().item()
                total_correct += correct
                total_examples += labels.size(0)
                total_loss += loss.item() * labels.size(0)

                running_loss = total_loss / total_examples
                running_acc = total_correct / total_examples
                progress_bar.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.4f}")

        epoch_loss = total_loss / total_examples
        epoch_accuracy = total_correct / total_examples

        return epoch_loss, epoch_accuracy

    def fit(self, epochs):
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} [{self.model.cell_type.upper()}] ---")
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch()

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_epoch = epoch + 1

            print(
                f"[{self.model.cell_type.upper()}] "
                f"Epoch {epoch + 1}/{epochs} Finished | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f}"
            )

        return {
            "train_losses": self.train_losses,
            "train_accuracies": self.train_accuracies,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "best_val_accuracy": self.best_val_accuracy,
            "best_epoch": self.best_epoch,
        }

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
