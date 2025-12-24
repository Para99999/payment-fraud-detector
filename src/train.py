"""
Training script for the Payment Fraud Detection Foundation Model
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

from model import PaymentFraudDetector, count_parameters
from data_loader import PaymentDataLoader


class Trainer:
    """Trainer for the fraud detection model"""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        config: dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config

        # Loss function (with class weights for imbalanced data)
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (features, labels) in enumerate(pbar):
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(features)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config.get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self) -> dict:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc="Validating"):
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits = self.model(features)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                num_batches += 1

                # Get predictions
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs)

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }

        return metrics

    def train(self, num_epochs: int, save_dir: str):
        """Full training loop"""
        print(f"\nTraining for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {count_parameters(self.model):,}\n")

        best_f1 = 0
        os.makedirs(save_dir, exist_ok=True)

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_metrics = self.validate()
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)

            # Print metrics
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}")

            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                save_path = os.path.join(save_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'metrics': val_metrics
                }, save_path)
                print(f"Saved best model (F1: {best_f1:.4f})")

            # Update learning rate
            self.scheduler.step()

        # Save final model
        final_path = os.path.join(save_dir, 'final_model.pt')
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': val_metrics
        }, final_path)

        print(f"\nTraining completed. Best F1: {best_f1:.4f}")

    def test(self) -> dict:
        """Test the model"""
        print("\nEvaluating on test set...")
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for features, labels in tqdm(self.test_loader, desc="Testing"):
                features = features.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(features)
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        auc = roc_auc_score(all_labels, all_probs)

        print(f"\nTest Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        print(f"\nConfusion Matrix:")
        print(cm)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm
        }

    def plot_training_curves(self, save_path: str = None):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss curves
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)

        # F1 score
        f1_scores = [m['f1'] for m in self.val_metrics]
        axes[1].plot(f1_scores)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Validation F1 Score')
        axes[1].grid(True)

        # AUC
        auc_scores = [m['auc'] for m in self.val_metrics]
        axes[2].plot(auc_scores)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].set_title('Validation AUC')
        axes[2].grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Training curves saved to {save_path}")
        else:
            plt.show()


def main():
    """Main training function"""
    # Get project root
    project_root = Path(__file__).parent.parent

    # Load config
    config_path = project_root / "configs" / "model_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    data_loader = PaymentDataLoader(str(project_root / "data"))

    try:
        df = data_loader.load_kaggle_data()
        print(f"Loaded {len(df)} transactions")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease download the dataset first by running:")
        print("python src/download_data.py")
        sys.exit(1)

    # Preprocess data
    print("\nPreprocessing data...")
    X, y = data_loader.auto_preprocess(df)
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Fraud ratio: {y.sum() / len(y):.4f}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = data_loader.create_dataloaders(
        X, y,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        batch_size=config['pretraining']['batch_size']
    )

    # Create model
    print("\nCreating model...")
    input_dim = data_loader.get_feature_dim(X)
    model = PaymentFraudDetector(
        input_dim=input_dim,
        hidden_size=config['foundation_model']['hidden_size'],
        num_hidden_layers=config['foundation_model']['num_hidden_layers'],
        num_attention_heads=config['foundation_model']['num_attention_heads'],
        intermediate_size=config['foundation_model']['intermediate_size'],
        max_position_embeddings=config['foundation_model']['max_position_embeddings'],
        dropout=config['foundation_model']['dropout']
    ).to(device)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test'],
        device=device,
        config={
            'learning_rate': config['pretraining']['learning_rate'],
            'weight_decay': config['pretraining']['weight_decay'],
            'num_epochs': config['pretraining']['num_epochs'],
            'max_grad_norm': config['pretraining']['max_grad_norm']
        }
    )

    # Train
    model_dir = project_root / "models"
    trainer.train(
        num_epochs=config['pretraining']['num_epochs'],
        save_dir=str(model_dir)
    )

    # Test
    test_metrics = trainer.test()

    # Plot training curves
    trainer.plot_training_curves(
        save_path=str(model_dir / "training_curves.png")
    )

    print("\nTraining completed successfully!")
    print(f"Models saved to: {model_dir}")


if __name__ == "__main__":
    main()
