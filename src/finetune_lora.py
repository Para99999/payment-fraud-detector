"""
Fine-tuning script with selective layer freezing
(Simplified approach - freeze encoder, train classifier)
"""
import os
import sys
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from model import PaymentFraudDetector, count_parameters
from data_loader import PaymentDataLoader
from train import Trainer


def apply_selective_freezing(model: nn.Module, config: dict) -> nn.Module:
    """
    Apply selective freezing for efficient fine-tuning

    Freezes the encoder (transformer layers) and only trains:
    - The classification head
    - Optionally the last transformer block

    This is a simplified alternative to LoRA that works with any model
    """
    print("\nApplying selective layer freezing...")

    # Freeze all encoder parameters
    for name, param in model.encoder.named_parameters():
        param.requires_grad = False

    # Unfreeze the last transformer block for fine-tuning
    if hasattr(model.encoder, 'transformer_blocks') and len(model.encoder.transformer_blocks) > 0:
        for param in model.encoder.transformer_blocks[-1].parameters():
            param.requires_grad = True
        print("Unfroze last transformer block for fine-tuning")

    # Keep classification head trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    # Print trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    return model


def load_pretrained_model(model: nn.Module, checkpoint_path: str, device: torch.device) -> nn.Module:
    """Load pre-trained model weights"""
    print(f"\nLoading pre-trained model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if 'metrics' in checkpoint:
        print(f"Pre-trained model metrics:")
        for key, value in checkpoint['metrics'].items():
            if key != 'confusion_matrix':
                print(f"  {key}: {value:.4f}")

    return model


def main():
    """Main fine-tuning function"""
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
    print(f"Fraud ratio: {y.sum() / len(y):.4f}")

    # Create dataloaders
    print("\nCreating dataloaders...")
    dataloaders = data_loader.create_dataloaders(
        X, y,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        batch_size=config['finetuning']['batch_size']
    )

    # Create base model
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

    # Try to load pre-trained weights
    model_dir = project_root / "models"
    pretrained_path = model_dir / "best_model.pt"

    if pretrained_path.exists():
        model = load_pretrained_model(model, str(pretrained_path), device)
    else:
        print("\nWarning: No pre-trained model found.")
        print("Training from scratch (not using LoRA).")
        print("To use LoRA, first train the base model with: python src/train.py")

    # Apply selective freezing if using fine-tuning method
    if config['finetuning']['method'] == 'lora' and pretrained_path.exists():
        model = apply_selective_freezing(model, config)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        test_loader=dataloaders['test'],
        device=device,
        config={
            'learning_rate': config['finetuning']['learning_rate'],
            'weight_decay': config['pretraining']['weight_decay'],
            'num_epochs': config['finetuning']['num_epochs'],
            'max_grad_norm': config['pretraining']['max_grad_norm']
        }
    )

    # Fine-tune
    finetuned_dir = model_dir / "finetuned"
    trainer.train(
        num_epochs=config['finetuning']['num_epochs'],
        save_dir=str(finetuned_dir)
    )

    # Test
    test_metrics = trainer.test()

    # Plot training curves
    trainer.plot_training_curves(
        save_path=str(finetuned_dir / "finetuning_curves.png")
    )

    print("\nFine-tuning completed successfully!")
    print(f"Fine-tuned models saved to: {finetuned_dir}")


if __name__ == "__main__":
    main()
