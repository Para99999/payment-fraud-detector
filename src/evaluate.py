"""
Model evaluation and visualization tools
"""
import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm

from model import PaymentFraudDetector
from data_loader import PaymentDataLoader


def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    print("Evaluating model...")
    with torch.no_grad():
        for features, labels in tqdm(test_loader):
            features = features.to(device)
            labels = labels.to(device)

            # Get predictions
            logits = model(features)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    return all_preds, all_labels, all_probs


def calculate_metrics(y_true, y_pred, y_probs):
    """Calculate all evaluation metrics"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_probs)
    }

    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Add labels
    plt.xticks([0.5, 1.5], ['Legitimate', 'Fraud'])
    plt.yticks([0.5, 1.5], ['Legitimate', 'Fraud'])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_roc_curve(y_true, y_probs, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    auc = roc_auc_score(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_precision_recall_curve(y_true, y_probs, save_path=None):
    """Plot precision-recall curve"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_probability_distribution(y_true, y_probs, save_path=None):
    """Plot fraud probability distribution"""
    plt.figure(figsize=(10, 6))

    # Separate by class
    legit_probs = y_probs[y_true == 0]
    fraud_probs = y_probs[y_true == 1]

    plt.hist(legit_probs, bins=50, alpha=0.6, label='Legitimate', color='green', density=True)
    plt.hist(fraud_probs, bins=50, alpha=0.6, label='Fraud', color='red', density=True)

    plt.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')

    plt.xlabel('Fraud Probability')
    plt.ylabel('Density')
    plt.title('Distribution of Fraud Probabilities by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability distribution saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_threshold_analysis(y_true, y_probs, save_path=None):
    """Plot metrics vs threshold"""
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []

    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_true, y_pred, zero_division=0))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', linewidth=2)
    plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)

    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Default Threshold')

    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs Classification Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Threshold analysis saved to {save_path}")
    else:
        plt.show()

    plt.close()


def generate_evaluation_report(metrics, y_true, y_pred, save_path=None):
    """Generate comprehensive evaluation report"""
    report = []

    report.append("=" * 60)
    report.append("MODEL EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")

    # Overall metrics
    report.append("OVERALL METRICS:")
    report.append("-" * 60)
    for key, value in metrics.items():
        report.append(f"{key.upper():20s}: {value:.4f}")
    report.append("")

    # Classification report
    report.append("DETAILED CLASSIFICATION REPORT:")
    report.append("-" * 60)
    clf_report = classification_report(y_true, y_pred, target_names=['Legitimate', 'Fraud'])
    report.append(clf_report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    report.append("CONFUSION MATRIX:")
    report.append("-" * 60)
    report.append(f"{'':15s} {'Predicted Legit':20s} {'Predicted Fraud':20s}")
    report.append(f"{'Actual Legit':15s} {cm[0, 0]:20d} {cm[0, 1]:20d}")
    report.append(f"{'Actual Fraud':15s} {cm[1, 0]:20d} {cm[1, 1]:20d}")
    report.append("")

    # Class distribution
    report.append("CLASS DISTRIBUTION:")
    report.append("-" * 60)
    report.append(f"Total Samples: {len(y_true)}")
    report.append(f"Legitimate: {(y_true == 0).sum()} ({(y_true == 0).sum() / len(y_true) * 100:.2f}%)")
    report.append(f"Fraud: {(y_true == 1).sum()} ({(y_true == 1).sum() / len(y_true) * 100:.2f}%)")
    report.append("")

    report.append("=" * 60)

    report_text = "\n".join(report)
    print(report_text)

    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to {save_path}")

    return report_text


def main():
    """Main evaluation function"""
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
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)

    X, y = data_loader.auto_preprocess(df)

    # Create dataloaders
    dataloaders = data_loader.create_dataloaders(
        X, y,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split'],
        batch_size=config['pretraining']['batch_size']
    )

    # Load model
    print("\nLoading model...")
    model_dir = project_root / "models"
    possible_paths = [
        model_dir / "finetuned" / "best_model.pt",
        model_dir / "best_model.pt",
        model_dir / "final_model.pt"
    ]

    checkpoint_path = None
    for path in possible_paths:
        if path.exists():
            checkpoint_path = path
            break

    if checkpoint_path is None:
        print("Error: No trained model found.")
        sys.exit(1)

    print(f"Loading from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get input dim
    input_projection_weight = checkpoint['model_state_dict']['encoder.input_projection.weight']
    input_dim = input_projection_weight.shape[1]

    # Create model
    model = PaymentFraudDetector(
        input_dim=input_dim,
        hidden_size=config['foundation_model']['hidden_size'],
        num_hidden_layers=config['foundation_model']['num_hidden_layers'],
        num_attention_heads=config['foundation_model']['num_attention_heads'],
        intermediate_size=config['foundation_model']['intermediate_size'],
        dropout=0.0
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate
    y_pred, y_true, y_probs = evaluate_model(model, dataloaders['test'], device)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_probs)

    # Create evaluation directory
    eval_dir = project_root / "evaluation"
    eval_dir.mkdir(exist_ok=True)

    # Generate report
    generate_evaluation_report(
        metrics, y_true, y_pred,
        save_path=str(eval_dir / "evaluation_report.txt")
    )

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(y_true, y_pred, save_path=str(eval_dir / "confusion_matrix.png"))
    plot_roc_curve(y_true, y_probs, save_path=str(eval_dir / "roc_curve.png"))
    plot_precision_recall_curve(y_true, y_probs, save_path=str(eval_dir / "precision_recall_curve.png"))
    plot_probability_distribution(y_true, y_probs, save_path=str(eval_dir / "probability_distribution.png"))
    plot_threshold_analysis(y_true, y_probs, save_path=str(eval_dir / "threshold_analysis.png"))

    print(f"\nEvaluation complete! Results saved to: {eval_dir}")


if __name__ == "__main__":
    main()
