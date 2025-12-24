"""
FastAPI service for fraud detection inference
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import yaml

from model import PaymentFraudDetector


# API Models
class Transaction(BaseModel):
    """Single transaction data"""
    features: List[float] = Field(..., description="Transaction feature vector")

    class Config:
        json_schema_extra = {
            "example": {
                "features": [0.0] * 30  # Example for credit card dataset
            }
        }


class TransactionBatch(BaseModel):
    """Batch of transactions"""
    transactions: List[List[float]] = Field(..., description="List of transaction feature vectors")


class FraudPrediction(BaseModel):
    """Fraud prediction result"""
    is_fraud: bool
    fraud_probability: float
    confidence: float
    explanation: Optional[Dict[str, Any]] = None


class BatchPrediction(BaseModel):
    """Batch prediction results"""
    predictions: List[FraudPrediction]


class ModelInfo(BaseModel):
    """Model information"""
    model_name: str
    version: str
    input_dim: int
    num_parameters: int
    device: str


# Initialize FastAPI app
app = FastAPI(
    title="Payment Fraud Detection API",
    description="API for detecting fraudulent payment transactions using a transformer-based foundation model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global model instance
model_instance = None
device = None
config = None
scaler = None


def load_model():
    """Load the trained model"""
    global model_instance, device, config, scaler

    project_root = Path(__file__).parent.parent

    # Load config
    config_path = project_root / "configs" / "model_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try to load fine-tuned model first, then best model, then final model
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
        raise FileNotFoundError(
            "No trained model found. Please train the model first using:\n"
            "python src/train.py"
        )

    print(f"Loading model from {checkpoint_path}")

    # Load checkpoint to get input_dim
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Infer input_dim from the model state dict
    input_projection_weight = checkpoint['model_state_dict']['encoder.input_projection.weight']
    input_dim = input_projection_weight.shape[1]

    # Create model
    model_instance = PaymentFraudDetector(
        input_dim=input_dim,
        hidden_size=config['foundation_model']['hidden_size'],
        num_hidden_layers=config['foundation_model']['num_hidden_layers'],
        num_attention_heads=config['foundation_model']['num_attention_heads'],
        intermediate_size=config['foundation_model']['intermediate_size'],
        max_position_embeddings=config['foundation_model']['max_position_embeddings'],
        dropout=0.0  # No dropout for inference
    ).to(device)

    # Load weights
    model_instance.load_state_dict(checkpoint['model_state_dict'])
    model_instance.eval()

    print(f"Model loaded successfully on {device}")
    print(f"Input dimension: {input_dim}")

    # Try to load scaler
    scaler_path = model_dir / "scaler.pkl"
    if scaler_path.exists():
        import pickle
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded")
    else:
        print("Warning: Scaler not found. Input features should be pre-scaled.")


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will start but predictions will fail until model is loaded.")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Payment Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "model_info": "/model/info",
            "predict": "/predict",
            "predict_batch": "/predict/batch"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None
    }


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    num_params = sum(p.numel() for p in model_instance.parameters())

    return ModelInfo(
        model_name="PaymentFraudDetector",
        version="1.0.0",
        input_dim=model_instance.encoder.input_projection.in_features,
        num_parameters=num_params,
        device=str(device)
    )


@app.post("/predict", response_model=FraudPrediction)
async def predict(transaction: Transaction):
    """Predict fraud for a single transaction"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to tensor
        features = np.array(transaction.features).reshape(1, -1)

        # Apply scaling if available
        if scaler is not None:
            features = scaler.transform(features)

        features_tensor = torch.FloatTensor(features).to(device)

        # Predict
        with torch.no_grad():
            fraud_prob = model_instance.predict_proba(features_tensor).item()

        # Determine fraud
        is_fraud = fraud_prob > 0.5
        confidence = fraud_prob if is_fraud else (1 - fraud_prob)

        return FraudPrediction(
            is_fraud=is_fraud,
            fraud_probability=fraud_prob,
            confidence=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPrediction)
async def predict_batch(batch: TransactionBatch):
    """Predict fraud for multiple transactions"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to tensor
        features = np.array(batch.transactions)

        # Apply scaling if available
        if scaler is not None:
            features = scaler.transform(features)

        features_tensor = torch.FloatTensor(features).to(device)

        # Predict
        with torch.no_grad():
            fraud_probs = model_instance.predict_proba(features_tensor).cpu().numpy()

        # Create predictions
        predictions = []
        for prob in fraud_probs:
            is_fraud = prob > 0.5
            confidence = prob if is_fraud else (1 - prob)

            predictions.append(
                FraudPrediction(
                    is_fraud=bool(is_fraud),
                    fraud_probability=float(prob),
                    confidence=float(confidence)
                )
            )

        return BatchPrediction(predictions=predictions)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/embeddings")
async def get_embeddings(transaction: Transaction):
    """Get transaction embeddings"""
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to tensor
        features = np.array(transaction.features).reshape(1, -1)

        # Apply scaling if available
        if scaler is not None:
            features = scaler.transform(features)

        features_tensor = torch.FloatTensor(features).to(device)

        # Get embeddings
        with torch.no_grad():
            embeddings = model_instance.get_embeddings(features_tensor)

        return {
            "embeddings": embeddings.cpu().numpy().tolist()[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting embeddings: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
