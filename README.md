# Payment Fraud Detector

Transformer-based fraud detection system using PyTorch with LoRA fine-tuning for efficient parameter adaptation.

## What it Does

- Detects fraudulent payment transactions using a custom transformer encoder architecture
- Provides foundation model pre-training for learning transaction patterns and representations
- Enables efficient fine-tuning via LoRA (Low-Rank Adaptation) with minimal parameter updates
- Exposes REST API endpoints for real-time fraud prediction and batch processing

## Architecture

The system implements a three-tier architecture:

### Model Layer
- **PaymentTransactionEncoder**: Transformer-based foundation model with multi-head self-attention
- **FraudDetectionHead**: Binary classification head for fraud probability scoring
- **LoRA Adapter**: Parameter-efficient fine-tuning using low-rank matrix decomposition

Input features flow through:
1. Input projection (linear transformation to hidden dimension)
2. Positional encoding (adds sequential information)
3. Transformer blocks (6 layers, 8 attention heads)
4. Classification head (binary fraud detection)

### API Layer
- FastAPI service for synchronous inference
- Endpoints: single prediction, batch prediction, embeddings, health checks
- Automatic model loading with checkpoint fallback

### UI Layer
- Streamlit dashboard for interactive fraud detection
- Single transaction analysis
- Batch transaction processing
- Model performance visualization

## Key Components

- `src/model.py`: Transformer architecture implementation
- `src/train.py`: Foundation model pre-training script
- `src/finetune_lora.py`: LoRA fine-tuning implementation
- `src/data_loader.py`: Dataset loading and preprocessing
- `src/evaluate.py`: Model evaluation and metrics
- `api/app.py`: FastAPI inference service
- `ui/streamlit_app.py`: Streamlit UI application
- `configs/model_config.yaml`: Model hyperparameters

## Tech Stack

- PyTorch 2.0+
- Transformers (Hugging Face)
- PEFT (Parameter-Efficient Fine-Tuning)
- FastAPI
- Streamlit
- scikit-learn
- NumPy, Pandas
- Matplotlib, Seaborn, Plotly

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, recommended for training)
- 8GB+ RAM
- Kaggle API credentials (for dataset download)

## Quickstart

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Configure Kaggle API credentials:
- Create account at https://www.kaggle.com/account
- Generate API token and place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows)

Download the dataset:

```bash
python src/download_data.py
```

Alternatively, manually download from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place `creditcard.csv` in the `data/` directory.

### 3. Train the Model

Pre-train the foundation model:

```bash
python src/train.py
```

Training outputs:
- `models/best_model.pt` (best validation performance)
- `models/final_model.pt` (final epoch checkpoint)
- `models/training_curves.png` (loss/accuracy plots)

### 4. Fine-tune with LoRA (Optional)

Apply LoRA fine-tuning for improved performance:

```bash
python src/finetune_lora.py
```

Fine-tuning outputs:
- `models/finetuned/best_model.pt`
- `models/finetuned/final_model.pt`
- `models/finetuned/finetuning_curves.png`

### 5. Evaluate Model

Generate evaluation metrics:

```bash
python src/evaluate.py
```

Produces confusion matrix, ROC curve, precision-recall curve, and performance metrics.

### 6. Run API Service

Start the FastAPI server:

```bash
python api/app.py
```

Or using uvicorn directly:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

API documentation available at: http://localhost:8000/docs

Example prediction request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": [0.0, 0.1, 0.2, -0.5, 1.2, ...]}'
```

### 7. Launch UI Dashboard

Start the Streamlit interface:

```bash
streamlit run ui/streamlit_app.py
```

Access at: http://localhost:8501

## Configuration

Edit `configs/model_config.yaml` to customize model architecture and training:

```yaml
foundation_model:
  hidden_size: 256          # Transformer hidden dimension
  num_attention_heads: 8    # Multi-head attention heads
  num_hidden_layers: 6      # Number of transformer blocks
  intermediate_size: 1024   # Feed-forward network size
  dropout: 0.1              # Dropout probability

pretraining:
  batch_size: 64
  learning_rate: 0.0001
  num_epochs: 10

finetuning:
  method: "lora"            # lora or full
  lora_r: 8                 # LoRA rank
  lora_alpha: 32            # LoRA scaling factor
  batch_size: 32
  learning_rate: 0.0002
  num_epochs: 5
```

### Environment Variables

Create `.env` file (optional):

```
MODEL_PATH=models/best_model.pt
DEVICE=cuda
LOG_LEVEL=INFO
```

## How to Run Tests

Currently, the project does not include automated tests. To verify functionality:

1. Run training script: `python src/train.py`
2. Check model outputs in `models/` directory
3. Test API endpoints: `curl http://localhost:8000/health`
4. Validate predictions via UI or API

## Project Structure

```
payment-fraud-detector/
├── api/
│   └── app.py                  # FastAPI inference service
├── archive/                    # Archived documentation
├── configs/
│   └── model_config.yaml       # Model configuration
├── data/
│   └── creditcard.csv          # Dataset (downloaded)
├── models/
│   ├── best_model.pt           # Best pre-trained model
│   ├── final_model.pt          # Final pre-trained model
│   ├── training_curves.png     # Training visualizations
│   └── finetuned/              # LoRA fine-tuned models
├── notebooks/                  # Jupyter notebooks (optional)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── model.py                # Transformer architecture
│   ├── train.py                # Pre-training script
│   ├── finetune_lora.py        # LoRA fine-tuning
│   ├── evaluate.py             # Model evaluation
│   └── download_data.py        # Kaggle dataset downloader
├── ui/
│   └── streamlit_app.py        # Streamlit dashboard
├── .env.example                # Environment template
├── .gitignore
├── requirements.txt            # Python dependencies
└── README.md
```

## Troubleshooting

### Model Not Found Error

**Problem**: `FileNotFoundError: No trained model found`

**Solution**: Train the model first:
```bash
python src/train.py
```

### Dataset Missing

**Problem**: `FileNotFoundError: No fraud dataset found in data/`

**Solution**: Download dataset using:
```bash
python src/download_data.py
```
Or manually download from Kaggle and place in `data/` directory.

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce batch size in `configs/model_config.yaml`:
```yaml
pretraining:
  batch_size: 32  # Reduce from 64
```

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'torch'`

**Solution**: Install all dependencies:
```bash
pip install -r requirements.txt
```

### API Connection Refused

**Problem**: Cannot connect to API at localhost:8000

**Solution**: Ensure API service is running:
```bash
python api/app.py
```

### Poor Model Performance

**Solutions**:
- Increase training epochs in config
- Adjust learning rate (try 1e-4 to 1e-5)
- Increase model capacity (hidden_size, num_layers)
- Apply class weights for imbalanced datasets
- Verify data preprocessing and normalization

## Roadmap / Future Improvements

- Implement model explainability using SHAP values for interpretable predictions
- Add automated testing suite with unit and integration tests
- Support for streaming inference via Kafka or Kinesis integration
- Multi-task learning for fraud detection and transaction categorization
- Model quantization for faster CPU inference
- Docker containerization for simplified deployment
- Continuous model monitoring and drift detection
- Automated retraining pipeline with MLflow tracking
- Support for additional fraud datasets and domains
- Graph neural network extension for transaction network analysis

## License

TBD
