"""
Streamlit UI for Payment Fraud Detection
Includes dashboard and chat interface
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import torch
import yaml
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict
import json

from model import PaymentFraudDetector
from data_loader import PaymentDataLoader


# Page config
st.set_page_config(
    page_title="Payment Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #635BFF;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #635BFF;
    }
    .fraud-alert {
        background-color: #fee;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #e74c3c;
    }
    .safe-alert {
        background-color: #efe;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2ecc71;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_config():
    """Load model and configuration"""
    project_root = Path(__file__).parent.parent

    # Load config
    config_path = project_root / "configs" / "model_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try to load model
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
        return None, None, config, device

    # Load checkpoint
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
    model.eval()

    return model, checkpoint.get('metrics', {}), config, device


@st.cache_data
def load_sample_data():
    """Load sample transaction data"""
    project_root = Path(__file__).parent.parent
    data_loader = PaymentDataLoader(str(project_root / "data"))

    try:
        df = data_loader.load_kaggle_data()
        X, y = data_loader.auto_preprocess(df)
        return X, y, df
    except:
        return None, None, None


def predict_fraud(model, features, device):
    """Make fraud prediction"""
    features_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)

    with torch.no_grad():
        fraud_prob = model.predict_proba(features_tensor).item()
        embeddings = model.get_embeddings(features_tensor).cpu().numpy()[0]

    return fraud_prob, embeddings


def create_gauge_chart(probability: float):
    """Create a gauge chart for fraud probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fraud Probability (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred" if probability > 0.5 else "darkgreen"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def chat_interface():
    """Chat interface for querying transactions"""
    st.markdown("### Chat with Fraud Detection Assistant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your fraud detection assistant. Ask me about transactions, fraud patterns, or how the model works."}
        ]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about fraud detection..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response = generate_response(prompt)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


def generate_response(prompt: str) -> str:
    """Generate response to user query"""
    prompt_lower = prompt.lower()

    # Simple rule-based responses (in production, use LLM)
    if "how" in prompt_lower and "work" in prompt_lower:
        return """The fraud detection system uses a **transformer-based foundation model** trained on payment transaction data. Here's how it works:

1. **Input Processing**: Transaction features are processed and normalized
2. **Transformer Encoding**: The model learns rich representations of transaction patterns
3. **Classification**: A fraud detection head predicts the probability of fraud
4. **Output**: You get a fraud probability score and explanation

The model was trained on millions of transactions and can detect subtle patterns that indicate fraud."""

    elif "feature" in prompt_lower or "input" in prompt_lower:
        return """The model analyzes various transaction features including:

- **Transaction Amount**: Unusual amounts can indicate fraud
- **Time Patterns**: Transactions at odd hours or in quick succession
- **Balance Changes**: Sudden changes in account balances
- **Transaction Type**: Different types have different risk profiles
- **Historical Patterns**: Deviation from normal user behavior

All features are processed through the transformer to capture complex relationships."""

    elif "accurate" in prompt_lower or "performance" in prompt_lower:
        return """Model Performance Metrics:

- **AUC-ROC**: Measures ability to distinguish fraud from legitimate transactions
- **Precision**: Of flagged frauds, how many are actual frauds
- **Recall**: Of all actual frauds, how many are caught
- **F1 Score**: Balance between precision and recall

The model is continuously evaluated on held-out test data to ensure reliability."""

    elif "fraud" in prompt_lower and ("prevent" in prompt_lower or "stop" in prompt_lower):
        return """To prevent fraud, the system:

1. **Real-time Monitoring**: Analyzes every transaction as it happens
2. **Risk Scoring**: Assigns a fraud probability to each transaction
3. **Automatic Flagging**: High-risk transactions are flagged for review
4. **Pattern Detection**: Identifies unusual patterns that humans might miss
5. **Continuous Learning**: Model is updated with new fraud patterns

This multi-layered approach provides robust fraud protection."""

    elif "lora" in prompt_lower or "fine-tune" in prompt_lower or "finetune" in prompt_lower:
        return """The model uses **LoRA (Low-Rank Adaptation)** for efficient fine-tuning:

- **Parameter Efficiency**: Only trains ~1% of parameters
- **Faster Training**: Reduces training time significantly
- **Memory Efficient**: Requires less GPU memory
- **Preserves Knowledge**: Keeps pre-trained knowledge while adapting to new data

This allows us to quickly adapt the model to new fraud patterns without retraining from scratch."""

    elif "threshold" in prompt_lower:
        return """The fraud detection threshold can be adjusted based on your needs:

- **Lower Threshold (e.g., 0.3)**: Catch more frauds, but more false positives
- **Default (0.5)**: Balanced approach
- **Higher Threshold (e.g., 0.7)**: Fewer false positives, but might miss some frauds

The optimal threshold depends on the cost of false positives vs. false negatives in your use case."""

    else:
        return """I can help you with:

- How the fraud detection model works
- What features are analyzed
- Model performance and accuracy
- Fraud prevention strategies
- LoRA fine-tuning approach
- Threshold configuration

Ask me anything about fraud detection!"""


def main():
    """Main Streamlit app"""
    # Header
    st.markdown('<h1 class="main-header">💳 Payment Fraud Detection System</h1>', unsafe_allow_html=True)

    # Load model
    with st.spinner("Loading model..."):
        model, metrics, config, device = load_model_and_config()

    # Sidebar
    with st.sidebar:
        st.image("https://stripe.com/img/v3/home/social.png", width=200)
        st.markdown("### Model Information")

        if model is not None:
            st.success("✅ Model Loaded")

            if metrics:
                st.markdown("#### Performance Metrics")
                for key, value in metrics.items():
                    if key != 'confusion_matrix' and isinstance(value, (int, float)):
                        st.metric(key.upper(), f"{value:.4f}")

            st.markdown(f"**Device**: {device}")
            num_params = sum(p.numel() for p in model.parameters())
            st.markdown(f"**Parameters**: {num_params:,}")
        else:
            st.error("❌ Model Not Found")
            st.info("Train the model first:\n```\npython src/train.py\n```")

        st.markdown("---")
        st.markdown("### Navigation")
        page = st.radio("", ["🏠 Dashboard", "🔍 Single Transaction", "📊 Batch Analysis", "💬 Chat Assistant"])

    # Main content
    if model is None:
        st.error("Please train the model first by running: `python src/train.py`")
        return

    if page == "🏠 Dashboard":
        show_dashboard(model, device)
    elif page == "🔍 Single Transaction":
        show_single_prediction(model, device)
    elif page == "📊 Batch Analysis":
        show_batch_analysis(model, device)
    elif page == "💬 Chat Assistant":
        chat_interface()


def show_dashboard(model, device):
    """Show dashboard with overview"""
    st.markdown("## Dashboard Overview")

    # Load sample data
    X, y, df = load_sample_data()

    if X is None:
        st.warning("No dataset found. Please download data first.")
        return

    # Display dataset stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Transactions", f"{len(X):,}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        fraud_count = int(y.sum())
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Fraudulent", f"{fraud_count:,}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        fraud_rate = (y.sum() / len(y)) * 100
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Features", X.shape[1])
        st.markdown('</div>', unsafe_allow_html=True)

    # Sample predictions
    st.markdown("### Sample Predictions")

    # Get random samples
    sample_indices = np.random.choice(len(X), min(100, len(X)), replace=False)
    X_sample = X[sample_indices]
    y_sample = y[sample_indices]

    # Predict
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_sample).to(device)
        fraud_probs = model.predict_proba(X_tensor).cpu().numpy()

    # Create dataframe
    results_df = pd.DataFrame({
        'Transaction ID': range(len(sample_indices)),
        'Actual Label': ['Fraud' if label == 1 else 'Legitimate' for label in y_sample],
        'Predicted Probability': fraud_probs,
        'Prediction': ['Fraud' if prob > 0.5 else 'Legitimate' for prob in fraud_probs]
    })

    # Display results
    st.dataframe(results_df, use_container_width=True)

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Fraud Probability Distribution")
        fig = px.histogram(results_df, x='Predicted Probability', nbins=50,
                          color='Actual Label',
                          title="Distribution of Fraud Probabilities")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Model Predictions")
        confusion_data = pd.crosstab(results_df['Actual Label'], results_df['Prediction'])
        fig = px.imshow(confusion_data, text_auto=True,
                       title="Confusion Matrix (Sample)",
                       labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig, use_container_width=True)


def show_single_prediction(model, device):
    """Show single transaction prediction interface"""
    st.markdown("## Single Transaction Analysis")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Input Transaction Features")

        # Get input dimension
        input_dim = model.encoder.input_projection.in_features

        # Option to use sample data
        use_sample = st.checkbox("Load Sample Transaction")

        if use_sample:
            X, y, df = load_sample_data()
            if X is not None:
                sample_idx = st.slider("Sample Index", 0, len(X) - 1, 0)
                features = X[sample_idx]
                actual_label = y[sample_idx] if y is not None else None
            else:
                features = np.random.randn(input_dim)
                actual_label = None
        else:
            # Manual input
            st.info(f"Enter {input_dim} comma-separated values")
            feature_input = st.text_area(
                "Features",
                value=", ".join(["0.0"] * input_dim),
                height=200
            )

            try:
                features = np.array([float(x.strip()) for x in feature_input.split(",")])
                if len(features) != input_dim:
                    st.error(f"Expected {input_dim} features, got {len(features)}")
                    return
                actual_label = None
            except:
                st.error("Invalid input format. Please enter comma-separated numbers.")
                return

        if st.button("🔍 Analyze Transaction", type="primary"):
            fraud_prob, embeddings = predict_fraud(model, features, device)

            # Display results in col2
            with col2:
                st.markdown("### Analysis Results")

                # Gauge chart
                fig = create_gauge_chart(fraud_prob)
                st.plotly_chart(fig, use_container_width=True)

                # Alert box
                if fraud_prob > 0.5:
                    st.markdown(
                        f'<div class="fraud-alert"><strong>⚠️ FRAUD ALERT</strong><br>'
                        f'This transaction has a high probability ({fraud_prob*100:.1f}%) of being fraudulent.</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="safe-alert"><strong>✅ LEGITIMATE</strong><br>'
                        f'This transaction appears to be legitimate (Fraud probability: {fraud_prob*100:.1f}%).</div>',
                        unsafe_allow_html=True
                    )

                # Show actual label if available
                if actual_label is not None:
                    st.markdown("---")
                    st.markdown(f"**Actual Label**: {'Fraud' if actual_label == 1 else 'Legitimate'}")
                    correct = (fraud_prob > 0.5) == (actual_label == 1)
                    st.markdown(f"**Prediction**: {'✅ Correct' if correct else '❌ Incorrect'}")


def show_batch_analysis(model, device):
    """Show batch analysis interface"""
    st.markdown("## Batch Transaction Analysis")

    # Load data
    X, y, df = load_sample_data()

    if X is None:
        st.warning("No dataset found.")
        return

    # Batch size selection
    batch_size = st.slider("Number of Transactions to Analyze", 10, 1000, 100)

    if st.button("🚀 Run Batch Analysis", type="primary"):
        with st.spinner("Analyzing transactions..."):
            # Get random samples
            sample_indices = np.random.choice(len(X), min(batch_size, len(X)), replace=False)
            X_batch = X[sample_indices]
            y_batch = y[sample_indices]

            # Predict
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_batch).to(device)
                fraud_probs = model.predict_proba(X_tensor).cpu().numpy()

            # Results
            predictions = (fraud_probs > 0.5).astype(int)
            accuracy = (predictions == y_batch).mean()

            # Display metrics
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Analyzed", batch_size)
            with col2:
                st.metric("Predicted Frauds", int(predictions.sum()))
            with col3:
                st.metric("Actual Frauds", int(y_batch.sum()))
            with col4:
                st.metric("Accuracy", f"{accuracy*100:.1f}%")

            # Detailed results
            results_df = pd.DataFrame({
                'Index': sample_indices,
                'Fraud Probability': fraud_probs,
                'Prediction': ['Fraud' if p == 1 else 'Legitimate' for p in predictions],
                'Actual': ['Fraud' if label == 1 else 'Legitimate' for label in y_batch],
                'Correct': predictions == y_batch
            })

            # Show high-risk transactions
            st.markdown("### High-Risk Transactions")
            high_risk = results_df[results_df['Fraud Probability'] > 0.7].sort_values('Fraud Probability', ascending=False)

            if len(high_risk) > 0:
                st.dataframe(high_risk, use_container_width=True)
            else:
                st.info("No high-risk transactions found in this batch.")

            # Show all results
            with st.expander("View All Results"):
                st.dataframe(results_df, use_container_width=True)


if __name__ == "__main__":
    main()
