"""
Transformer-based Foundation Model for Payment Transactions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransactionTransformerBlock(nn.Module):
    """Single transformer block"""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            hidden_size,
            num_attention_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.attention_norm(x + attn_out)

        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)

        return x


class PaymentTransactionEncoder(nn.Module):
    """
    Foundation Model: Transformer-based encoder for payment transactions
    This model learns rich representations of transaction patterns
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        pooling: str = "cls"  # cls or mean
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.pooling = pooling

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_size)

        # CLS token for classification tasks (learned)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_size, max_position_embeddings, dropout)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransactionTransformerBlock(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout=dropout
            )
            for _ in range(num_hidden_layers)
        ])

        # Output normalization
        self.output_norm = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_all_tokens: bool = False
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor of shape [batch_size, input_dim] or [batch_size, seq_len, input_dim]
            return_all_tokens: If True, return all token representations, else return pooled output

        Returns:
            Tensor of shape [batch_size, hidden_size] or [batch_size, seq_len, hidden_size]
        """
        batch_size = x.size(0)

        # Handle single transaction (no sequence dimension)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, input_dim]

        # Project input to hidden size
        x = self.input_projection(x)  # [batch_size, seq_len, hidden_size]

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_size]
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, seq_len+1, hidden_size]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final normalization
        x = self.output_norm(x)

        # Return representations
        if return_all_tokens:
            return x
        else:
            # Pooling
            if self.pooling == "cls":
                return x[:, 0, :]  # CLS token representation
            elif self.pooling == "mean":
                return x[:, 1:, :].mean(dim=1)  # Mean of all tokens (excluding CLS)
            else:
                return x[:, 0, :]

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get transaction embeddings (for similarity search, clustering, etc.)"""
        with torch.no_grad():
            return self.forward(x, return_all_tokens=False)


class FraudDetectionHead(nn.Module):
    """Classification head for fraud detection (fine-tuning)"""

    def __init__(self, hidden_size: int, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class PaymentFraudDetector(nn.Module):
    """
    Complete model: Foundation model + fraud detection head
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 1024,
        max_position_embeddings: int = 512,
        dropout: float = 0.1,
        num_classes: int = 2
    ):
        super().__init__()

        # Foundation model (encoder)
        self.encoder = PaymentTransactionEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout
        )

        # Classification head
        self.classifier = FraudDetectionHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get embeddings from encoder
        embeddings = self.encoder(x)

        # Classify
        logits = self.classifier(embeddings)

        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get transaction embeddings"""
        return self.encoder.get_embeddings(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get fraud probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            return probs[:, 1]  # Return probability of fraud class


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    batch_size = 32
    input_dim = 30  # Credit card fraud dataset has 30 features
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    model = PaymentFraudDetector(
        input_dim=input_dim,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(batch_size, input_dim).to(device)
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")

    # Test embeddings
    embeddings = model.get_embeddings(x)
    print(f"Embeddings shape: {embeddings.shape}")

    # Test probabilities
    probs = model.predict_proba(x)
    print(f"Fraud probabilities shape: {probs.shape}")
