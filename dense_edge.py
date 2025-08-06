"""DenseEdge: Transformer layer for inter-module communication."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DenseEdge(nn.Module):
    """Transformer encoder layer for processing hidden states between LM modules.
    
    This module maintains the tensor shape (T, d) where T is sequence length 
    and d is hidden dimension (896 for Qwen2-0.5B).
    """
    
    def __init__(
        self,
        d_model: int = 896,  # Hidden size for Qwen2-0.5B
        n_heads: int = 8,
        d_ff: int = 3584,  # 4 * d_model
        n_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        layer_norm_eps: float = 1e-5,
    ):
        """Initialize DenseEdge transformer.
        
        Args:
            d_model: Model dimension (must match LM hidden size)
            n_heads: Number of attention heads
            d_ff: Feedforward dimension
            n_layers: Number of transformer layers
            dropout: Dropout probability
            activation: Activation function ("gelu" or "relu")
            layer_norm_eps: Layer normalization epsilon
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Ensure d_model is divisible by n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        # Create transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=False,  # We use [T, d] format
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model, eps=layer_norm_eps),
        )
        
        # Optional: Projection layers for dimension matching if needed
        self.input_projection = None
        self.output_projection = None
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def add_projection(self, input_dim: int, output_dim: int):
        """Add projection layers for dimension matching.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
        """
        if input_dim != self.d_model:
            self.input_projection = nn.Linear(input_dim, self.d_model)
            
        if output_dim != self.d_model:
            self.output_projection = nn.Linear(self.d_model, output_dim)
            
        # Initialize projection weights
        if self.input_projection is not None:
            nn.init.xavier_uniform_(self.input_projection.weight)
            nn.init.zeros_(self.input_projection.bias)
            
        if self.output_projection is not None:
            nn.init.xavier_uniform_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)
    
    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """Process hidden states through transformer layers.
        
        Args:
            h: Hidden state tensor of shape [T, d] or [B, T, d]
            mask: Optional attention mask
            return_attention: Whether to return attention weights (not implemented)
            
        Returns:
            Processed hidden state tensor maintaining input shape
        """
        # Handle batch dimension
        if h.dim() == 2:
            # [T, d] - single sequence
            h_input = h
            squeeze_output = True
        elif h.dim() == 3:
            # [B, T, d] - batched sequences
            # Transformer expects [T, B, d] when batch_first=False
            h_input = h.transpose(0, 1)  # [T, B, d]
            squeeze_output = False
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got {h.dim()}D")
        
        # Apply input projection if needed
        if self.input_projection is not None:
            h_input = self.input_projection(h_input)
        
        # Process through transformer
        h_output = self.transformer(
            h_input,
            mask=mask,
            src_key_padding_mask=None,  # Could add padding mask support
        )
        
        # Apply output projection if needed
        if self.output_projection is not None:
            h_output = self.output_projection(h_output)
        
        # Restore original format
        if not squeeze_output and h.dim() == 3:
            # Convert back to [B, T, d]
            h_output = h_output.transpose(0, 1)
        
        return h_output
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters.
        
        Args:
            trainable_only: Whether to count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())
    
    def freeze(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
            
    def save_state_dict(self, path: str):
        """Save model state dict.
        
        Args:
            path: Path to save state dict
        """
        torch.save(self.state_dict(), path)
        
    def load_state_dict_from_path(self, path: str, strict: bool = True):
        """Load model state dict from path.
        
        Args:
            path: Path to load state dict from
            strict: Whether to strictly enforce matching keys
        """
        state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(state_dict, strict=strict)