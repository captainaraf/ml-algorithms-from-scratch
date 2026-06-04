from helpers import TransformerEncoderLayer, PositionalEncoding
from torch import nn
import torch

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff=None, max_len=5000, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=mask)

        return x