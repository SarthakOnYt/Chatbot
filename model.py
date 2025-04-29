import torch
import torch.nn as nn
import json

# Load config.json
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

vocab_size = config["vocab_size"]
embed_dim = config["embed_dim"]
hidden_dim = config["hidden_dim"]
num_layers = config["num_layers"]
num_heads = config.get("num_heads")
dropout = config.get("dropout", 0.1)
max_len = config.get("max_len", 2048)

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
    
    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1)  # causal mask
        mask = mask.masked_fill(mask == 1, float('-inf'))
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out

class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout):
        super().__init__()
        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x)
        x = x + self.attn(x)
        x = self.norm2(x)
        x = x + self.mlp(x)
        return x

class ScalableDecoderOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, x):
        bsz, seq_len = x.size()
        x = self.embed(x) + self.pos_embed[:, :seq_len]

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.fc(x)
        return logits
