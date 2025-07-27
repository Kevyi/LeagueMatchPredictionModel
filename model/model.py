import torch
import torch.nn as nn

class LoLAttentionWinPredictor(nn.Module):
    def __init__(self, num_champions=1000, embedding_dim=32, num_heads=4, dropout=0.1):
        super().__init__()

        # Step 1: Embed champion IDs
        self.embedding = nn.Embedding(num_embeddings=num_champions, embedding_dim=embedding_dim)

        # Step 2: Multi-Head Self-Attention
        self.attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

        # Step 3: Residual connection + LayerNorm for stable learning
        self.norm = nn.LayerNorm(embedding_dim)

        # Step 4: Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Step 5: Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 10, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x):  # x: (batch_size, 10)
        emb = self.embedding(x)                  # (B, 10, E)

        # Attention: contextualize each champ embedding
        attn_out, _ = self.attn(emb, emb, emb)   # (B, 10, E)

        # Add & Normalize (residual connection)
        out = self.norm(emb + attn_out)          # (B, 10, E)
        out = self.dropout(out)

        # Flatten for MLP classifier
        out = out.view(out.size(0), -1)          # (B, 10*E)

        logits = self.classifier(out)            # (B, 1)
        return logits.squeeze(1)                 # (B,) - raw logits
