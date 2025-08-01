import torch
import torch.nn as nn

class LoLAttentionWinPredictor(nn.Module):
    #Based on champions IDS. Find upper range for num_champions for ids.
    def __init__(self, num_champions=1000, embedding_dim=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.flatten = nn.Flatten()  
        
        self.playerCount = 10
        self.inFeatures = embedding_dim * self.playerCount
        self.outFeatures = 128

        # Step 1: Embed champion IDs
        self.embedding = nn.Embedding(
            num_embeddings = num_champions, 
            embedding_dim = embedding_dim #32 is good because we don't have that much samples + not too many champions.
        )

        # Step 2: Multi-Head Self-Attention
            #seq_len = 10 for champions in game
            #embeded_dim = how long vector is for each champion_id.
        self.attn = nn.MultiheadAttention(
            embed_dim = embedding_dim, 
            num_heads = num_heads, 
            batch_first = True #batch dimension first. Means inputs/outputs are shaped (batch, seq_len, embed_dim) instead of (seq_len, batch, embed_dim).
        )

        # Step 3: Residual connection + LayerNorm for stable learning
        self.norm = nn.LayerNorm(embedding_dim) 

        # Step 4: Dropout for regularization
        self.dropout = nn.Dropout(dropout) #Helps with generalizing, stops overfitting.

        # Step 5: Classifier MLP
            #Essentially one hidden linear layer might not be enough for fitting. Make another if underfitting.
        self.classifier = nn.Sequential(
            nn.Linear(self.inFeatures, self.outFeatures), #embedding_dim x players, outFeatures = 128
            nn.ReLU(),
            nn.Dropout(dropout), #randomly zeroing out. Prevents overfitting, prevent being dependent on popular champs.
            nn.Linear(self.outFeatures, 1)
        )

    def forward(self, x):  # x: (batch_size, 10)
        emb = self.embedding(x)                  # (B, 10, E)

        # Attention: contextualize each champ embedding
        attn_output, attn_output_weights = self.attn(emb, emb, emb)   # (B, 10, E)

        # Add & Normalize (residual connection)
        out = self.norm(emb + attn_output) #adds attention to original embeddings w/ residual then normalizes (converts 0-1).
        out = self.dropout(out) #Needed?

        # Flatten for MLP classifier --> Already flatten on default (for now).
        out = self.flatten(out)          # (B, 10*E)

        logits = self.classifier(out)            # (B, 1)
        return logits.squeeze(1)                 # (B,) - raw logits
