import torch
import torch.nn as nn

#Two multiheaded embeddings, Cross attention model.
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
        # Blue attends to red, and vice versa
        self.attn_blue = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)
        self.attn_red  = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, batch_first=True)

        # Step 3: Residual connection + LayerNorm for stable learning
        self.norm_blue = nn.LayerNorm(embedding_dim)
        self.norm_red  = nn.LayerNorm(embedding_dim)

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
        blue_ids = x[:, :5]  # (B, 5)
        red_ids  = x[:, 5:]  # (B, 5)

        blue_emb = self.embedding(blue_ids)  # (B, 5, E)
        red_emb  = self.embedding(red_ids)   # (B, 5, E)

        # Attention: contextualize each champ embedding
        # Blue attends to red
        blue_attn, _ = self.attn_blue(query=blue_emb, key=red_emb, value=red_emb)  # (B, 5, E)
        blue_ctx = self.norm_blue(blue_emb + self.dropout(blue_attn))              # residual + norm

        # Red attends to blue
        red_attn, _ = self.attn_red(query=red_emb, key=blue_emb, value=blue_emb)
        red_ctx = self.norm_red(red_emb + self.dropout(red_attn))

        # Concatenate both attention contexts (matchup-aware)
        full = torch.cat([blue_ctx, red_ctx], dim=1)  # (B, 10, E)

        # Flatten for MLP classifier --> Already flatten on default (for now).
        out = self.flatten(full)          # (B, 10*E)

        logits = self.classifier(out)            # (B, 1)
        return logits.squeeze(1)                 # (B,) - raw logits
