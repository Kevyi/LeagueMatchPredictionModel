import torch
import torch.nn as nn

# Define an Embedding layer
# num_embeddings=10 means there are 10 unique entities (e.g., words)
# embedding_dim=5 means each entity will be represented by a 5-dimensional vector
embedding_layer = nn.Embedding(num_embeddings=10, embedding_dim=5)

# Example input: a tensor of indices (e.g., tokenized words)
input_indices = torch.tensor([1])

# Forward pass through the embedding layer
output_embeddings = embedding_layer(input_indices)

print(output_embeddings)