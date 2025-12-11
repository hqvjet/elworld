import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embedding=512, embedding_dim=64, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # This is the codebook contains 512 embeddings, each of dimension 64
        self.embeddings = nn.Embedding(self.num_embedding, self.embedding_dim) # [512, 64]
        self.embeddings.weight.data.uniform_(-1/self.num_embedding, 1/self.num_embedding)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous() # [B, 64, 24, 32] -> [B, 24, 32, 64]
        flat_x = x.view(-1, self.embedding_dim) # x: [B*24*32, 64]

        # Compute distances
        distances = (
            torch.sum(flat_x**2, dim=1, keepdim=True) 
            + torch.sum(self.embeddings.weight**2, dim=1)
            - 2 * torch.matmul(flat_x, self.embeddings.weight.t())
        )  # [B*24*32, 512]

        # Get the encoding that has the min distance
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)  # [B*24*32, 1]
        encodings = torch.zeros(encoding_indices.size(0), self.num_embedding, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)  # One-hot encodings

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight)  # [B*24*32, 64]
        quantized = quantized.view(x.size())  # [B, 24, 32, 64]

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), x) # commitment loss
        q_latent_loss = F.mse_loss(quantized, x.detach()) # codebook loss
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # [B, 64, 24, 32]

        perplexity = torch.exp(-torch.sum(encodings * torch.log(encodings + 1e-10)) / encodings.size(0))

        return {
            'quantized': quantized, # [B, 64, 24, 32]
            'loss': loss,
            'perplexity': perplexity,
            'encoding_indices': encoding_indices.view(x.size(0), x.size(1), x.size(2))  # [B, 24, 32]
        }