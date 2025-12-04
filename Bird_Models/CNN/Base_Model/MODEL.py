"""
MODEL.py - Model Architecture and Training
Contains the model definition and training loop.
"""

from networkx import config
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from typing import Tuple, Dict, List
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
from torchvision.models import ResNet18_Weights
from PREP import EpisodeSampler


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class EmbeddingNetwork(nn.Module):
    """
    ResNet18 backbone without projection layers.
    Returns 512-D L2-normalized embeddings.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        if pretrained:
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        self.embedding_dim = 512

        # Remove final FC layer - use ResNet features directly
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.encoder.parameters():
            param.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ResNet18 backbone without projection layers.
        Returns 512-D L2-normalized embeddings.
        """
        x = self.encoder(x)  # (batch, 512, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 512)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize
        return x


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Network using EmbeddingNetwork (fixed ResNet18 backbone).
    """ 
    
    def __init__(self, embedding_network: EmbeddingNetwork):
        super().__init__()
        self.embedding_network = embedding_network
        
    def compute_prototypes(self, support_embeddings: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        """
        Compute class prototypes from support embeddings.
        
        Args:
            support_embeddings: (n_way * k_shot, embedding_dim)
            n_way: Number of classes
            k_shot: Support examples per class
        
        Returns:
            prototypes: (n_way, embedding_dim)
        """
        # Reshape and compute mean per class
        embedding_dim = support_embeddings.size(-1)
        support_embeddings = support_embeddings.view(n_way, k_shot, embedding_dim)
        prototypes = support_embeddings.mean(dim=1)
        return prototypes
    
    def forward(self, support_images: torch.Tensor, query_images: torch.Tensor, n_way: int, k_shot: int) -> torch.Tensor:
        """
        Forward pass for one episode.
        
        Args:
            support_images: (n_way * k_shot, 3, H, W)
            query_images: (n_query, 3, H, W)
            n_way: Number of classes
            k_shot: Support examples per class
        
        Returns:
            logits: (n_query, n_way)
        """
        # Embed support and query
        support_embeddings = self.embedding_network(support_images)
        query_embeddings = self.embedding_network(query_images)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_embeddings, n_way, k_shot)
        
        # Compute Euclidean distances
        distances = torch.cdist(query_embeddings, prototypes, p=2)
        
        # Convert to logits (negative distances)
        return -distances


# ============================================================================
# MODEL LOADING
# ============================================================================

def create_frozen_baseline(device: str = 'cuda') -> PrototypicalNetwork:
    """Create frozen ImageNet baseline - no training needed."""
    embedding_net = EmbeddingNetwork(pretrained=True)  # Already frozen
    model = PrototypicalNetwork(embedding_network=embedding_net)
    model.to(device)
    model.eval()
    return model