import torch
import torch.nn as nn
from typing import Tuple

class MLPAutoEncoder(nn.Module):
    def __init__(self, dim_in, dim_latent=128, hidden=512, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(dim_latent, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim_in),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
