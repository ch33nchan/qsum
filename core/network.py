import torch
import torch.nn as nn
from typing import Tuple

class SUMNetworkCPU(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 128, n_actions: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        self.action_head = nn.Linear(hidden_dim // 2, n_actions)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        self.uncertainty_head = nn.Linear(hidden_dim // 2, 8)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        
        action_logits = self.action_head(features)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        value = self.value_head(features)
        
        uncertainty_logits = self.uncertainty_head(features)
        uncertainty_probs = torch.softmax(uncertainty_logits, dim=-1)
        
        return action_probs, value, uncertainty_probs

class SUMNetworkGPU(nn.Module):
    def __init__(self, input_dim: int = 64, hidden_dim: int = 512, n_actions: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )
        
        self.action_head = nn.Linear(hidden_dim // 2, n_actions)
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        self.uncertainty_head = nn.Linear(hidden_dim // 2, 16)
        self.strategy_head = nn.Linear(hidden_dim // 2, 8)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        
        action_logits = self.action_head(features)
        action_probs = torch.softmax(action_logits, dim=-1)
        
        value = self.value_head(features)
        
        uncertainty_logits = self.uncertainty_head(features)
        uncertainty_probs = torch.softmax(uncertainty_logits, dim=-1)
        
        strategy_logits = self.strategy_head(features)
        strategy_probs = torch.softmax(strategy_logits, dim=-1)
        
        return action_probs, value, uncertainty_probs, strategy_probs