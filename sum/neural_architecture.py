import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameEncoder(nn.Module):
    def __init__(self, input_dim: int = 52, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Card features: 52 cards -> 52*32 = 1664 features when flattened
        self.card_encoder = nn.Linear(52, 64)  # Compress card vector
        self.position_embedding = nn.Embedding(2, 16)
        self.stack_encoder = nn.Linear(2, 32)
        self.pot_encoder = nn.Linear(1, 16)
        
        # Total input: raw_features(6) + card_features(64) + position(16) + stacks(32) + pot(16) = 134
        # Actual measured input dimension is 134
        total_input_dim = 134
        
        self.encoder_layers = nn.Sequential(
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
        
        logger.info(f"GameEncoder initialized with total_input_dim={total_input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
    
    def forward(self, game_state: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size = game_state['cards'].size(0)
        
        # Convert cards to float for linear layer
        card_features = self.card_encoder(game_state['cards'].float())
        position_features = self.position_embedding(game_state['position'])
        stack_features = self.stack_encoder(game_state['stacks'])
        pot_features = self.pot_encoder(game_state['pot'].unsqueeze(-1))
        
        combined_features = torch.cat([
            game_state['raw_features'],
            card_features,
            position_features.squeeze(1),
            stack_features.squeeze(1),
            pot_features.squeeze(1)
        ], dim=-1)
        
        return self.encoder_layers(combined_features)

class HistoryEncoder(nn.Module):
    def __init__(self, action_dim: int = 4, hidden_dim: int = 128, output_dim: int = 64):
        super(HistoryEncoder, self).__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.action_embedding = nn.Embedding(action_dim, 32)
        self.amount_encoder = nn.Linear(1, 16)
        
        self.lstm = nn.LSTM(
            input_size=32 + 16,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        logger.info(f"HistoryEncoder initialized with action_dim={action_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
    
    def forward(self, action_history: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = action_history['actions'].size()
        
        action_embeds = self.action_embedding(action_history['actions'])
        amount_features = self.amount_encoder(action_history['amounts'].unsqueeze(-1))
        
        sequence_features = torch.cat([action_embeds, amount_features], dim=-1)
        
        lstm_out, (hidden, _) = self.lstm(sequence_features)
        
        history_encoding = self.output_projection(hidden[-1])
        return history_encoding

class StrategyHead(nn.Module):
    def __init__(self, input_dim: int = 192, hidden_dim: int = 256, num_strategies: int = 8, action_dim: int = 4):
        super(StrategyHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_strategies = num_strategies
        self.action_dim = action_dim
        
        self.strategy_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim)
            ) for _ in range(num_strategies)
        ])
        
        logger.info(f"StrategyHead initialized with {num_strategies} strategies, input_dim={input_dim}, action_dim={action_dim}")
    
    def forward(self, encoded_state: torch.Tensor) -> torch.Tensor:
        batch_size = encoded_state.size(0)
        strategies = torch.zeros(batch_size, self.num_strategies, self.action_dim, device=encoded_state.device)
        
        for i, strategy_net in enumerate(self.strategy_networks):
            strategy_logits = strategy_net(encoded_state)
            strategies[:, i, :] = F.softmax(strategy_logits, dim=-1)
        
        return strategies

class WeightHead(nn.Module):
    def __init__(self, input_dim: int = 192, hidden_dim: int = 128, num_strategies: int = 8):
        super(WeightHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_strategies = num_strategies
        
        self.weight_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_strategies)
        )
        
        logger.info(f"WeightHead initialized for {num_strategies} strategies, input_dim={input_dim}")
    
    def forward(self, encoded_state: torch.Tensor) -> torch.Tensor:
        weight_logits = self.weight_network(encoded_state)
        weights = F.softmax(weight_logits, dim=-1)
        return weights

class CommitmentHead(nn.Module):
    def __init__(self, input_dim: int = 192, hidden_dim: int = 128):
        super(CommitmentHead, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.commitment_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        logger.info(f"CommitmentHead initialized with input_dim={input_dim}, hidden_dim={hidden_dim}")
    
    def forward(self, encoded_state: torch.Tensor) -> torch.Tensor:
        commitment_logits = self.commitment_network(encoded_state)
        commitment_prob = torch.sigmoid(commitment_logits)
        return commitment_prob

class SUMNeuralArchitecture(nn.Module):
    def __init__(self, 
                 game_input_dim: int = 52,
                 action_dim: int = 4,
                 num_strategies: int = 8,
                 game_hidden_dim: int = 256,
                 history_hidden_dim: int = 128,
                 strategy_hidden_dim: int = 256,
                 weight_hidden_dim: int = 128,
                 commitment_hidden_dim: int = 128):
        super(SUMNeuralArchitecture, self).__init__()
        
        self.num_strategies = num_strategies
        self.action_dim = action_dim
        
        self.game_encoder = GameEncoder(
            input_dim=game_input_dim,
            hidden_dim=game_hidden_dim,
            output_dim=128
        )
        
        self.history_encoder = HistoryEncoder(
            action_dim=action_dim,
            hidden_dim=history_hidden_dim,
            output_dim=64
        )
        
        combined_dim = 128 + 64
        
        self.strategy_head = StrategyHead(
            input_dim=combined_dim,
            hidden_dim=strategy_hidden_dim,
            num_strategies=num_strategies,
            action_dim=action_dim
        )
        
        self.weight_head = WeightHead(
            input_dim=combined_dim,
            hidden_dim=weight_hidden_dim,
            num_strategies=num_strategies
        )
        
        self.commitment_head = CommitmentHead(
            input_dim=combined_dim,
            hidden_dim=commitment_hidden_dim
        )
        
        self.uncertainty_threshold = 0.5
        self.commitment_history = []
        
        logger.info(f"SUMNeuralArchitecture initialized with {num_strategies} strategies")
    
    def forward(self, game_state: Dict[str, torch.Tensor], action_history: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        game_encoding = self.game_encoder(game_state)
        history_encoding = self.history_encoder(action_history)
        
        combined_encoding = torch.cat([game_encoding, history_encoding], dim=-1)
        
        strategies = self.strategy_head(combined_encoding)
        weights = self.weight_head(combined_encoding)
        commitment_prob = self.commitment_head(combined_encoding)
        
        weighted_strategy = torch.sum(strategies * weights.unsqueeze(-1), dim=1)
        
        uncertainty = self._calculate_uncertainty(strategies, weights)
        
        return {
            'strategies': strategies,
            'weights': weights,
            'commitment_prob': commitment_prob,
            'weighted_strategy': weighted_strategy,
            'uncertainty': uncertainty,
            'combined_encoding': combined_encoding
        }
    
    def _calculate_uncertainty(self, strategies: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weighted_strategies = strategies * weights.unsqueeze(-1)
        strategy_variance = torch.var(weighted_strategies, dim=1)
        uncertainty = torch.mean(strategy_variance, dim=-1)
        return uncertainty
    
    def should_commit(self, uncertainty: torch.Tensor, commitment_prob: torch.Tensor) -> torch.Tensor:
        uncertainty_condition = uncertainty < self.uncertainty_threshold
        commitment_condition = commitment_prob.squeeze(-1) > 0.5
        
        should_commit = uncertainty_condition | commitment_condition
        return should_commit
    
    def get_action(self, game_state: Dict[str, torch.Tensor], action_history: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        with torch.no_grad():
            outputs = self.forward(game_state, action_history)
            
            should_commit = self.should_commit(outputs['uncertainty'], outputs['commitment_prob'])
            
            if should_commit.any():
                action_probs = outputs['weighted_strategy']
                self.commitment_history.append(True)
            else:
                random_strategy_idx = torch.randint(0, self.num_strategies, (game_state['cards'].size(0),))
                action_probs = outputs['strategies'][torch.arange(game_state['cards'].size(0)), random_strategy_idx]
                self.commitment_history.append(False)
            
            actions = torch.multinomial(action_probs, 1).squeeze(-1)
            
            return actions, outputs
    
    def reset_commitment_history(self):
        self.commitment_history = []
    
    def get_commitment_stats(self) -> Dict[str, float]:
        if not self.commitment_history:
            return {'commitment_rate': 0.0, 'total_decisions': 0}
        
        commitment_rate = sum(self.commitment_history) / len(self.commitment_history)
        return {
            'commitment_rate': commitment_rate,
            'total_decisions': len(self.commitment_history)
        }