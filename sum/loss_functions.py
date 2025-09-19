import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SUMLossFunction(nn.Module):
    def __init__(self, 
                 lambda_commitment: float = 0.3,
                 lambda_deception: float = 0.1,
                 temperature: float = 1.0):
        super(SUMLossFunction, self).__init__()
        self.lambda_commitment = lambda_commitment
        self.lambda_deception = lambda_deception
        self.temperature = temperature
        
        logger.info(f"SUMLossFunction initialized with lambda_commitment={lambda_commitment}, lambda_deception={lambda_deception}")
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor],
                game_context: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        
        strategy_loss = self._compute_strategy_loss(predictions, targets)
        commitment_loss = self._compute_commitment_loss(predictions, targets, game_context)
        deception_loss = self._compute_deception_loss(predictions, targets, game_context)
        
        total_loss = (
            strategy_loss + 
            self.lambda_commitment * commitment_loss + 
            self.lambda_deception * deception_loss
        )
        
        return {
            'total_loss': total_loss,
            'strategy_loss': strategy_loss,
            'commitment_loss': commitment_loss,
            'deception_loss': deception_loss
        }
    
    def _compute_strategy_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        predicted_actions = predictions['weighted_strategy']
        target_actions = targets['actions']
        
        if target_actions.dtype == torch.long:
            strategy_loss = F.cross_entropy(predicted_actions, target_actions)
        else:
            strategy_loss = F.kl_div(
                F.log_softmax(predicted_actions / self.temperature, dim=-1),
                F.softmax(target_actions / self.temperature, dim=-1),
                reduction='batchmean'
            )
        
        return strategy_loss
    
    def _compute_commitment_loss(self, 
                                predictions: Dict[str, torch.Tensor], 
                                targets: Dict[str, torch.Tensor],
                                game_context: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        commitment_prob = predictions['commitment_prob'].squeeze(-1)
        uncertainty = predictions['uncertainty']
        
        target_commitment = targets.get('commitment', None)
        if target_commitment is not None:
            commitment_loss = F.binary_cross_entropy(commitment_prob, target_commitment.float())
        else:
            information_gain = self._calculate_information_gain(game_context)
            optimal_commitment = (uncertainty < 0.5) & (information_gain > 0.3)
            commitment_loss = F.binary_cross_entropy(commitment_prob, optimal_commitment.float())
        
        uncertainty_penalty = torch.mean(uncertainty * commitment_prob)
        
        total_commitment_loss = commitment_loss + 0.1 * uncertainty_penalty
        
        return total_commitment_loss
    
    def _compute_deception_loss(self, 
                               predictions: Dict[str, torch.Tensor], 
                               targets: Dict[str, torch.Tensor],
                               game_context: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        strategies = predictions['strategies']
        weights = predictions['weights']
        
        strategy_diversity = self._calculate_strategy_diversity(strategies)
        
        opponent_confusion = self._calculate_opponent_confusion(predictions, game_context)
        
        hand_strength = game_context.get('hand_strength', torch.zeros_like(weights[:, 0]))
        
        bluff_incentive = self._calculate_bluff_incentive(hand_strength, strategies, weights)
        
        deception_reward = strategy_diversity + opponent_confusion + bluff_incentive
        
        deception_loss = -torch.mean(deception_reward)
        
        return deception_loss
    
    def _calculate_information_gain(self, game_context: Dict[str, torch.Tensor]) -> torch.Tensor:
        pot_size = game_context.get('pot_size', torch.ones(game_context['hand_strength'].size(0)))
        stack_sizes = game_context.get('stack_sizes', torch.ones_like(pot_size))
        
        pot_odds = pot_size / (pot_size + stack_sizes)
        
        information_gain = torch.clamp(pot_odds, 0.0, 1.0)
        
        return information_gain
    
    def _calculate_strategy_diversity(self, strategies: torch.Tensor) -> torch.Tensor:
        batch_size, num_strategies, action_dim = strategies.shape
        
        pairwise_kl = torch.zeros(batch_size, device=strategies.device)
        
        for i in range(num_strategies):
            for j in range(i + 1, num_strategies):
                kl_div = F.kl_div(
                    F.log_softmax(strategies[:, i], dim=-1),
                    F.softmax(strategies[:, j], dim=-1),
                    reduction='none'
                ).sum(dim=-1)
                pairwise_kl += kl_div
        
        diversity = pairwise_kl / (num_strategies * (num_strategies - 1) / 2)
        
        return diversity
    
    def _calculate_opponent_confusion(self, predictions: Dict[str, torch.Tensor], game_context: Dict[str, torch.Tensor]) -> torch.Tensor:
        strategies = predictions['strategies']
        weights = predictions['weights']
        
        weighted_strategy = torch.sum(strategies * weights.unsqueeze(-1), dim=1)
        
        action_entropy = -torch.sum(weighted_strategy * torch.log(weighted_strategy + 1e-8), dim=-1)
        
        max_entropy = torch.log(torch.tensor(strategies.size(-1), dtype=torch.float32))
        normalized_entropy = action_entropy / max_entropy
        
        return normalized_entropy
    
    def _calculate_bluff_incentive(self, hand_strength: torch.Tensor, strategies: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # Determine batch size from strategies tensor (most reliable)
        if strategies.dim() >= 2:
            batch_size = strategies.shape[0]
        elif weights.dim() >= 1:
            batch_size = weights.shape[0]
        else:
            batch_size = 1
        
        # Ensure hand_strength is properly shaped [batch_size]
        if hand_strength.dim() == 0:
            hand_strength = hand_strength.unsqueeze(0).expand(batch_size)
        elif hand_strength.dim() > 1:
            hand_strength = hand_strength.view(-1)
            if hand_strength.shape[0] != batch_size:
                hand_strength = hand_strength[:batch_size] if hand_strength.shape[0] > batch_size else hand_strength.expand(batch_size)
        elif hand_strength.shape[0] != batch_size:
            hand_strength = hand_strength.expand(batch_size)
        
        # Ensure strategies has shape [batch_size, num_strategies, action_dim]
        if strategies.dim() == 2:
            # [batch_size, action_dim] -> [batch_size, 1, action_dim]
            strategies = strategies.unsqueeze(1)
        elif strategies.dim() == 1:
            # [action_dim] -> [1, 1, action_dim] -> [batch_size, 1, action_dim]
            strategies = strategies.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        
        # Ensure weights has shape [batch_size, num_strategies]
        if weights.dim() == 1:
            if weights.shape[0] == batch_size:
                # [batch_size] -> [batch_size, 1]
                weights = weights.unsqueeze(1)
            else:
                # [num_strategies] -> [1, num_strategies] -> [batch_size, num_strategies]
                weights = weights.unsqueeze(0).expand(batch_size, -1)
        elif weights.dim() == 0:
            # scalar -> [batch_size, 1]
            weights = weights.unsqueeze(0).unsqueeze(0).expand(batch_size, 1)
        elif weights.shape[0] != batch_size:
            # Adjust batch dimension
            weights = weights[:batch_size] if weights.shape[0] > batch_size else weights.expand(batch_size, -1)
        
        # Ensure dimensions match
        num_strategies = strategies.shape[1]
        if weights.shape[1] != num_strategies:
            if weights.shape[1] > num_strategies:
                weights = weights[:, :num_strategies]
            else:
                padding = torch.ones(batch_size, num_strategies - weights.shape[1], device=weights.device) / num_strategies
                weights = torch.cat([weights, padding], dim=1)
        
        # Calculate aggressive actions (last action dimension)
        aggressive_actions = strategies[:, :, -1]  # [batch_size, num_strategies]
        weighted_aggression = torch.sum(aggressive_actions * weights, dim=1)  # [batch_size]
        
        # Ensure all tensors have the same batch size
        min_size = min(hand_strength.shape[0], weighted_aggression.shape[0])
        hand_strength = hand_strength[:min_size]
        weighted_aggression = weighted_aggression[:min_size]
        
        weak_hand_mask = hand_strength < 0.3
        strong_hand_mask = hand_strength > 0.7
        
        bluff_reward = torch.zeros_like(hand_strength)
        
        if weak_hand_mask.any():
            bluff_reward[weak_hand_mask] = weighted_aggression[weak_hand_mask]
        
        if strong_hand_mask.any():
            bluff_reward[strong_hand_mask] = weighted_aggression[strong_hand_mask]
        
        return bluff_reward

class StrategyRegularizer(nn.Module):
    def __init__(self, regularization_strength: float = 0.01):
        super(StrategyRegularizer, self).__init__()
        self.regularization_strength = regularization_strength
        
    def forward(self, strategies: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weight_entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)
        
        strategy_smoothness = torch.mean(torch.var(strategies, dim=1), dim=-1)
        
        regularization_loss = self.regularization_strength * (torch.mean(weight_entropy) + torch.mean(strategy_smoothness))
        
        return regularization_loss

class AdaptiveLossWeights(nn.Module):
    def __init__(self, initial_lambda_commitment: float = 0.3, initial_lambda_deception: float = 0.1):
        super(AdaptiveLossWeights, self).__init__()
        self.lambda_commitment = nn.Parameter(torch.tensor(initial_lambda_commitment))
        self.lambda_deception = nn.Parameter(torch.tensor(initial_lambda_deception))
        
        self.performance_history = []
        self.adaptation_rate = 0.01
        
    def forward(self, loss_components: Dict[str, torch.Tensor], performance_metrics: Dict[str, float]) -> Dict[str, torch.Tensor]:
        self.performance_history.append(performance_metrics)
        
        if len(self.performance_history) > 10:
            self._adapt_weights()
        
        total_loss = (
            loss_components['strategy_loss'] + 
            self.lambda_commitment * loss_components['commitment_loss'] + 
            self.lambda_deception * loss_components['deception_loss']
        )
        
        return {
            'total_loss': total_loss,
            'lambda_commitment': self.lambda_commitment,
            'lambda_deception': self.lambda_deception
        }
    
    def _adapt_weights(self):
        recent_performance = self.performance_history[-10:]
        
        win_rate_trend = np.mean([p.get('win_rate', 0.5) for p in recent_performance[-5:]]) - np.mean([p.get('win_rate', 0.5) for p in recent_performance[:5]])
        
        if win_rate_trend < 0:
            self.lambda_commitment.data += self.adaptation_rate
            self.lambda_deception.data += self.adaptation_rate * 0.5
        else:
            self.lambda_commitment.data = torch.clamp(self.lambda_commitment.data - self.adaptation_rate * 0.5, 0.1, 1.0)
            self.lambda_deception.data = torch.clamp(self.lambda_deception.data - self.adaptation_rate * 0.25, 0.05, 0.5)

class MultiObjectiveLossManager:
    def __init__(self, 
                 lambda_commitment: float = 0.3,
                 lambda_deception: float = 0.1,
                 use_adaptive_weights: bool = False,
                 regularization_strength: float = 0.01):
        
        self.use_adaptive_weights = use_adaptive_weights
        
        if use_adaptive_weights:
            self.loss_function = SUMLossFunction(lambda_commitment=1.0, lambda_deception=1.0)
            self.adaptive_weights = AdaptiveLossWeights(lambda_commitment, lambda_deception)
        else:
            self.loss_function = SUMLossFunction(lambda_commitment, lambda_deception)
            self.adaptive_weights = None
        
        self.regularizer = StrategyRegularizer(regularization_strength)
        
        logger.info(f"MultiObjectiveLossManager initialized with adaptive_weights={use_adaptive_weights}")
    
    def compute_loss(self, 
                    predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    game_context: Dict[str, torch.Tensor],
                    performance_metrics: Dict[str, float] = None) -> Dict[str, torch.Tensor]:
        
        loss_components = self.loss_function(predictions, targets, game_context)
        
        regularization_loss = self.regularizer(predictions['strategies'], predictions['weights'])
        
        if self.use_adaptive_weights and performance_metrics is not None:
            adaptive_output = self.adaptive_weights(loss_components, performance_metrics)
            total_loss = adaptive_output['total_loss'] + regularization_loss
            
            return {
                **loss_components,
                'total_loss': total_loss,
                'regularization_loss': regularization_loss,
                'lambda_commitment': adaptive_output['lambda_commitment'],
                'lambda_deception': adaptive_output['lambda_deception']
            }
        else:
            total_loss = loss_components['total_loss'] + regularization_loss
            
            return {
                **loss_components,
                'total_loss': total_loss,
                'regularization_loss': regularization_loss
            }