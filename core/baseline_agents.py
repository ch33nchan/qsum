#!/usr/bin/env python3
"""
Baseline Agents for Research Comparison
Implements classical poker strategies for academic validation
"""

import numpy as np
import random
from typing import Dict, Tuple
from .utils import HandEvaluator, Action

class BaselineAgent:
    """Base class for baseline agents"""
    def __init__(self, name: str):
        self.name = name
        self.training_metrics = {
            'episode_rewards': [],
            'win_rates': []
        }
    
    def get_action(self, state: Dict) -> int:
        raise NotImplementedError
    
    def store_experience(self, state: Dict, action: int, reward: float, next_state: Dict, done: bool):
        pass  # Baseline agents don't learn
    
    def train(self) -> float:
        return 0  # No training for baseline agents

class RandomAgent(BaselineAgent):
    """Random action baseline"""
    def __init__(self):
        super().__init__("Random")
    
    def get_action(self, state: Dict) -> int:
        return random.randint(0, 2)

class TightAggressiveAgent(BaselineAgent):
    """Tight-aggressive classical strategy"""
    def __init__(self):
        super().__init__("Tight-Aggressive")
    
    def get_action(self, state: Dict) -> int:
        if not state['player_cards']:
            return Action.FOLD.value
        
        hand_strength = HandEvaluator.evaluate_hand(
            state['player_cards'], state['community_cards']
        )
        
        pot_odds = state['pot'] / (state['player_stack'] + state['pot']) if state['player_stack'] > 0 else 0
        
        # Tight-aggressive logic
        if hand_strength > 0.8:  # Very strong hands
            return Action.RAISE.value
        elif hand_strength > 0.6:  # Strong hands
            if pot_odds < 0.3:
                return Action.RAISE.value
            else:
                return Action.CALL.value
        elif hand_strength > 0.4:  # Medium hands
            if pot_odds < 0.2:
                return Action.CALL.value
            else:
                return Action.FOLD.value
        else:  # Weak hands
            return Action.FOLD.value

class LoosePassiveAgent(BaselineAgent):
    """Loose-passive classical strategy"""
    def __init__(self):
        super().__init__("Loose-Passive")
    
    def get_action(self, state: Dict) -> int:
        if not state['player_cards']:
            return Action.CALL.value
        
        hand_strength = HandEvaluator.evaluate_hand(
            state['player_cards'], state['community_cards']
        )
        
        # Loose-passive logic - calls with most hands
        if hand_strength > 0.7:
            return Action.RAISE.value
        elif hand_strength > 0.2:
            return Action.CALL.value
        else:
            return Action.FOLD.value

class ClassicalMixedStrategyAgent(BaselineAgent):
    """Classical mixed strategy using game theory"""
    def __init__(self):
        super().__init__("Classical-Mixed-Strategy")
        # Pre-computed mixed strategy probabilities
        self.strategy_matrix = {
            'strong': [0.1, 0.2, 0.7],    # [fold, call, raise] for strong hands
            'medium': [0.3, 0.6, 0.1],   # [fold, call, raise] for medium hands
            'weak': [0.7, 0.25, 0.05]    # [fold, call, raise] for weak hands
        }
    
    def get_action(self, state: Dict) -> int:
        if not state['player_cards']:
            return Action.FOLD.value
        
        hand_strength = HandEvaluator.evaluate_hand(
            state['player_cards'], state['community_cards']
        )
        
        # Categorize hand strength
        if hand_strength > 0.7:
            strategy = 'strong'
        elif hand_strength > 0.4:
            strategy = 'medium'
        else:
            strategy = 'weak'
        
        # Sample action from mixed strategy
        probabilities = self.strategy_matrix[strategy]
        return np.random.choice([0, 1, 2], p=probabilities)

class NashEquilibriumAgent(BaselineAgent):
    """Approximation of Nash equilibrium strategy"""
    def __init__(self):
        super().__init__("Nash-Equilibrium")
        # Simplified Nash equilibrium approximation
        self.bluff_frequency = 0.15  # Optimal bluffing frequency
        self.value_bet_frequency = 0.8  # Value betting with strong hands
    
    def get_action(self, state: Dict) -> int:
        if not state['player_cards']:
            return Action.FOLD.value
        
        hand_strength = HandEvaluator.evaluate_hand(
            state['player_cards'], state['community_cards']
        )
        
        pot_odds = state['pot'] / (state['player_stack'] + state['pot']) if state['player_stack'] > 0 else 0
        
        # Nash equilibrium approximation
        if hand_strength > 0.8:  # Very strong hands
            if random.random() < self.value_bet_frequency:
                return Action.RAISE.value
            else:
                return Action.CALL.value
        elif hand_strength > 0.6:  # Strong hands
            return Action.CALL.value
        elif hand_strength > 0.3:  # Medium hands
            if pot_odds < 0.25:
                return Action.CALL.value
            else:
                return Action.FOLD.value
        else:  # Weak hands
            # Optimal bluffing
            if random.random() < self.bluff_frequency and state['street'] > 0:
                return Action.RAISE.value
            else:
                return Action.FOLD.value

def get_baseline_agent(agent_type: str) -> BaselineAgent:
    """Factory function for baseline agents"""
    agents = {
        'random': RandomAgent,
        'tight_aggressive': TightAggressiveAgent,
        'loose_passive': LoosePassiveAgent,
        'classical_mixed_strategy': ClassicalMixedStrategyAgent,
        'nash_equilibrium': NashEquilibriumAgent
    }
    
    if agent_type not in agents:
        raise ValueError(f"Unknown baseline agent type: {agent_type}")
    
    return agents[agent_type]()