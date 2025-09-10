import numpy as np
import torch
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class Action(Enum):
    FOLD = 0
    CALL = 1
    RAISE = 2

@dataclass
class Card:
    rank: int  # 0-12 (2-A)
    suit: int  # 0-3
    
    @property
    def value(self) -> int:
        return self.rank
    
    def __str__(self) -> str:
        ranks = '23456789TJQKA'
        suits = '♠♥♦♣'
        return f"{ranks[self.rank]}{suits[self.suit]}"

class Deck:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.cards = [Card(rank, suit) for rank in range(13) for suit in range(4)]
        random.shuffle(self.cards)
    
    def deal(self, n: int = 1) -> List[Card]:
        return [self.cards.pop() for _ in range(min(n, len(self.cards)))]

class HandEvaluator:
    @staticmethod
    def evaluate_hand(hole_cards: List[Card], community_cards: List[Card] = None) -> float:
        if community_cards is None:
            community_cards = []
        
        if len(hole_cards) != 2:
            return 0.0
        
        if len(community_cards) == 0:
            return HandEvaluator._preflop_strength(hole_cards)
        
        all_cards = hole_cards + community_cards
        return HandEvaluator._postflop_strength(all_cards)
    
    @staticmethod
    def _preflop_strength(hole_cards: List[Card]) -> float:
        card1, card2 = hole_cards
        
        # Pocket pairs
        if card1.rank == card2.rank:
            return 0.7 + (card1.rank / 12) * 0.3
        
        # Suited bonus
        suited_bonus = 0.1 if card1.suit == card2.suit else 0.0
        
        # High cards
        high_card_strength = (card1.rank + card2.rank) / 24
        
        # Connected cards
        gap = abs(card1.rank - card2.rank)
        connected_bonus = 0.05 if gap <= 1 else 0.0
        
        return min(1.0, high_card_strength + suited_bonus + connected_bonus)
    
    @staticmethod
    def _postflop_strength(all_cards: List[Card]) -> float:
        ranks = [card.rank for card in all_cards]
        suits = [card.suit for card in all_cards]
        
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1
        
        suit_counts = {}
        for suit in suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        max_rank_count = max(rank_counts.values())
        max_suit_count = max(suit_counts.values())
        
        if max_rank_count >= 4:
            return 0.9
        elif max_rank_count == 3:
            return 0.7
        elif max_suit_count >= 5:
            return 0.8
        elif list(rank_counts.values()).count(2) >= 2:
            return 0.5
        elif max_rank_count == 2:
            return 0.3
        else:
            return max(ranks) / 12

class StrategicUncertaintyState:
    def __init__(self, n_strategies: int = 8):
        self.n_strategies = n_strategies
        self.amplitudes = np.ones(n_strategies, dtype=np.complex64) / np.sqrt(n_strategies)
        self.collapsed = False
        self.collapsed_strategy = None
        
    def get_probabilities(self) -> np.ndarray:
        return np.abs(self.amplitudes) ** 2
    
    def get_entropy(self) -> float:
        probs = self.get_probabilities()
        probs = probs[probs > 1e-10]
        return -np.sum(probs * np.log(probs)) if len(probs) > 0 else 0.0
    
    def collapse_to_strategy(self, strategy_idx: int):
        self.amplitudes = np.zeros(self.n_strategies, dtype=np.complex64)
        self.amplitudes[strategy_idx] = 1.0
        self.collapsed = True
        self.collapsed_strategy = strategy_idx
    
    def should_collapse(self, opponent_action: str, context: Dict) -> bool:
        if self.collapsed:
            return False
        
        if opponent_action in ['raise', 'all_in']:
            return np.random.random() < 0.3
        
        pot_size = context.get('pot_size', 0)
        if pot_size > 50:
            return np.random.random() < 0.2
        
        return False

def set_random_seeds(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)