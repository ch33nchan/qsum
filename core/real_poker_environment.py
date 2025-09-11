#!/usr/bin/env python3
"""
Real Poker Environment using Treys and PyPokerEngine
Implements actual poker game logic with real hand evaluation
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from treys import Card, Evaluator, Deck
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from pypokerengine.engine.hand_evaluator import HandEvaluator
import time

@dataclass
class PokerGameState:
    """Real poker game state"""
    hole_cards: List[str]
    community_cards: List[str]
    pot_size: int
    current_bet: int
    player_stack: int
    opponent_stack: int
    position: str  # 'button' or 'big_blind'
    round_state: str  # 'preflop', 'flop', 'turn', 'river'
    valid_actions: List[str]
    hand_strength: float
    pot_odds: float
    
class RealPokerEnvironment:
    """Real poker environment using actual poker libraries"""
    
    def __init__(self, initial_stack: int = 1000, small_blind: int = 5, big_blind: int = 10):
        self.initial_stack = initial_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.evaluator = Evaluator()
        self.deck = Deck()
        
        # Game state
        self.player_stack = initial_stack
        self.opponent_stack = initial_stack
        self.pot_size = 0
        self.current_bet = 0
        self.position = 'button'  # Alternates each hand
        self.round_state = 'preflop'
        
        # Cards
        self.hole_cards = []
        self.community_cards = []
        self.opponent_hole_cards = []
        
        # Hand history
        self.hand_history = []
        self.current_hand_actions = []
        
    def reset(self) -> PokerGameState:
        """Start a new hand"""
        # Reset deck and deal new cards
        self.deck = Deck()
        
        # Deal hole cards
        self.hole_cards = self.deck.draw(2)
        self.opponent_hole_cards = self.deck.draw(2)
        self.community_cards = []
        
        # Reset betting
        self.pot_size = self.small_blind + self.big_blind
        self.current_bet = self.big_blind
        self.round_state = 'preflop'
        
        # Alternate position
        self.position = 'big_blind' if self.position == 'button' else 'button'
        
        # Deduct blinds
        if self.position == 'button':
            self.player_stack -= self.small_blind
            self.opponent_stack -= self.big_blind
        else:
            self.player_stack -= self.big_blind
            self.opponent_stack -= self.small_blind
        
        self.current_hand_actions = []
        
        return self._get_game_state()
    
    def _calculate_hand_strength(self, hole_cards: List, community_cards: List) -> float:
        """Calculate hand strength using actual poker evaluation"""
        if len(community_cards) < 3:
            return self._calculate_preflop_strength(hole_cards)
        
        # Use Treys evaluator for actual hand strength
        current_hand = hole_cards + community_cards
        current_score = self.evaluator.evaluate(current_hand, community_cards)
        
        # Normalize score (lower is better in Treys, so invert)
        # Treys scores range from 1 (royal flush) to 7462 (high card)
        normalized_strength = 1.0 - (current_score - 1) / 7461
        
        return max(0.0, min(1.0, normalized_strength))
    
    def _calculate_preflop_strength(self, hole_cards: List) -> float:
        """Calculate preflop hand strength"""
        # Convert Treys cards to string format for evaluation
        rank1 = Card.get_rank_int(hole_cards[0])
        rank2 = Card.get_rank_int(hole_cards[1])
        suit1 = Card.get_suit_int(hole_cards[0])
        suit2 = Card.get_suit_int(hole_cards[1])
        
        # Pocket pairs
        if rank1 == rank2:
            if rank1 >= 10:  # TT+
                return 0.9
            elif rank1 >= 7:  # 77-99
                return 0.7
            else:  # 22-66
                return 0.5
        
        # Suited connectors and high cards
        if suit1 == suit2:  # Suited
            if abs(rank1 - rank2) <= 1:  # Suited connectors
                return 0.6
            elif max(rank1, rank2) >= 11:  # Suited with high card
                return 0.7
        
        # High cards
        if max(rank1, rank2) >= 12:  # Ace or King
            if min(rank1, rank2) >= 10:  # AK, AQ, AJ, KQ
                return 0.8
            elif min(rank1, rank2) >= 8:  # A9+, K9+
                return 0.6
        
        return 0.3  # Default for weak hands
    
    def _get_game_state(self) -> PokerGameState:
        """Get current game state"""
        # Calculate hand strength
        hand_strength = self._calculate_hand_strength(self.hole_cards, self.community_cards)
        
        # Calculate pot odds
        pot_odds = self.current_bet / (self.pot_size + self.current_bet) if self.pot_size > 0 else 0
        
        # Determine valid actions
        valid_actions = ['fold', 'call']
        if self.player_stack > self.current_bet:
            valid_actions.append('raise')
        
        return PokerGameState(
            hole_cards=[Card.int_to_str(card) for card in self.hole_cards],
            community_cards=[Card.int_to_str(card) for card in self.community_cards],
            pot_size=self.pot_size,
            current_bet=self.current_bet,
            player_stack=self.player_stack,
            opponent_stack=self.opponent_stack,
            position=self.position,
            round_state=self.round_state,
            valid_actions=valid_actions,
            hand_strength=hand_strength,
            pot_odds=pot_odds
        )

if __name__ == "__main__":
    # Test the real poker environment
    env = RealPokerEnvironment()
    
    print("Testing Real Poker Environment")
    print("==============================")
    
    for hand in range(3):
        print(f"\nHand {hand + 1}:")
        state = env.reset()
        print(f"Hole cards: {state.hole_cards}")
        print(f"Hand strength: {state.hand_strength:.3f}")
        print(f"Community cards: {state.community_cards}")
    
    print("\nReal poker environment test completed!")