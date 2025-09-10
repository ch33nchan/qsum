import numpy as np
from typing import Dict, Tuple, List
from .utils import Deck, HandEvaluator, Action, Card

class PokerEnvironment:
    def __init__(self, starting_stack: int = 200, big_blind: int = 2):
        self.starting_stack = starting_stack
        self.big_blind = big_blind
        self.small_blind = big_blind // 2
        self.deck = Deck()
        self.reset()
    
    def reset(self) -> Dict:
        self.deck.reset()
        self.player_cards = self.deck.deal(2)
        self.opponent_cards = self.deck.deal(2)
        self.community_cards = []
        self.pot = self.small_blind + self.big_blind
        self.player_stack = self.starting_stack - self.big_blind
        self.opponent_stack = self.starting_stack - self.small_blind
        self.street = 0  # 0=preflop, 1=flop, 2=turn, 3=river
        self.done = False
        self.winner = None
        
        return self._get_state()
    
    def _get_state(self) -> Dict:
        return {
            'player_cards': self.player_cards,
            'community_cards': self.community_cards,
            'pot': self.pot,
            'player_stack': self.player_stack,
            'opponent_stack': self.opponent_stack,
            'street': self.street,
            'done': self.done
        }
    
    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        reward = 0
        info = {}
        
        if action == Action.FOLD.value:
            self.done = True
            reward = -self.big_blind
        elif action == Action.CALL.value:
            call_amount = min(self.big_blind, self.player_stack)
            self.pot += call_amount
            self.player_stack -= call_amount
        elif action == Action.RAISE.value:
            raise_amount = min(self.big_blind * 2, self.player_stack)
            self.pot += raise_amount
            self.player_stack -= raise_amount
        
        if not self.done:
            opponent_action = self._get_opponent_action()
            info['opponent_action'] = opponent_action
            
            if opponent_action == 'fold':
                self.done = True
                reward = self.pot - self.big_blind
            elif opponent_action == 'call':
                self._advance_street()
            elif opponent_action == 'raise':
                # Simplified opponent raise handling
                pass
        
        if self.street > 3 and not self.done:
            self._showdown()
            reward = self._calculate_reward()
        
        return self._get_state(), reward, self.done, info
    
    def _get_opponent_action(self) -> str:
        hand_strength = HandEvaluator.evaluate_hand(self.opponent_cards, self.community_cards)
        
        if hand_strength > 0.7:
            return np.random.choice(['call', 'raise'], p=[0.3, 0.7])
        elif hand_strength > 0.4:
            return np.random.choice(['call', 'fold'], p=[0.7, 0.3])
        else:
            return np.random.choice(['fold', 'call'], p=[0.8, 0.2])
    
    def _advance_street(self):
        self.street += 1
        if self.street == 1:  # Flop
            self.community_cards.extend(self.deck.deal(3))
        elif self.street in [2, 3]:  # Turn, River
            self.community_cards.extend(self.deck.deal(1))
    
    def _showdown(self):
        player_strength = HandEvaluator.evaluate_hand(self.player_cards, self.community_cards)
        opponent_strength = HandEvaluator.evaluate_hand(self.opponent_cards, self.community_cards)
        
        if player_strength > opponent_strength:
            self.winner = 'player'
        elif opponent_strength > player_strength:
            self.winner = 'opponent'
        else:
            self.winner = 'tie'
        
        self.done = True
    
    def _calculate_reward(self) -> float:
        if hasattr(self, 'winner'):
            if self.winner == 'player':
                return self.pot - self.big_blind
            elif self.winner == 'opponent':
                return -self.big_blind
            else:
                return 0
        return 0