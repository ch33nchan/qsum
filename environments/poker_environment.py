import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import logging
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from pypokerengine.engine.hand_evaluator import HandEvaluator
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PokerEnvironment:
    def __init__(self, 
                 initial_stack: int = 200,
                 small_blind: int = 1,
                 big_blind: int = 2,
                 max_round: int = 1000):
        
        self.initial_stack = initial_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.max_round = max_round
        
        self.action_mapping = {
            'fold': 0,
            'call': 1,
            'raise': 2,
            'check': 3
        }
        
        self.reverse_action_mapping = {v: k for k, v in self.action_mapping.items()}
        
        self.game_history = []
        self.current_game_state = None
        
        logger.info(f"PokerEnvironment initialized: stack={initial_stack}BB, blinds={small_blind}/{big_blind}")
    
    def create_game_config(self, players: List[BasePokerPlayer]) -> Dict:
        config = setup_config(
            max_round=self.max_round,
            initial_stack=self.initial_stack,
            small_blind_amount=self.small_blind
        )
        
        for i, player in enumerate(players):
            config.register_player(name=f"player_{i}", algorithm=player)
        
        return config
    
    def encode_game_state(self, round_state: Dict, player_id: int) -> Dict[str, torch.Tensor]:
        batch_size = 1
        
        # Extract hole cards from seats
        seats = round_state.get('seats', [])
        if len(seats) > player_id:
            hole_cards = seats[player_id].get('hole_card', [])
        else:
            hole_cards = []
        
        # Extract community cards
        community_cards = round_state.get('community_card', [])
        
        card_features = self._encode_cards(hole_cards, community_cards)
        
        position = torch.tensor([player_id], dtype=torch.long)
        
        # Extract stack sizes from seats
        stacks = []
        for seat in seats[:2]:  # Only first 2 players for heads-up
            stacks.append(seat.get('stack', 200))
        
        while len(stacks) < 2:
            stacks.append(200)  # Default stack size
            
        stacks_tensor = torch.tensor(stacks, dtype=torch.float32).unsqueeze(0)
        
        # Extract pot size
        pot = round_state.get('pot', {'main': {'amount': 0}})
        pot_size = torch.tensor([pot['main']['amount']], dtype=torch.float32)
        
        # Extract street
        street = round_state.get('street', 'preflop')
        street_encoding = self._encode_street(street)
        
        hand_strength = self._calculate_hand_strength(hole_cards, community_cards)
        
        raw_features = torch.cat([
            street_encoding.unsqueeze(0),
            torch.tensor([hand_strength], dtype=torch.float32).unsqueeze(0),
            torch.tensor([len(community_cards)], dtype=torch.float32).unsqueeze(0)
        ], dim=-1)
        
        return {
            'cards': card_features.unsqueeze(0),
            'position': position.unsqueeze(0),
            'stacks': stacks_tensor,
            'pot': pot_size,
            'raw_features': raw_features,
            'hand_strength': torch.tensor([hand_strength], dtype=torch.float32),
            'pot_size': pot_size,
            'stack_sizes': stacks_tensor.mean(dim=-1)
        }
    
    def encode_action_history(self, action_history: List[Dict]) -> Dict[str, torch.Tensor]:
        max_seq_len = 50
        
        if not action_history:
            return {
                'actions': torch.zeros(1, max_seq_len, dtype=torch.long),
                'amounts': torch.zeros(1, max_seq_len, dtype=torch.float32)
            }
        
        actions = []
        amounts = []
        
        for action in action_history[-max_seq_len:]:
            action_type = action.get('action', 'check')
            action_id = self.action_mapping.get(action_type, 3)
            actions.append(action_id)
            
            amount = action.get('amount', 0)
            amounts.append(float(amount))
        
        while len(actions) < max_seq_len:
            actions.insert(0, 3)
            amounts.insert(0, 0.0)
        
        return {
            'actions': torch.tensor(actions, dtype=torch.long).unsqueeze(0),
            'amounts': torch.tensor(amounts, dtype=torch.float32).unsqueeze(0)
        }
    
    def _encode_cards(self, hole_cards: List, community_cards: List) -> torch.Tensor:
        card_vector = torch.zeros(52, dtype=torch.long)
        
        all_cards = hole_cards + community_cards
        
        for card in all_cards:
            if hasattr(card, 'to_id'):
                card_id = card.to_id()
            else:
                card_id = self._card_to_id(card)
            
            if 0 <= card_id < 52:
                card_vector[card_id] = 1
        
        return card_vector
    
    def _card_to_id(self, card) -> int:
        if hasattr(card, 'suit') and hasattr(card, 'rank'):
            return card.suit * 13 + card.rank - 2
        return 0
    
    def _encode_street(self, street: str) -> torch.Tensor:
        street_mapping = {
            'preflop': [1, 0, 0, 0],
            'flop': [0, 1, 0, 0],
            'turn': [0, 0, 1, 0],
            'river': [0, 0, 0, 1]
        }
        
        encoding = street_mapping.get(street, [0, 0, 0, 0])
        return torch.tensor(encoding, dtype=torch.float32)
    
    def _calculate_hand_strength(self, hole_cards: List, community_cards: List) -> float:
        if not hole_cards:
            return 0.0
        
        try:
            if len(community_cards) >= 3:
                all_cards = hole_cards + community_cards
                hand_strength = HandEvaluator.eval_hand(hole_cards, community_cards)
                normalized_strength = (7462 - hand_strength) / 7462.0
            else:
                win_rate = estimate_hole_card_win_rate(
                    nb_simulation=1000,
                    nb_player=2,
                    hole_card=gen_cards(hole_cards),
                    community_card=gen_cards(community_cards) if community_cards else None
                )
                normalized_strength = win_rate
            
            return float(normalized_strength)
        
        except Exception as e:
            logger.warning(f"Error calculating hand strength: {e}")
            return 0.5
    
    def decode_action(self, action_tensor: torch.Tensor, valid_actions: List[Dict]) -> Dict:
        action_id = action_tensor.item() if torch.is_tensor(action_tensor) else action_tensor
        action_name = self.reverse_action_mapping.get(action_id, 'check')
        
        valid_action_names = [action['action'] for action in valid_actions]
        
        if action_name not in valid_action_names:
            if 'check' in valid_action_names:
                action_name = 'check'
            elif 'call' in valid_action_names:
                action_name = 'call'
            elif 'fold' in valid_action_names:
                action_name = 'fold'
            else:
                action_name = valid_action_names[0]
        
        for valid_action in valid_actions:
            if valid_action['action'] == action_name:
                return valid_action
        
        return valid_actions[0]
    
    def run_tournament(self, 
                      player1: BasePokerPlayer, 
                      player2: BasePokerPlayer, 
                      num_games: int = 1000) -> Dict[str, Any]:
        
        config = self.create_game_config([player1, player2])
        
        start_time = time.time()
        
        try:
            game_result = start_poker(config, verbose=0)
        except Exception as e:
            logger.error(f"Error running tournament: {e}")
            return self._create_error_result()
        
        end_time = time.time()
        
        return self._process_tournament_results(game_result, start_time, end_time, num_games)
    
    def _process_tournament_results(self, game_result: Dict, start_time: float, end_time: float, num_games: int) -> Dict[str, Any]:
        duration = end_time - start_time
        
        player_results = {}
        total_winnings = 0
        
        for player_info in game_result['players']:
            player_name = player_info['name']
            # Handle both dict and int stack formats
            if isinstance(player_info['stack'], dict):
                stack = player_info['stack'].get('amount', self.initial_stack)
            else:
                stack = player_info['stack']
            winnings = stack - self.initial_stack
            
            player_results[player_name] = {
                'final_stack': stack,
                'winnings': winnings,
                'win_rate': self._calculate_win_rate(winnings, num_games)
            }
            
            total_winnings += abs(winnings)
        
        hands_per_second = num_games / duration if duration > 0 else 0
        
        return {
            'success': True,
            'duration': duration,
            'hands_per_second': hands_per_second,
            'total_hands': num_games,
            'player_results': player_results,
            'total_winnings': total_winnings,
            'game_result': game_result
        }
    
    def _calculate_win_rate(self, winnings: int, num_games: int) -> float:
        if num_games == 0:
            return 0.0
        
        mbb_per_100 = (winnings / (num_games / 100)) / self.big_blind * 1000
        
        win_rate = 0.5 + (mbb_per_100 / 2000)
        
        return max(0.0, min(1.0, win_rate))
    
    def _create_error_result(self) -> Dict[str, Any]:
        return {
            'success': False,
            'duration': 0,
            'hands_per_second': 0,
            'total_hands': 0,
            'player_results': {},
            'total_winnings': 0,
            'error': 'Tournament execution failed'
        }
    
    def validate_environment(self) -> bool:
        try:
            from pypokerengine.players import BasePokerPlayer
            
            class DummyPlayer(BasePokerPlayer):
                def declare_action(self, valid_actions, hole_card, round_state):
                    return valid_actions[0]['action'], valid_actions[0]['amount']
                
                def receive_game_start_message(self, game_info):
                    pass
                
                def receive_round_start_message(self, round_count, hole_card, seats):
                    pass
                
                def receive_street_start_message(self, street, round_state):
                    pass
                
                def receive_game_update_message(self, action, round_state):
                    pass
                
                def receive_round_result_message(self, winners, hand_info, round_state):
                    pass
            
            dummy1 = DummyPlayer()
            dummy2 = DummyPlayer()
            
            result = self.run_tournament(dummy1, dummy2, num_games=10)
            
            logger.info(f"Environment validation: {'SUCCESS' if result['success'] else 'FAILED'}")
            return result['success']
            
        except Exception as e:
            logger.error(f"Environment validation failed: {e}")
            return False

class PokerGameLogger:
    def __init__(self, log_file: str = None):
        self.log_file = log_file
        self.game_logs = []
        
    def log_game_state(self, game_state: Dict, player_id: int, action: Dict, timestamp: float):
        log_entry = {
            'timestamp': timestamp,
            'player_id': player_id,
            'game_state': self._serialize_game_state(game_state),
            'action': action
        }
        
        self.game_logs.append(log_entry)
        
        if self.log_file and len(self.game_logs) % 100 == 0:
            self._save_logs()
    
    def _serialize_game_state(self, game_state: Dict) -> Dict:
        try:
            return {
                'street': game_state.get('street', 'unknown'),
                'pot': game_state.get('pot', {}),
                'community_cards': str(game_state.get('community_card', [])),
                'round_count': game_state.get('round_count', 0)
            }
        except Exception as e:
            logger.warning(f"Error serializing game state: {e}")
            return {}
    
    def _save_logs(self):
        if self.log_file:
            try:
                with open(self.log_file, 'w') as f:
                    json.dump(self.game_logs, f, indent=2)
            except Exception as e:
                logger.error(f"Error saving logs: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        if not self.game_logs:
            return {}
        
        total_games = len(self.game_logs)
        
        action_counts = {}
        for log in self.game_logs:
            action = log['action'].get('action', 'unknown')
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            'total_games': total_games,
            'action_distribution': action_counts,
            'average_game_duration': self._calculate_average_duration()
        }
    
    def _calculate_average_duration(self) -> float:
        if len(self.game_logs) < 2:
            return 0.0
        
        durations = []
        for i in range(1, len(self.game_logs)):
            duration = self.game_logs[i]['timestamp'] - self.game_logs[i-1]['timestamp']
            durations.append(duration)
        
        return sum(durations) / len(durations) if durations else 0.0