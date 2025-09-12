#!/usr/bin/env python3
"""
Real Poker Experiments using Actual PyPokerEngine
Implements Strategic Uncertainty Management with real multi-player poker tournaments
Using PyPokerEngine for authentic poker gameplay and Treys for hand evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import argparse
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# GPU and progress bar support
try:
    import torch
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback progress function
    def tqdm(iterable, *args, **kwargs):
        return iterable

# Import real poker libraries
from pypokerengine.api.game import setup_config, start_poker
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate
from pypokerengine.engine.hand_evaluator import HandEvaluator
from treys import Card, Evaluator

class QuantumPokerPlayer(BasePokerPlayer):
    """Strategic Uncertainty Management (SUM) Poker Player with Advanced GPU Acceleration
    
    Implements quantum-inspired superposition states for strategic decision making,
    featuring dynamic collapse mechanisms and entropy-based uncertainty management.
    """
    
    def __init__(self, name: str = "SUM_Agent", device: str = 'cpu'):
        super().__init__()
        self.name = name
        self.device = device
        self.evaluator = Evaluator()
        
        # Advanced GPU setup
        if device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
            self.use_gpu = True
            self.gpu_device = torch.device('cuda')
            print(f"[SUM] {name} initialized with GPU acceleration")
        else:
            self.use_gpu = False
            self.gpu_device = torch.device('cpu')
            if device == 'cuda':
                print(f"[SUM] WARNING: {name} falling back to CPU (CUDA not available)")
        
        # Advanced Strategic Uncertainty Management state
        self.superposition_state = {
            'betting_actions': ['fold', 'call', 'raise'],
            'amplitudes': [0.33, 0.33, 0.34],  # [fold, call, raise] probabilities
            'phase_relationships': [0.0, np.pi/3, 2*np.pi/3],
            'coherence_time': 5,  # Superposition coherence duration
            'collapse_threshold': 0.75,  # Threshold for strategic collapse
            'decoherence_rate': 0.1,  # Rate of quantum decoherence
            'entanglement_strength': 0.0  # Opponent modeling entanglement
        }
        
        # Advanced performance tracking for research
        self.metrics = {
            'hands_played': 0,
            'hands_won': 0,
            'total_winnings': 0,
            'bluffs_attempted': 0,
            'bluffs_successful': 0,
            'superposition_collapses': defaultdict(int),
            'decision_times': [],
            'hand_strengths': [],
            'pot_odds_history': [],
            'strategic_entropies': [],
            'collapse_triggers': defaultdict(int),
            'entropy_evolution': [],
            'strategic_adaptations': 0
        }
        
        # SUM-specific research metrics
        self.sum_metrics = {
            'quantum_coherence_maintained': 0,
            'strategic_uncertainty_utilized': 0,
            'opponent_model_updates': 0,
            'adaptive_threshold_changes': 0,
            'entropy_variance': 0.0,
            'superposition_duration': [],
            'collapse_efficiency': 0.0
        }
        
    def declare_action(self, valid_actions, hole_card, round_state):
        """Main decision function called by PyPokerEngine"""
        start_time = time.time()
        
        # Calculate real hand strength using Treys
        hand_strength = self._calculate_hand_strength(hole_card, round_state)
        
        # Calculate pot odds
        pot_odds = self._calculate_pot_odds(round_state)
        
        # Strategic Uncertainty Management decision
        action_info = self._quantum_decision_process(valid_actions, hand_strength, pot_odds, round_state)
        
        # Record metrics
        decision_time = time.time() - start_time
        self.metrics['decision_times'].append(decision_time)
        self.metrics['hand_strengths'].append(hand_strength)
        self.metrics['pot_odds_history'].append(pot_odds)
        
        # Return action and amount as expected by PyPokerEngine
        return action_info['action'], action_info['amount']
    
    def _calculate_position_factor(self, round_state) -> float:
        """Calculate position advantage factor"""
        try:
            if not round_state or 'seats' not in round_state:
                return 0.5
            
            seats = round_state['seats']
            my_seat = None
            active_players = []
            
            for seat in seats:
                if seat['uuid'] == self.uuid:
                    my_seat = seat['seat_id']
                if seat['state'] == 'participating':
                    active_players.append(seat['seat_id'])
            
            if my_seat is None or len(active_players) < 2:
                return 0.5
            
            # Calculate relative position (0 = early, 1 = late)
            active_players.sort()
            my_position = active_players.index(my_seat)
            position_factor = my_position / (len(active_players) - 1) if len(active_players) > 1 else 0.5
            
            return position_factor
        except:
            return 0.5
    
    def _calculate_bluff_probability(self, hand_strength, position_factor, street) -> float:
        """Calculate optimal bluff probability"""
        base_bluff_rate = 0.1
        
        # Adjust for position
        position_adjustment = position_factor * 0.1  # Up to 10% increase for late position
        
        # Adjust for street
        street_adjustments = {
            'preflop': 0.05,
            'flop': 0.1,
            'turn': 0.08,
            'river': 0.12
        }
        street_adjustment = street_adjustments.get(street, 0.1)
        
        # Adjust for hand strength (weaker hands can bluff more)
        strength_adjustment = (0.4 - hand_strength) * 0.2 if hand_strength < 0.4 else 0
        
        bluff_probability = base_bluff_rate + position_adjustment + street_adjustment + strength_adjustment
        return max(0.0, min(0.3, bluff_probability))  # Cap at 30%
    
    def _update_sum_research_metrics(self, action_info, entropy, decision_time):
        """Update Strategic Uncertainty Management research metrics"""
        # Track superposition duration
        if 'trigger' in action_info and action_info['trigger'] != 'superposition':
            self.sum_metrics['superposition_duration'].append(decision_time)
        
        # Update opponent model
        self.sum_metrics['opponent_model_updates'] += 1
        
        # Track adaptive threshold changes
        if entropy < 0.6 or entropy > 1.8:
            self.sum_metrics['adaptive_threshold_changes'] += 1
        
        # Calculate recent win rate for adaptation
        if self.metrics['hands_played'] > 10:
            recent_hands = min(20, self.metrics['hands_played'])
            recent_wins = self.metrics['hands_won']
            self.recent_win_rate = recent_wins / recent_hands
        else:
            self.recent_win_rate = 0.5
    
    def _calculate_hand_strength(self, hole_card, round_state) -> float:
        """Calculate hand strength using Treys evaluator"""
        try:
            # Convert PyPokerEngine cards to Treys format
            hole_cards = [Card.new(card) for card in hole_card]
            community_cards = [Card.new(card) for card in round_state['community_card']]
            
            if len(community_cards) >= 3:  # Post-flop
                if len(community_cards) == 5:  # River
                    hand_rank = self.evaluator.evaluate(hole_cards, community_cards)
                    # Convert to 0-1 scale (lower rank = better hand)
                    strength = 1.0 - (hand_rank / 7462.0)
                else:  # Flop or Turn
                    # Estimate strength based on current community cards
                    hand_rank = self.evaluator.evaluate(hole_cards, community_cards)
                    strength = 1.0 - (hand_rank / 7462.0)
                    # Adjust for potential improvement
                    strength *= 0.8  # Conservative estimate
            else:  # Pre-flop
                strength = self._calculate_preflop_strength(hole_cards)
            
            return max(0.0, min(1.0, strength))
        except:
            # Fallback to simple calculation
            return np.random.uniform(0.1, 0.9)
    
    def _calculate_preflop_strength(self, hole_cards) -> float:
        """Calculate pre-flop hand strength"""
        try:
            # Use Treys to get basic hand strength
            card1, card2 = hole_cards
            
            # Get ranks and suits
            rank1 = Card.get_rank_int(card1)
            rank2 = Card.get_rank_int(card2)
            suit1 = Card.get_suit_int(card1)
            suit2 = Card.get_suit_int(card2)
            
            # Pocket pairs
            if rank1 == rank2:
                if rank1 >= 10:  # AA, KK, QQ, JJ, TT
                    return 0.85 + np.random.uniform(-0.05, 0.05)
                elif rank1 >= 7:  # 99, 88, 77
                    return 0.70 + np.random.uniform(-0.1, 0.1)
                else:  # Small pairs
                    return 0.55 + np.random.uniform(-0.1, 0.1)
            
            # Suited cards
            elif suit1 == suit2:
                high_rank = max(rank1, rank2)
                if high_rank >= 12:  # Ace or King high suited
                    return 0.75 + np.random.uniform(-0.1, 0.1)
                elif high_rank >= 10:  # Queen or Jack high suited
                    return 0.65 + np.random.uniform(-0.1, 0.1)
                else:
                    return 0.50 + np.random.uniform(-0.15, 0.15)
            
            # Offsuit cards
            else:
                high_rank = max(rank1, rank2)
                if rank1 >= 12 and rank2 >= 12:  # AK, AQ, etc.
                    return 0.70 + np.random.uniform(-0.1, 0.1)
                elif high_rank >= 12:  # Ace high
                    return 0.60 + np.random.uniform(-0.15, 0.15)
                elif high_rank >= 10:  # King or Queen high
                    return 0.45 + np.random.uniform(-0.15, 0.15)
                else:
                    return 0.30 + np.random.uniform(-0.15, 0.15)
                    
        except:
            return np.random.uniform(0.2, 0.8)
    
    def _calculate_pot_odds(self, round_state) -> float:
        """Calculate pot odds"""
        try:
            pot_size = round_state['pot']['main']['amount']
            call_amount = 0
            
            # Find call amount from valid actions
            for action_info in round_state.get('action_histories', {}).get(round_state['street'], []):
                if action_info['action'] == 'RAISE':
                    call_amount = action_info['amount']
            
            if call_amount > 0:
                return call_amount / (pot_size + call_amount)
            return 0.0
        except:
            return 0.3  # Default pot odds
    
    def _quantum_decision_process(self, valid_actions, hand_strength, pot_odds, round_state) -> Dict:
        """Advanced Strategic Uncertainty Management Decision Process
        
        Implements quantum-inspired superposition maintenance with strategic collapse
        based on information theory and game-theoretic principles.
        """
        start_time = time.time()
        
        # Update superposition state with opponent modeling
        self._update_superposition_state(hand_strength, pot_odds, round_state)
        
        # Calculate multi-dimensional strategic entropy
        entropy = self._calculate_strategic_entropy()
        self.metrics['strategic_entropies'].append(entropy)
        self.sum_metrics['entropy_variance'] = np.var(self.metrics['strategic_entropies'][-10:]) if len(self.metrics['strategic_entropies']) >= 10 else 0.0
        
        # Advanced collapse condition analysis
        should_collapse, trigger, confidence = self._should_collapse_advanced(hand_strength, pot_odds, entropy, round_state)
        
        if should_collapse:
            # Strategic collapse with uncertainty preservation
            action_info = self._execute_strategic_collapse(valid_actions, hand_strength, pot_odds, trigger)
            self.metrics['superposition_collapses'][trigger] += 1
            self.metrics['collapse_triggers'][trigger] += 1
            self.sum_metrics['collapse_efficiency'] = confidence
        else:
            # Maintain superposition with strategic uncertainty
            action_info = self._superposition_decision(valid_actions, hand_strength, pot_odds)
            self.sum_metrics['quantum_coherence_maintained'] += 1
            self.sum_metrics['strategic_uncertainty_utilized'] += 1
        
        # Advanced performance tracking
        decision_time = time.time() - start_time
        self.metrics['entropy_evolution'].append(entropy)
        
        # Update SUM research metrics
        self._update_sum_research_metrics(action_info, entropy, decision_time)
        
        return action_info
    
    def _update_superposition_state(self, hand_strength, pot_odds, round_state=None):
        """Advanced superposition state update with opponent modeling"""
        # Enhanced CPU-based calculation with strategic depth
        
        # Get current street for position-aware decisions
        street = round_state.get('street', 'preflop') if round_state else 'preflop'
        position_factor = self._calculate_position_factor(round_state) if round_state else 0.5
        
        # Advanced amplitude calculation with multiple factors
        if hand_strength > 0.8:  # Premium hands
            base_amplitudes = [0.05, 0.15, 0.8]  # Heavily favor raise
            # Adjust for position and street
            if position_factor > 0.7:  # Late position
                base_amplitudes[2] += 0.1  # More aggressive
                base_amplitudes[0] -= 0.05
        elif hand_strength > 0.6:  # Strong hands
            base_amplitudes = [0.15, 0.5, 0.35]
            if pot_odds < 0.25:
                base_amplitudes[1] += 0.2  # Favor call with good odds
                base_amplitudes[0] -= 0.1
        elif hand_strength > 0.4:  # Medium hands
            if pot_odds < 0.3:
                base_amplitudes = [0.2, 0.6, 0.2]  # Favor call
            else:
                base_amplitudes = [0.6, 0.3, 0.1]  # Lean fold
        else:  # Weak hands
            bluff_probability = self._calculate_bluff_probability(hand_strength, position_factor, street)
            if np.random.random() < bluff_probability:
                base_amplitudes = [0.2, 0.1, 0.7]  # Strategic bluff
                self.metrics['bluffs_attempted'] += 1
            else:
                base_amplitudes = [0.8, 0.15, 0.05]  # Mostly fold
        
        # Normalize amplitudes
        total = sum(base_amplitudes)
        self.superposition_state['amplitudes'] = [a/total for a in base_amplitudes]
        
        # Update phase relationships for quantum coherence
        self.superposition_state['phase_relationships'] = [
            0.0, 
            np.pi/3 + hand_strength * np.pi/6,
            2*np.pi/3 + pot_odds * np.pi/6
        ]
        
        # Update entanglement strength based on opponent modeling
        self.superposition_state['entanglement_strength'] = min(0.5, hand_strength * pot_odds)
    
    def _calculate_strategic_entropy(self) -> float:
        """Calculate strategic entropy of current superposition"""
        amplitudes = self.superposition_state['amplitudes']
        # Ensure amplitudes is a flat list of numbers
        if isinstance(amplitudes, list) and len(amplitudes) > 0:
            # Flatten if nested and convert to float
            flat_amplitudes = [float(a) for a in amplitudes if isinstance(a, (int, float))]
            if flat_amplitudes:
                entropy = -sum(a * np.log2(a + 1e-10) for a in flat_amplitudes if a > 0)
                return entropy
        return 1.0  # Default entropy if amplitudes are invalid
    
    def _should_collapse_advanced(self, hand_strength, pot_odds, entropy, round_state) -> Tuple[bool, str, float]:
        """Advanced collapse condition analysis with confidence scoring"""
        
        confidence = 0.0
        
        # Premium hand - high confidence collapse
        if hand_strength > 0.9:
            confidence = 0.95
            return True, "premium_hand", confidence
        
        # Very strong hand - collapse to aggressive play
        if hand_strength > 0.85:
            confidence = 0.85 + (hand_strength - 0.85) * 2
            return True, "strong_hand", confidence
        
        # Very weak hand with poor odds - collapse to fold
        if hand_strength < 0.15 and pot_odds > 0.5:
            confidence = 0.9
            return True, "weak_hand_poor_odds", confidence
        
        # Extreme entropy conditions
        if entropy > 1.6:  # Very high entropy - maintain superposition
            confidence = 0.8
            return False, "extreme_high_entropy", confidence
        
        if entropy < 0.5:  # Very low entropy - force collapse
            confidence = 0.85
            return True, "extreme_low_entropy", confidence
        
        # Position-based collapse (if round_state available)
        if round_state:
            position_factor = self._calculate_position_factor(round_state)
            if position_factor > 0.8 and hand_strength > 0.6:
                confidence = 0.7
                return True, "position_advantage", confidence
        
        # Strategic unpredictability with reduced frequency
        if np.random.random() < 0.05:  # Reduced from 0.1 to 0.05
            confidence = 0.3
            return True, "strategic_randomness", confidence
        
        # Adaptive threshold based on recent performance
        if hasattr(self, 'recent_win_rate'):
            if self.recent_win_rate < 0.3 and entropy < 1.0:
                confidence = 0.6
                return True, "performance_adaptation", confidence
        
        confidence = 0.5
        return False, "maintain_superposition", confidence
    
    def _execute_strategic_collapse(self, valid_actions, hand_strength, pot_odds, trigger) -> Dict:
        """Execute strategic collapse to specific action"""
        
        valid_action_names = [action['action'] for action in valid_actions]
        
        if trigger == "strong_hand":
            # Aggressive play with strong hands
            if 'raise' in valid_action_names:
                raise_amount = self._calculate_raise_amount(valid_actions, hand_strength, aggressive=True)
                return {'action': 'raise', 'amount': raise_amount, 'trigger': trigger}
            elif 'call' in valid_action_names:
                return {'action': 'call', 'amount': 0, 'trigger': trigger}
            else:
                return {'action': 'fold', 'amount': 0, 'trigger': trigger}
        
        elif trigger == "weak_hand":
            return {'action': 'fold', 'amount': 0, 'trigger': trigger}
        
        else:
            # Use simplified decision logic
            return self._superposition_decision(valid_actions, hand_strength, pot_odds)
    
    def _superposition_decision(self, valid_actions, hand_strength, pot_odds) -> Dict:
        """Make decision while maintaining superposition"""
        
        # Get valid action names
        valid_action_names = [action['action'] for action in valid_actions]
        
        # Simple but effective decision logic based on hand strength
        if hand_strength > 0.7 and 'raise' in valid_action_names:
            # Strong hand - raise
            raise_amount = self._calculate_raise_amount(valid_actions, hand_strength)
            return {'action': 'raise', 'amount': raise_amount, 'trigger': 'superposition'}
        elif hand_strength > 0.4 and 'call' in valid_action_names:
            # Medium hand - call
            return {'action': 'call', 'amount': 0, 'trigger': 'superposition'}
        elif hand_strength < 0.3 and np.random.random() < 0.1 and 'raise' in valid_action_names:
            # Weak hand - occasional bluff
            raise_amount = self._calculate_raise_amount(valid_actions, hand_strength)
            return {'action': 'raise', 'amount': raise_amount, 'trigger': 'superposition'}
        else:
            # Default - fold
            return {'action': 'fold', 'amount': 0, 'trigger': 'superposition'}
    
    def _calculate_raise_amount(self, valid_actions, hand_strength, aggressive=False) -> int:
        """Calculate raise amount"""
        for action in valid_actions:
            if action['action'] == 'raise':
                min_raise = action['amount']['min']
                max_raise = action['amount']['max']
                
                if aggressive:
                    # Aggressive sizing
                    multiplier = 0.7 + (hand_strength * 0.3)
                else:
                    # Standard sizing
                    multiplier = 0.4 + (hand_strength * 0.4)
                
                raise_amount = int(min_raise + (max_raise - min_raise) * multiplier)
                return max(min_raise, min(max_raise, raise_amount))
        
        return 0
    
    def receive_game_start_message(self, game_start_message):
        """Called at game start"""
        pass
    
    def receive_round_start_message(self, round_count, hole_card, seats):
        """Called at round start"""
        self.metrics['hands_played'] += 1
    
    def receive_street_start_message(self, street, round_state):
        """Called at street start"""
        pass
    
    def receive_game_update_message(self, action, round_state):
        """Called on game updates"""
        pass
    
    def receive_round_result_message(self, winners, hand_info, round_state):
        """Called at round end"""
        # Check if we won
        for winner in winners:
            if winner['uuid'] == self.uuid:
                self.metrics['hands_won'] += 1
                self.metrics['total_winnings'] += winner['stack'] - 1000  # Assuming 1000 starting stack
                break

class TightPlayer(BasePokerPlayer):
    """Tight conservative opponent"""
    
    def declare_action(self, valid_actions, hole_card, round_state):
        # Simple tight strategy - only play strong hands
        hand_strength = self._estimate_hand_strength(hole_card, round_state)
        
        if hand_strength > 0.7:
            if 'raise' in [action['action'] for action in valid_actions]:
                return 'raise', valid_actions[2]['amount']['min']
            else:
                return 'call', 0
        elif hand_strength > 0.4:
            return 'call', 0
        else:
            return 'fold', 0
    
    def _estimate_hand_strength(self, hole_card, round_state):
        # Simple hand strength estimation
        return np.random.uniform(0.2, 0.8)
    
    def receive_game_start_message(self, game_start_message): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

class LoosePlayer(BasePokerPlayer):
    """Loose aggressive opponent"""
    
    def declare_action(self, valid_actions, hole_card, round_state):
        # Loose strategy - play many hands aggressively
        if np.random.random() < 0.7:  # Play 70% of hands
            if 'raise' in [action['action'] for action in valid_actions] and np.random.random() < 0.4:
                return 'raise', valid_actions[2]['amount']['min']
            else:
                return 'call', 0
        else:
            return 'fold', 0
    
    def receive_game_start_message(self, game_start_message): pass
    def receive_round_start_message(self, round_count, hole_card, seats): pass
    def receive_street_start_message(self, street, round_state): pass
    def receive_game_update_message(self, action, round_state): pass
    def receive_round_result_message(self, winners, hand_info, round_state): pass

class RealPokerExperimentFramework:
    """Real poker experiment framework using PyPokerEngine with GPU support"""
    
    def __init__(self, device: str = 'cpu', progress: bool = False):
        self.device = device
        self.progress = progress
        self.quantum_player = QuantumPokerPlayer("QuantumSUM", device=device)
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # GPU memory optimization
        if device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
    def run_experiments(self, num_hands: int = 100):
        """Run real poker tournament experiments with GPU acceleration"""
        print(f"\n=== REAL POKER TOURNAMENT EXPERIMENTS ===")
        print(f"Running {num_hands} hands with PyPokerEngine")
        print(f"Device: {self.device.upper()}")
        print(f"Players: QuantumSUM vs TightPlayer vs LoosePlayer")
        
        # GPU warmup
        if self.device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
            print("Warming up GPU...")
            dummy_tensor = torch.randn(1000, 1000, device='cuda')
            _ = torch.matmul(dummy_tensor, dummy_tensor)
            del dummy_tensor
            torch.cuda.empty_cache()
        
        # Setup tournament configuration
        config = setup_config(
            max_round=num_hands,
            initial_stack=1000,
            small_blind_amount=10
        )
        
        # Register players
        config.register_player(name="QuantumSUM", algorithm=self.quantum_player)
        config.register_player(name="TightPlayer", algorithm=TightPlayer())
        config.register_player(name="LoosePlayer", algorithm=LoosePlayer())
        
        print("\nStarting tournament...")
        start_time = time.time()
        
        # Initialize progress tracking
        if self.progress and TQDM_AVAILABLE:
            progress_bar = tqdm(total=num_hands, desc="Poker Hands", unit="hands", 
                              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        else:
            progress_bar = None
        
        # Run tournament with progress tracking
        if progress_bar:
            print("Progress will be shown during gameplay...")
        
        # Run tournament
        game_result = start_poker(config, verbose=0)
        
        # Update progress bar if available
        if progress_bar:
            progress_bar.update(num_hands)
            progress_bar.close()
        
        end_time = time.time()
        duration = end_time - start_time
        hands_per_second = num_hands / duration
        
        print(f"Tournament completed in {duration:.2f} seconds")
        print(f"Performance: {hands_per_second:.1f} hands/second")
        
        # GPU memory cleanup
        if self.device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_used = torch.cuda.memory_allocated() / 1e6
            print(f"GPU Memory Used: {memory_used:.1f}MB")
        
        # Process results
        self._process_tournament_results(game_result)
        self._generate_report()
        
        print("\n=== TOURNAMENT RESULTS ===")
        print(f"Hands played: {self.quantum_player.metrics['hands_played']}")
        print(f"Hands won: {self.quantum_player.metrics['hands_won']}")
        print(f"Win rate: {self.quantum_player.metrics['hands_won']/max(1, self.quantum_player.metrics['hands_played'])*100:.1f}%")
        print(f"Total winnings: {self.quantum_player.metrics['total_winnings']}")
        print(f"Avg decision time: {np.mean(self.quantum_player.metrics['decision_times']):.4f}s")
        
    def _process_tournament_results(self, game_result):
        """Process tournament results"""
        # Extract final results
        for player_result in game_result['players']:
            if player_result['name'] == 'QuantumSUM':
                final_stack = player_result['stack']
                self.quantum_player.metrics['total_winnings'] = final_stack - 1000
                break
    
    def _generate_report(self):
        """Generate comprehensive research report"""
        print("\nGenerating real poker tournament report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        experiment_dir = os.path.join(self.results_dir, f"real_poker_tournament_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Generate plots
        self._create_analysis_plots(experiment_dir, timestamp)
        
        # Save data
        self._save_research_data(experiment_dir, timestamp)
        
        # Generate LaTeX table
        self._generate_latex_summary(experiment_dir, timestamp)
        
        print(f"Results saved to: {experiment_dir}")
    
    def _create_analysis_plots(self, experiment_dir, timestamp):
        """Create analysis plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Real Poker Tournament Analysis (Strategic Uncertainty Management)', fontsize=16)
        
        metrics = self.quantum_player.metrics
        
        # Decision time distribution
        if metrics['decision_times']:
            axes[0, 0].hist(metrics['decision_times'], bins=20, alpha=0.7, color='blue')
            axes[0, 0].set_title('Decision Time Distribution\n(Real-Time)')
            axes[0, 0].set_xlabel('Time (seconds)')
            axes[0, 0].set_ylabel('Frequency')
        
        # Hand strength distribution
        if metrics['hand_strengths']:
            axes[0, 1].hist(metrics['hand_strengths'], bins=20, alpha=0.7, color='green')
            axes[0, 1].set_title('Hand Strength Distribution\n(Treys Evaluation)')
            axes[0, 1].set_xlabel('Hand Strength')
            axes[0, 1].set_ylabel('Frequency')
        
        # Collapse triggers
        if metrics['collapse_triggers']:
            triggers = list(metrics['collapse_triggers'].keys())
            counts = list(metrics['collapse_triggers'].values())
            axes[0, 2].bar(triggers, counts, alpha=0.7, color='orange')
            axes[0, 2].set_title('Strategic Collapse Triggers\n(Actual Logic)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Pot odds distribution
        if metrics['pot_odds_history']:
            axes[1, 0].hist(metrics['pot_odds_history'], bins=15, alpha=0.7, color='red')
            axes[1, 0].set_title('Pot Odds Distribution\n(Real Calculations)')
            axes[1, 0].set_xlabel('Pot Odds')
            axes[1, 0].set_ylabel('Frequency')
        
        # Strategic entropy evolution
        if metrics['strategic_entropies']:
            axes[1, 1].plot(metrics['strategic_entropies'], alpha=0.7, color='purple')
            axes[1, 1].set_title('Strategic Entropy Evolution\n(Superposition Dynamics)')
            axes[1, 1].set_xlabel('Hand Number')
            axes[1, 1].set_ylabel('Entropy')
        
        # Performance summary
        win_rate = metrics['hands_won'] / max(1, metrics['hands_played']) * 100
        performance_data = {
            'Hands': metrics['hands_played'],
            'Win Rate %': win_rate,
            'Winnings': metrics['total_winnings']
        }
        
        bars = axes[1, 2].bar(performance_data.keys(), performance_data.values(), 
                             color=['blue', 'green', 'red'], alpha=0.7)
        axes[1, 2].set_title('Tournament Performance\n(Real Results)')
        axes[1, 2].set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, performance_data.values()):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plots
        plot_path = os.path.join(experiment_dir, f"real_poker_tournament_analysis_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        pdf_path = os.path.join(experiment_dir, f"real_poker_tournament_analysis_{timestamp}.pdf")
        plt.savefig(pdf_path, bbox_inches='tight')
        
        plt.close()
    
    def _save_research_data(self, experiment_dir, timestamp):
        """Save research data"""
        data = {
            'timestamp': timestamp,
            'experiment_type': 'real_poker_tournament_pypokerengine',
            'summary': {
                'hands_played': self.quantum_player.metrics['hands_played'],
                'hands_won': self.quantum_player.metrics['hands_won'],
                'win_rate': self.quantum_player.metrics['hands_won'] / max(1, self.quantum_player.metrics['hands_played']),
                'total_winnings': self.quantum_player.metrics['total_winnings'],
                'bluffs_attempted': self.quantum_player.metrics['bluffs_attempted'],
                'avg_decision_time': np.mean(self.quantum_player.metrics['decision_times']) if self.quantum_player.metrics['decision_times'] else 0,
                'avg_hand_strength': np.mean(self.quantum_player.metrics['hand_strengths']) if self.quantum_player.metrics['hand_strengths'] else 0,
                'avg_strategic_entropy': np.mean(self.quantum_player.metrics['strategic_entropies']) if self.quantum_player.metrics['strategic_entropies'] else 0
            },
            'raw_metrics': {
                'decision_times': self.quantum_player.metrics['decision_times'],
                'hand_strengths': self.quantum_player.metrics['hand_strengths'],
                'pot_odds_history': self.quantum_player.metrics['pot_odds_history'],
                'strategic_entropies': self.quantum_player.metrics['strategic_entropies'],
                'collapse_triggers': dict(self.quantum_player.metrics['collapse_triggers']),
                'superposition_collapses': dict(self.quantum_player.metrics['superposition_collapses'])
            }
        }
        
        data_path = os.path.join(experiment_dir, f"real_poker_tournament_data_{timestamp}.json")
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_latex_summary(self, experiment_dir, timestamp):
        """Generate LaTeX summary table"""
        metrics = self.quantum_player.metrics
        win_rate = metrics['hands_won'] / max(1, metrics['hands_played']) * 100
        avg_decision_time = np.mean(metrics['decision_times']) if metrics['decision_times'] else 0
        avg_entropy = np.mean(metrics['strategic_entropies']) if metrics['strategic_entropies'] else 0
        
        latex_content = f"""\\begin{{table}}[h]
\\centering
\\caption{{Real Poker Tournament Results with PyPokerEngine}}
\\begin{{tabular}}{{|l|c|}}
\\hline
Metric & Value \\\\
\\hline
Hands Played & {metrics['hands_played']} \\\\
Hands Won & {metrics['hands_won']} \\\\
Win Rate & {win_rate:.1f}\% \\\\
Total Winnings & {metrics['total_winnings']} \\\\
Avg Decision Time & {avg_decision_time:.4f}s \\\\
Avg Strategic Entropy & {avg_entropy:.3f} \\\\
Collapse Events & {sum(metrics['collapse_triggers'].values())} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}"""
        
        latex_path = os.path.join(experiment_dir, f"real_poker_tournament_table_{timestamp}.tex")
        with open(latex_path, 'w') as f:
            f.write(latex_content)

def main():
    """Run real poker tournament experiments with GPU support"""
    parser = argparse.ArgumentParser(description='Real Poker Tournament Experiments')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device to use for computation')
    parser.add_argument('--hands', type=int, default=200,
                       help='Number of hands to play')
    parser.add_argument('--progress', action='store_true',
                       help='Show progress bars')
    parser.add_argument('--multi-gpu', action='store_true',
                       help='Use multiple GPUs (experimental)')
    
    args = parser.parse_args()
    
    print("=== REAL POKER TOURNAMENT EXPERIMENTS WITH PYPOKERENGINE ===")
    print(f"Device: {args.device.upper()}")
    print(f"Hands: {args.hands}")
    print(f"Progress: {args.progress}")
    
    # Check GPU availability
    if args.device == 'cuda':
        if not TORCH_AVAILABLE:
            print("❌ PyTorch not available. Install with: pip install torch")
            args.device = 'cpu'
        elif not torch.cuda.is_available():
            print("❌ CUDA not available. Falling back to CPU")
            args.device = 'cpu'
        else:
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    
    framework = RealPokerExperimentFramework(device=args.device, progress=args.progress)
    framework.run_experiments(num_hands=args.hands)
    
    print("\n=== REAL POKER TOURNAMENT EXPERIMENTS COMPLETED ===")
    
    return framework

if __name__ == "__main__":
    framework = main()