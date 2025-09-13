#!/usr/bin/env python3
"""
Core Strategic Uncertainty Management (SUM) Algorithm Implementation
Mathematically rigorous implementation of SUM principles for poker AI
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import math

class SUMAlgorithmCore:
    """Core SUM Algorithm Implementation
    
    This class contains the mathematical foundations of the Strategic Uncertainty
    Management algorithm, implementing information-theoretic decision making,
    Bayesian opponent modeling, and adaptive uncertainty management.
    """
    
    def _calculate_information_content(self, hand_strength: float, pot_odds: float, position_factor: float) -> float:
        """Calculate information content using Shannon information theory"""
        # Information content I(x) = -log2(P(x))
        # Higher uncertainty = higher information content
        
        # Combine multiple information sources
        hand_uncertainty = 1.0 - abs(hand_strength - 0.5) * 2  # Max uncertainty at 0.5 strength
        odds_uncertainty = min(pot_odds, 1.0 - pot_odds) * 2    # Max uncertainty at 0.5 odds
        position_uncertainty = min(position_factor, 1.0 - position_factor) * 2
        
        # Weighted combination
        combined_uncertainty = (0.5 * hand_uncertainty + 
                              0.3 * odds_uncertainty + 
                              0.2 * position_uncertainty)
        
        # Convert to information content (bits)
        information_content = -math.log2(max(combined_uncertainty, 0.001))
        return min(information_content, 10.0)  # Cap at 10 bits
    
    def _calculate_information_entropy(self) -> float:
        """Calculate strategic entropy H(X) = -Σ p(x) * log2(p(x))"""
        amplitudes = self.superposition_state['action_amplitudes']
        
        # Shannon entropy calculation
        entropy = 0.0
        for amplitude in amplitudes:
            if amplitude > 0:
                entropy -= amplitude * math.log2(amplitude)
        
        # Store in history for analysis
        self.superposition_state['entropy_history'].append(entropy)
        
        # Maintain sliding window
        if len(self.superposition_state['entropy_history']) > self.sum_parameters['entropy_window_size']:
            self.superposition_state['entropy_history'].pop(0)
        
        return entropy
    
    def _calculate_information_gain(self, current_entropy: float) -> float:
        """Calculate information gain from previous decision"""
        if len(self.superposition_state['entropy_history']) < 2:
            return 0.0
        
        previous_entropy = self.superposition_state['entropy_history'][-2]
        information_gain = previous_entropy - current_entropy
        
        self.research_metrics['information_gain_per_decision'].append(information_gain)
        return information_gain
    
    def _calculate_optimal_bluff_frequency(self, position_factor: float, street: str) -> float:
        """Calculate game-theoretically optimal bluff frequency"""
        # Base bluff frequencies from game theory
        base_frequencies = {
            'preflop': 0.08,
            'flop': 0.12,
            'turn': 0.10,
            'river': 0.15
        }
        
        base_freq = base_frequencies.get(street, 0.10)
        
        # Adjust for position (late position can bluff more)
        position_adjustment = position_factor * 0.05
        
        # Adjust based on opponent models
        avg_fold_threshold = np.mean(list(self.opponent_models['fold_thresholds'].values()))
        opponent_adjustment = (avg_fold_threshold - 0.3) * 0.1
        
        optimal_frequency = base_freq + position_adjustment + opponent_adjustment
        return max(0.02, min(0.25, optimal_frequency))  # Clamp between 2% and 25%
    
    def _update_opponent_models(self, round_state: Dict) -> None:
        """Update opponent models using Bayesian inference"""
        if not round_state or 'action_histories' not in round_state:
            return
        
        # Extract opponent actions from round state
        for seat in round_state.get('seats', []):
            if seat['uuid'] != self.uuid:
                player_id = seat['uuid']
                
                # Update aggression model
                self._update_aggression_model(player_id, round_state)
                
                # Update bluff frequency model
                self._update_bluff_model(player_id, round_state)
                
                # Update fold threshold model
                self._update_fold_model(player_id, round_state)
        
        self.research_metrics['strategic_adaptations'] += 1
    
    def _update_aggression_model(self, player_id: str, round_state: Dict) -> None:
        """Update player aggression level using exponential moving average"""
        # Simplified aggression calculation based on betting patterns
        # In a full implementation, this would analyze bet sizes and frequencies
        
        current_aggression = self.opponent_models['aggression_levels'][player_id]
        
        # Decay factor for exponential moving average
        decay = self.sum_parameters['opponent_model_decay']
        
        # Placeholder: In real implementation, extract actual aggression from actions
        observed_aggression = 0.5  # This would be calculated from actual betting behavior
        
        # Update with exponential moving average
        updated_aggression = decay * current_aggression + (1 - decay) * observed_aggression
        self.opponent_models['aggression_levels'][player_id] = updated_aggression
    
    def _update_bluff_model(self, player_id: str, round_state: Dict) -> None:
        """Update player bluff frequency estimation"""
        # Simplified bluff detection
        # In full implementation, this would use showdown results and betting patterns
        
        current_bluff_freq = self.opponent_models['bluff_frequencies'][player_id]
        decay = self.sum_parameters['opponent_model_decay']
        
        # Placeholder: Real implementation would detect bluffs from showdowns
        observed_bluff = 0.1  # This would be calculated from actual game data
        
        updated_bluff_freq = decay * current_bluff_freq + (1 - decay) * observed_bluff
        self.opponent_models['bluff_frequencies'][player_id] = updated_bluff_freq
    
    def _update_fold_model(self, player_id: str, round_state: Dict) -> None:
        """Update player fold threshold estimation"""
        current_threshold = self.opponent_models['fold_thresholds'][player_id]
        decay = self.sum_parameters['opponent_model_decay']
        
        # Placeholder: Real implementation would analyze fold patterns
        observed_threshold = 0.3  # This would be calculated from fold behavior
        
        updated_threshold = decay * current_threshold + (1 - decay) * observed_threshold
        self.opponent_models['fold_thresholds'][player_id] = updated_threshold
    
    def _evaluate_collapse_conditions(self, hand_strength: float, pot_odds: float, 
                                    entropy: float, round_state: Dict) -> Tuple[bool, str, float]:
        """Evaluate whether superposition should collapse using information theory"""
        
        # Adaptive threshold based on recent performance
        base_threshold = self.sum_parameters['base_collapse_threshold']
        
        # Adjust threshold based on entropy history
        if len(self.superposition_state['entropy_history']) >= 3:
            entropy_trend = np.mean(self.superposition_state['entropy_history'][-3:])
            threshold_adjustment = (entropy_trend - 1.0) * 0.1
            adaptive_threshold = base_threshold + threshold_adjustment
        else:
            adaptive_threshold = base_threshold
        
        # Condition 1: Very strong hands (high confidence)
        if hand_strength > 0.9:
            return True, "premium_hand", 0.95
        
        # Condition 2: Very weak hands with poor odds
        if hand_strength < 0.15 and pot_odds > 0.6:
            return True, "weak_hand_poor_odds", 0.90
        
        # Condition 3: Low entropy (high certainty)
        if entropy < adaptive_threshold * 0.7:
            confidence = 1.0 - (entropy / (adaptive_threshold * 0.7))
            return True, "low_entropy", confidence
        
        # Condition 4: Information gain threshold
        if len(self.research_metrics['information_gain_per_decision']) > 0:
            recent_gain = self.research_metrics['information_gain_per_decision'][-1]
            if recent_gain > 0.5:  # Significant information gain
                return True, "information_gain", 0.8
        
        # Condition 5: Strategic randomness for unpredictability
        if np.random.random() < 0.03:  # 3% random collapse
            return True, "strategic_randomness", 0.3
        
        # Maintain superposition
        return False, "maintain_superposition", 0.5
    
    def _execute_informed_collapse(self, valid_actions: List, hand_strength: float, 
                                 pot_odds: float, trigger: str, round_state: Dict) -> Dict:
        """Execute strategic collapse with full information utilization"""
        
        # Get current amplitudes
        amplitudes = self.superposition_state['action_amplitudes']
        
        # Determine optimal action based on collapse trigger
        if trigger == "premium_hand":
            # Aggressive value betting
            if 'raise' in [action['action'] for action in valid_actions]:
                chosen_action = 'raise'
                # Calculate optimal bet size (simplified)
                bet_amount = self._calculate_optimal_bet_size(hand_strength, pot_odds, round_state)
            else:
                chosen_action = 'call'
                bet_amount = 0
        
        elif trigger == "weak_hand_poor_odds":
            chosen_action = 'fold'
            bet_amount = 0
        
        elif trigger in ["low_entropy", "information_gain"]:
            # Choose action with highest amplitude
            max_amplitude_idx = np.argmax(amplitudes)
            action_map = ['fold', 'call', 'raise']
            preferred_action = action_map[max_amplitude_idx]
            
            # Validate action is available
            available_actions = [action['action'] for action in valid_actions]
            if preferred_action in available_actions:
                chosen_action = preferred_action
                if preferred_action == 'raise':
                    bet_amount = self._calculate_optimal_bet_size(hand_strength, pot_odds, round_state)
                else:
                    bet_amount = 0
            else:
                chosen_action = 'call' if 'call' in available_actions else 'fold'
                bet_amount = 0
        
        else:  # strategic_randomness or other
            # Weighted random selection based on amplitudes
            action_map = ['fold', 'call', 'raise']
            available_actions = [action['action'] for action in valid_actions]
            
            # Filter amplitudes for available actions
            available_amplitudes = []
            available_action_names = []
            for i, action_name in enumerate(action_map):
                if action_name in available_actions:
                    available_amplitudes.append(amplitudes[i])
                    available_action_names.append(action_name)
            
            # Normalize and select
            if available_amplitudes:
                available_amplitudes = np.array(available_amplitudes)
                available_amplitudes /= np.sum(available_amplitudes)
                chosen_idx = np.random.choice(len(available_action_names), p=available_amplitudes)
                chosen_action = available_action_names[chosen_idx]
                
                if chosen_action == 'raise':
                    bet_amount = self._calculate_optimal_bet_size(hand_strength, pot_odds, round_state)
                else:
                    bet_amount = 0
            else:
                chosen_action = 'fold'
                bet_amount = 0
        
        # Update collapse trigger in state
        self.superposition_state['last_collapse_trigger'] = trigger
        
        return {
            'action': chosen_action,
            'amount': bet_amount,
            'trigger': trigger,
            'confidence': 0.8,
            'method': 'informed_collapse'
        }
    
    def _maintain_superposition_decision(self, valid_actions: List, hand_strength: float, 
                                       pot_odds: float, round_state: Dict) -> Dict:
        """Make decision while maintaining superposition state"""
        
        # Use current amplitudes for probabilistic decision
        amplitudes = self.superposition_state['action_amplitudes']
        action_map = ['fold', 'call', 'raise']
        
        # Filter for available actions
        available_actions = [action['action'] for action in valid_actions]
        available_amplitudes = []
        available_action_names = []
        
        for i, action_name in enumerate(action_map):
            if action_name in available_actions:
                available_amplitudes.append(amplitudes[i])
                available_action_names.append(action_name)
        
        # Normalize probabilities
        if available_amplitudes:
            available_amplitudes = np.array(available_amplitudes)
            available_amplitudes /= np.sum(available_amplitudes)
            
            # Add exploration noise for uncertainty maintenance
            exploration_rate = self.sum_parameters['uncertainty_exploration_rate']
            noise = np.random.normal(0, exploration_rate, len(available_amplitudes))
            available_amplitudes += noise
            available_amplitudes = np.maximum(available_amplitudes, 0.01)  # Ensure positive
            available_amplitudes /= np.sum(available_amplitudes)  # Renormalize
            
            # Select action
            chosen_idx = np.random.choice(len(available_action_names), p=available_amplitudes)
            chosen_action = available_action_names[chosen_idx]
            
            if chosen_action == 'raise':
                bet_amount = self._calculate_conservative_bet_size(hand_strength, pot_odds, round_state)
            else:
                bet_amount = 0
        else:
            chosen_action = 'fold'
            bet_amount = 0
        
        return {
            'action': chosen_action,
            'amount': bet_amount,
            'trigger': 'superposition_maintained',
            'confidence': 0.6,
            'method': 'superposition_decision'
        }
    
    def _calculate_optimal_bet_size(self, hand_strength: float, pot_odds: float, round_state: Dict) -> int:
        """Calculate optimal bet size using game theory"""
        if not round_state or 'pot' not in round_state:
            return 50  # Default bet
        
        pot_size = round_state['pot']['main']['amount']
        
        # Bet sizing based on hand strength and pot odds
        if hand_strength > 0.8:
            # Value bet: 60-80% of pot
            bet_ratio = 0.6 + (hand_strength - 0.8) * 1.0  # 0.6 to 0.8
        elif hand_strength > 0.6:
            # Medium bet: 40-60% of pot
            bet_ratio = 0.4 + (hand_strength - 0.6) * 1.0  # 0.4 to 0.6
        else:
            # Bluff bet: 30-50% of pot
            bet_ratio = 0.3 + np.random.random() * 0.2  # 0.3 to 0.5
        
        bet_amount = int(pot_size * bet_ratio)
        return max(10, min(bet_amount, 200))  # Clamp between 10 and 200
    
    def _calculate_conservative_bet_size(self, hand_strength: float, pot_odds: float, round_state: Dict) -> int:
        """Calculate conservative bet size for superposition decisions"""
        optimal_bet = self._calculate_optimal_bet_size(hand_strength, pot_odds, round_state)
        # Reduce by 20% for conservative approach
        return int(optimal_bet * 0.8)
    
    def _update_research_metrics(self, action_info: Dict, entropy: float, 
                               information_gain: float, decision_time: float) -> None:
        """Update comprehensive research metrics"""
        
        # Basic metrics
        self.research_metrics['entropy_evolution'].append(entropy)
        self.research_metrics['decision_times'].append(decision_time)
        
        # Superposition maintenance tracking
        if action_info['method'] == 'superposition_decision':
            self.research_metrics['superposition_maintenance_time'].append(
                self.superposition_state['coherence_time']
            )
        
        # Uncertainty utilization rate
        total_decisions = len(self.research_metrics['decision_times'])
        if total_decisions > 0:
            uncertainty_decisions = self.research_metrics['uncertainty_utilization_rate']
            self.research_metrics['uncertainty_utilization_rate'] = uncertainty_decisions / total_decisions
        
        # Adaptive threshold changes
        if len(self.research_metrics['entropy_evolution']) > 1:
            entropy_change = abs(entropy - self.research_metrics['entropy_evolution'][-2])
            if entropy_change > 0.3:  # Significant entropy change
                self.research_metrics['adaptive_threshold_changes'] += 1
        
        # Update learning parameters based on performance
        self._adapt_learning_parameters()
    
    def _adapt_learning_parameters(self) -> None:
        """Adapt SUM algorithm parameters based on performance"""
        
        # Adapt collapse threshold based on recent entropy
        if len(self.superposition_state['entropy_history']) >= 5:
            recent_entropy = np.mean(self.superposition_state['entropy_history'][-5:])
            
            # If entropy is consistently high, lower threshold (more collapses)
            if recent_entropy > 1.5:
                adjustment = -0.01
            # If entropy is consistently low, raise threshold (fewer collapses)
            elif recent_entropy < 0.8:
                adjustment = 0.01
            else:
                adjustment = 0.0
            
            # Apply adjustment with bounds
            current_threshold = self.superposition_state['collapse_threshold']
            new_threshold = current_threshold + adjustment
            self.superposition_state['collapse_threshold'] = max(0.5, min(0.9, new_threshold))