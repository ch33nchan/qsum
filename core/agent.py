import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict, Tuple
from .utils import HandEvaluator, StrategicUncertaintyState
from .network import SUMNetworkCPU, SUMNetworkGPU

class SUMAgent:
    def __init__(self, network_type: str = 'cpu', learning_rate: float = 1e-3, device: str = 'cpu'):
        self.device = torch.device(device)
        self.network_type = network_type
        
        if network_type == 'cpu':
            self.network = SUMNetworkCPU().to(self.device)
            self.batch_size = 32
            self.memory_size = 1000
        else:  # gpu
            self.network = SUMNetworkGPU().to(self.device)
            self.batch_size = 256
            self.memory_size = 50000
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.uncertainty_state = StrategicUncertaintyState()
        
        # Training parameters
        self.gamma = 0.99
        self.epsilon = 0.1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        # Experience buffer
        self.memory = deque(maxlen=self.memory_size)
        
        # Performance tracking
        self.training_metrics = {
            'episode_rewards': [],
            'training_losses': [],
            'win_rates': [],
            'uncertainty_entropy': [],
            'collapse_events': [],
            'strategic_diversity': [],
            'deception_success': [],
            'hand_strength_correlation': [],
            'betting_patterns': [],
            'opponent_exploitation': []
        }
        
        # Research-specific metrics
        self.research_metrics = {
            'superposition_duration': [],
            'collapse_timing_optimality': [],
            'strategic_phase_utilization': [],
            'information_advantage': [],
            'bluff_success_rate': [],
            'value_bet_accuracy': []
        }
    
    def get_action(self, state: Dict) -> int:
        features = self._extract_features(state)
        
        with torch.no_grad():
            if self.network_type == 'cpu':
                action_probs, value, uncertainty_probs = self.network(features)
            else:
                action_probs, value, uncertainty_probs, strategy_probs = self.network(features)
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            action = torch.argmax(action_probs).item()
        
        # Update uncertainty state
        self._update_uncertainty_state(state, uncertainty_probs)
        
        return action
    
    def _extract_features(self, state: Dict) -> torch.Tensor:
        features = [
            state['pot'] / 100.0,
            state['player_stack'] / 200.0,
            state['opponent_stack'] / 200.0,
            state['street'] / 3.0,
            len(state['community_cards']) / 5.0,
        ]
        
        # Add hand strength
        if state['player_cards']:
            hand_strength = HandEvaluator.evaluate_hand(
                state['player_cards'], state['community_cards']
            )
            features.append(hand_strength)
        else:
            features.append(0.5)
        
        # Add uncertainty state features
        uncertainty_probs = self.uncertainty_state.get_probabilities()
        features.extend(uncertainty_probs[:8])
        
        # Add entropy
        features.append(self.uncertainty_state.get_entropy() / 3.0)
        
        # Pad features based on network type
        target_size = 64 if self.network_type == 'gpu' else 32
        while len(features) < target_size:
            features.append(0.0)
        
        return torch.tensor(features[:target_size], dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def _update_uncertainty_state(self, state: Dict, uncertainty_probs: torch.Tensor):
        probs = uncertainty_probs.cpu().numpy().flatten()
        
        # Update amplitudes
        self.uncertainty_state.amplitudes = np.sqrt(probs[:8]).astype(np.complex64)
        
        # Check for collapse
        context = {
            'pot_size': state['pot'],
            'street': state['street']
        }
        
        if self.uncertainty_state.should_collapse('call', context):
            strategy_idx = np.argmax(probs[:8])
            self.uncertainty_state.collapse_to_strategy(strategy_idx)
    
    def store_experience(self, state: Dict, action: int, reward: float, 
                        next_state: Dict, done: bool):
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0
        
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_features = torch.cat([self._extract_features(s) for s in states])
        action_indices = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_state_features = torch.cat([self._extract_features(s) for s in next_states])
        dones_tensor = torch.tensor(dones, dtype=torch.bool).to(self.device)
        
        # Current predictions
        if self.network_type == 'cpu':
            current_action_probs, current_values, current_uncertainty = self.network(state_features)
            with torch.no_grad():
                next_action_probs, next_values, next_uncertainty = self.network(next_state_features)
        else:
            current_action_probs, current_values, current_uncertainty, current_strategy = self.network(state_features)
            with torch.no_grad():
                next_action_probs, next_values, next_uncertainty, next_strategy = self.network(next_state_features)
        
        current_q_values = current_values.squeeze()
        next_q_values = next_values.squeeze()
        target_q_values = rewards_tensor + (self.gamma * next_q_values * ~dones_tensor)
        
        # Compute losses
        value_loss = nn.MSELoss()(current_q_values, target_q_values)
        
        advantages = target_q_values - current_q_values.detach()
        action_log_probs = torch.log(current_action_probs.gather(1, action_indices.unsqueeze(1)).squeeze())
        policy_loss = -(action_log_probs * advantages).mean()
        
        # Uncertainty regularization
        uncertainty_entropy = -(current_uncertainty * torch.log(current_uncertainty + 1e-8)).sum(dim=1).mean()
        
        if self.network_type == 'gpu':
            strategy_entropy = -(current_strategy * torch.log(current_strategy + 1e-8)).sum(dim=1).mean()
            total_loss = value_loss + policy_loss - 0.01 * (uncertainty_entropy + strategy_entropy)
        else:
            total_loss = value_loss + policy_loss - 0.01 * uncertainty_entropy
        
        # Optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return total_loss.item()
    
    def collect_research_metrics(self, state: Dict, action: int, reward: float, info: Dict):
        """Collect comprehensive metrics for research validation"""
        # Uncertainty metrics
        entropy = self.uncertainty_state.get_entropy()
        self.training_metrics['uncertainty_entropy'].append(entropy)
        
        # Collapse event tracking
        if self.uncertainty_state.collapsed:
            self.training_metrics['collapse_events'].append({
                'episode_step': len(self.training_metrics['episode_rewards']),
                'pot_size': state['pot'],
                'street': state['street'],
                'hand_strength': HandEvaluator.evaluate_hand(state['player_cards'], state['community_cards']) if state['player_cards'] else 0
            })
        
        # Strategic diversity (action entropy)
        if hasattr(self, '_recent_actions'):
            self._recent_actions.append(action)
            if len(self._recent_actions) > 10:
                self._recent_actions.pop(0)
                action_probs = np.bincount(self._recent_actions, minlength=3) / len(self._recent_actions)
                action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
                self.training_metrics['strategic_diversity'].append(action_entropy)
        else:
            self._recent_actions = [action]
        
        # Deception effectiveness (betting with weak hands)
        if state['player_cards']:
            hand_strength = HandEvaluator.evaluate_hand(state['player_cards'], state['community_cards'])
            if action == 2 and hand_strength < 0.3:  # Raise with weak hand
                self.research_metrics['bluff_success_rate'].append(reward > 0)
            elif action == 2 and hand_strength > 0.7:  # Raise with strong hand
                self.research_metrics['value_bet_accuracy'].append(reward > 0)
        
        # Superposition duration tracking
        if not self.uncertainty_state.collapsed:
            if hasattr(self, '_superposition_start'):
                duration = len(self.training_metrics['episode_rewards']) - self._superposition_start
                self.research_metrics['superposition_duration'].append(duration)
            else:
                self._superposition_start = len(self.training_metrics['episode_rewards'])
        else:
            if hasattr(self, '_superposition_start'):
                delattr(self, '_superposition_start')
    
    def get_statistical_summary(self) -> Dict:
        """Generate statistical summary for research paper"""
        import numpy as np
        from scipy import stats
        
        summary = {}
        
        # Basic performance metrics
        if self.training_metrics['episode_rewards']:
            rewards = np.array(self.training_metrics['episode_rewards'])
            summary['reward_stats'] = {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'median': float(np.median(rewards)),
                'min': float(np.min(rewards)),
                'max': float(np.max(rewards)),
                'q25': float(np.percentile(rewards, 25)),
                'q75': float(np.percentile(rewards, 75))
            }
        
        # Win rate statistics
        if self.training_metrics['win_rates']:
            win_rates = np.array(self.training_metrics['win_rates'])
            summary['win_rate_stats'] = {
                'final_win_rate': float(win_rates[-1]) if len(win_rates) > 0 else 0,
                'mean_win_rate': float(np.mean(win_rates)),
                'std_win_rate': float(np.std(win_rates)),
                'improvement': float(win_rates[-1] - win_rates[0]) if len(win_rates) > 1 else 0
            }
        
        # Uncertainty management metrics
        if self.training_metrics['uncertainty_entropy']:
            entropy = np.array(self.training_metrics['uncertainty_entropy'])
            summary['uncertainty_stats'] = {
                'mean_entropy': float(np.mean(entropy)),
                'entropy_variance': float(np.var(entropy)),
                'entropy_trend': float(np.corrcoef(range(len(entropy)), entropy)[0,1]) if len(entropy) > 1 else 0
            }
        
        # Strategic diversity metrics
        if self.training_metrics['strategic_diversity']:
            diversity = np.array(self.training_metrics['strategic_diversity'])
            summary['strategic_diversity_stats'] = {
                'mean_diversity': float(np.mean(diversity)),
                'diversity_consistency': float(1.0 - np.std(diversity))  # Higher is more consistent
            }
        
        # Collapse pattern analysis
        if self.training_metrics['collapse_events']:
            collapse_events = self.training_metrics['collapse_events']
            pot_sizes = [event['pot_size'] for event in collapse_events]
            hand_strengths = [event['hand_strength'] for event in collapse_events]
            
            summary['collapse_stats'] = {
                'collapse_frequency': len(collapse_events) / max(len(self.training_metrics['episode_rewards']), 1),
                'avg_collapse_pot_size': float(np.mean(pot_sizes)) if pot_sizes else 0,
                'collapse_hand_strength_correlation': float(np.corrcoef(pot_sizes, hand_strengths)[0,1]) if len(pot_sizes) > 1 else 0
            }
        
        # Deception effectiveness
        if self.research_metrics['bluff_success_rate']:
            bluff_success = np.array(self.research_metrics['bluff_success_rate'])
            summary['deception_stats'] = {
                'bluff_success_rate': float(np.mean(bluff_success)),
                'bluff_attempts': len(bluff_success)
            }
        
        if self.research_metrics['value_bet_accuracy']:
            value_accuracy = np.array(self.research_metrics['value_bet_accuracy'])
            summary['value_betting_stats'] = {
                'value_bet_accuracy': float(np.mean(value_accuracy)),
                'value_bet_attempts': len(value_accuracy)
            }
        
        # Superposition utilization
        if self.research_metrics['superposition_duration']:
            durations = np.array(self.research_metrics['superposition_duration'])
            summary['superposition_stats'] = {
                'avg_superposition_duration': float(np.mean(durations)),
                'superposition_utilization_rate': len(durations) / max(len(self.training_metrics['episode_rewards']), 1)
            }
        
        return summary