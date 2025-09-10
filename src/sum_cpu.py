#!/usr/bin/env python3
"""
Strategic Uncertainty Management Poker Agent - CPU Training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import matplotlib.pyplot as plt
import numpy as np
from core.environment import PokerEnvironment
from core.agent import SUMAgent
from core.utils import set_random_seeds
from config.cpu_config import CPUConfig

class CPUTrainer:
    def __init__(self):
        set_random_seeds(42)
        self.config = CPUConfig()
        self.agent = SUMAgent(network_type='cpu', learning_rate=self.config.learning_rate, device=self.config.device)
        self.env = PokerEnvironment(self.config.starting_stack, self.config.big_blind)
        
        # Create results directory
        os.makedirs(self.config.results_dir, exist_ok=True)
    
    def train(self):
        print(f"Starting CPU training for {self.config.n_episodes} episodes")
        print(f"Device: {self.config.device}")
        
        start_time = time.time()
        
        for episode in range(self.config.n_episodes):
            episode_reward, episode_length = self._train_episode()
            self.agent.training_metrics['episode_rewards'].append(episode_reward)
            
            # Train the agent
            if episode > 0 and episode % 10 == 0:
                loss = self.agent.train()
                if loss > 0:
                    self.agent.training_metrics['training_losses'].append(loss)
            
            # Evaluate
            if episode % self.config.eval_interval == 0:
                win_rate = self._evaluate(n_games=20)
                self.agent.training_metrics['win_rates'].append(win_rate)
                
                if episode % self.config.log_interval == 0:
                    print(f"Episode {episode}: Reward = {episode_reward:.2f}, Win Rate = {win_rate:.2%}, Epsilon = {self.agent.epsilon:.3f}")
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        if self.config.save_results:
            self._save_results()
            self._plot_results()
    
    def _train_episode(self):
        state = self.env.reset()
        total_reward = 0
        episode_length = 0
        
        while not state['done']:
            action = self.agent.get_action(state)
            next_state, reward, done, info = self.env.step(action)
            
            self.agent.store_experience(state, action, reward, next_state, done)
            
            total_reward += reward
            episode_length += 1
            state = next_state
        
        return total_reward, episode_length
    
    def _evaluate(self, n_games: int = 50):
        wins = 0
        
        for _ in range(n_games):
            state = self.env.reset()
            
            while not state['done']:
                action = self.agent.get_action(state)
                state, reward, done, info = self.env.step(action)
            
            if reward > 0:
                wins += 1
        
        return wins / n_games
    
    def _save_results(self):
        import json
        results = {
            'training_metrics': self.agent.training_metrics,
            'config': {
                'n_episodes': self.config.n_episodes,
                'learning_rate': self.config.learning_rate,
                'device': self.config.device
            }
        }
        
        with open(f"{self.config.results_dir}/training_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    def _plot_results(self):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Episode rewards
        if self.agent.training_metrics['episode_rewards']:
            axes[0].plot(self.agent.training_metrics['episode_rewards'])
            axes[0].set_title('Episode Rewards')
            axes[0].set_xlabel('Episode')
            axes[0].set_ylabel('Reward')
        
        # Training losses
        if self.agent.training_metrics['training_losses']:
            axes[1].plot(self.agent.training_metrics['training_losses'])
            axes[1].set_title('Training Loss')
            axes[1].set_xlabel('Training Step')
            axes[1].set_ylabel('Loss')
        
        # Win rates
        if self.agent.training_metrics['win_rates']:
            axes[2].plot(self.agent.training_metrics['win_rates'])
            axes[2].set_title('Win Rate')
            axes[2].set_xlabel('Evaluation')
            axes[2].set_ylabel('Win Rate')
        
        plt.tight_layout()
        plt.savefig(f"{self.config.results_dir}/training_plots.png", dpi=150, bbox_inches='tight')
        plt.show()

def main():
    trainer = CPUTrainer()
    trainer.train()
    
    # Final evaluation
    final_win_rate = trainer._evaluate(n_games=100)
    print(f"Final win rate: {final_win_rate:.2%}")
    
    return trainer

if __name__ == "__main__":
    trainer = main()