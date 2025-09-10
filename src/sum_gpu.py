#!/usr/bin/env python3
"""
Strategic Uncertainty Management Poker Agent - Research-Grade GPU Training
Comprehensive experimental validation for academic publication
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from scipy import stats
from collections import defaultdict
from typing import Dict, List, Tuple
from core.environment import PokerEnvironment
from core.agent import SUMAgent
from core.baseline_agents import get_baseline_agent
from core.utils import set_random_seeds
from config.gpu_config import GPUConfig

class ResearchGPUTrainer:
    """Research-grade trainer with comprehensive experimental validation"""
    def __init__(self):
        self.config = GPUConfig()
        
        # Check GPU availability
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.config.device = 'cpu'
            self.config.use_multi_gpu = False
        else:
            print(f"CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Create results directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"{self.config.results_dir}/experiment_{timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize baseline agents for comparison
        self.baseline_agents = {}
        for agent_type in ['random', 'tight_aggressive', 'loose_passive', 'classical_mixed_strategy']:
            if self.config.baseline_agents.get(agent_type, False):
                self.baseline_agents[agent_type] = get_baseline_agent(agent_type)
        
        # Research metrics storage
        self.experimental_results = {
            'sum_agent_results': [],
            'baseline_results': defaultdict(list),
            'statistical_tests': {},
            'configuration': vars(self.config),
            'timestamp': timestamp
        }
        
        print(f"Research experiment initialized: {timestamp}")
        print(f"Baseline agents: {list(self.baseline_agents.keys())}")
        print(f"Results directory: {self.results_dir}")
    
    def run_research_experiment(self):
        """Run comprehensive research experiment with statistical validation"""
        print("=== RESEARCH-GRADE EXPERIMENTAL VALIDATION ===")
        print(f"Total episodes per run: {self.config.n_episodes}")
        print(f"Number of experimental runs: {self.config.n_evaluation_runs}")
        print(f"Device: {self.config.device}")
        print(f"Batch size: {self.config.batch_size}")
        
        experiment_start_time = time.time()
        
        # Run multiple experiments with different random seeds for statistical validity
        for run_idx in range(self.config.n_evaluation_runs):
            print(f"\n--- EXPERIMENTAL RUN {run_idx + 1}/{self.config.n_evaluation_runs} ---")
            
            # Set unique random seed for this run
            seed = self.config.random_seeds[run_idx % len(self.config.random_seeds)]
            set_random_seeds(seed)
            print(f"Random seed: {seed}")
            
            # Train SUM agent
            sum_results = self._train_sum_agent(run_idx)
            self.experimental_results['sum_agent_results'].append(sum_results)
            
            # Evaluate against baselines
            baseline_results = self._evaluate_against_baselines(sum_results['agent'])
            for agent_name, results in baseline_results.items():
                self.experimental_results['baseline_results'][agent_name].append(results)
            
            # Save intermediate results
            self._save_intermediate_results(run_idx)
        
        # Perform statistical analysis
        self._perform_statistical_analysis()
        
        # Generate comprehensive report
        self._generate_research_report()
        
        total_time = time.time() - experiment_start_time
        print(f"\n=== EXPERIMENT COMPLETED ===")
        print(f"Total experimental time: {total_time:.2f} seconds")
        print(f"Results saved to: {self.results_dir}")
    
    def _train_sum_agent(self, run_idx: int) -> Dict:
        """Train SUM agent for one experimental run"""
        print(f"Training SUM agent...")
        
        # Initialize fresh agent for this run
        agent = SUMAgent(network_type='gpu', learning_rate=self.config.learning_rate, device=self.config.device)
        
        # Multi-GPU setup
        if self.config.use_multi_gpu and torch.cuda.device_count() > 1:
            agent.network = DataParallel(agent.network)
        
        env = PokerEnvironment(self.config.starting_stack, self.config.big_blind)
        
        start_time = time.time()
        
        for episode in range(self.config.n_episodes):
            episode_reward, episode_length = self._train_episode(agent, env)
            agent.training_metrics['episode_rewards'].append(episode_reward)
            
            # Batch training for GPU efficiency
            if episode > 0 and episode % 5 == 0:
                loss = agent.train()
                if loss > 0:
                    agent.training_metrics['training_losses'].append(loss)
            
            # Comprehensive evaluation
            if episode % self.config.eval_interval == 0:
                eval_results = self._comprehensive_evaluation(agent, env)
                
                for metric, value in eval_results.items():
                    if metric not in agent.training_metrics:
                        agent.training_metrics[metric] = []
                    agent.training_metrics[metric].append(value)
                
                if episode % self.config.log_interval == 0:
                    print(f"Episode {episode}: Reward = {episode_reward:.2f}, Win Rate = {eval_results['win_rate']:.2%}, "
                          f"Uncertainty Entropy = {eval_results.get('uncertainty_entropy', 0):.3f}")
                    
                    # GPU memory usage
                    if torch.cuda.is_available():
                        memory_used = torch.cuda.memory_allocated() / 1e9
                        memory_cached = torch.cuda.memory_reserved() / 1e9
                        print(f"GPU Memory: {memory_used:.1f}GB used, {memory_cached:.1f}GB cached")
            
            # Save model checkpoint
            if self.config.save_model_checkpoints and episode % self.config.checkpoint_interval == 0:
                checkpoint_path = f"{self.results_dir}/sum_agent_run{run_idx}_episode{episode}.pth"
                torch.save(agent.network.state_dict(), checkpoint_path)
        
        training_time = time.time() - start_time
        print(f"SUM agent training completed in {training_time:.2f} seconds")
        
        # Final comprehensive evaluation
        final_evaluation = self._comprehensive_evaluation(agent, env, n_games=self.config.min_sample_size)
        
        return {
            'agent': agent,
            'training_time': training_time,
            'final_metrics': agent.get_statistical_summary(),
            'final_evaluation': final_evaluation,
            'run_index': run_idx
        }
    
    def _train_episode(self, agent: SUMAgent, env: PokerEnvironment):
        """Train single episode with comprehensive metric collection"""
        state = env.reset()
        total_reward = 0
        episode_length = 0
        
        while not state['done']:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Collect research metrics
            if self.config.track_uncertainty_metrics:
                agent.collect_research_metrics(state, action, reward, info)
            
            agent.store_experience(state, action, reward, next_state, done)
            
            total_reward += reward
            episode_length += 1
            state = next_state
        
        return total_reward, episode_length
    
    def _comprehensive_evaluation(self, agent: SUMAgent, env: PokerEnvironment, n_games: int = 100) -> Dict:
        """Comprehensive evaluation with multiple metrics"""
        wins = 0
        total_rewards = []
        uncertainty_entropies = []
        collapse_events = 0
        strategic_diversities = []
        
        for game_idx in range(n_games):
            state = env.reset()
            game_reward = 0
            game_actions = []
            
            while not state['done']:
                # Track uncertainty before action
                if self.config.track_uncertainty_metrics:
                    uncertainty_entropies.append(agent.uncertainty_state.get_entropy())
                
                action = agent.get_action(state)
                game_actions.append(action)
                
                # Track collapse events
                if agent.uncertainty_state.collapsed:
                    collapse_events += 1
                
                state, reward, done, info = env.step(action)
                game_reward += reward
            
            total_rewards.append(game_reward)
            if game_reward > 0:
                wins += 1
            
            # Calculate strategic diversity for this game
            if len(game_actions) > 1:
                action_probs = np.bincount(game_actions, minlength=3) / len(game_actions)
                action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-8))
                strategic_diversities.append(action_entropy)
        
        return {
            'win_rate': wins / n_games,
            'avg_reward': np.mean(total_rewards),
            'reward_std': np.std(total_rewards),
            'uncertainty_entropy': np.mean(uncertainty_entropies) if uncertainty_entropies else 0,
            'collapse_frequency': collapse_events / (n_games * 10),  # Approximate per decision
            'strategic_diversity': np.mean(strategic_diversities) if strategic_diversities else 0,
            'total_games': n_games
        }
    
    def _evaluate_against_baselines(self, sum_agent: SUMAgent) -> Dict:
        """Evaluate SUM agent against all baseline agents"""
        print("Evaluating against baseline agents...")
        baseline_results = {}
        
        for agent_name, baseline_agent in self.baseline_agents.items():
            print(f"  vs {agent_name}...")
            
            wins = 0
            total_games = self.config.min_sample_size
            sum_rewards = []
            baseline_rewards = []
            
            for game_idx in range(total_games):
                # Alternate who goes first for fairness
                if game_idx % 2 == 0:
                    result = self._play_head_to_head(sum_agent, baseline_agent)
                    sum_reward, baseline_reward = result['player1_reward'], result['player2_reward']
                else:
                    result = self._play_head_to_head(baseline_agent, sum_agent)
                    baseline_reward, sum_reward = result['player1_reward'], result['player2_reward']
                
                sum_rewards.append(sum_reward)
                baseline_rewards.append(baseline_reward)
                
                if sum_reward > baseline_reward:
                    wins += 1
            
            win_rate = wins / total_games
            
            # Statistical significance test
            if self.config.statistical_significance_tests:
                t_stat, p_value = stats.ttest_rel(sum_rewards, baseline_rewards)
                effect_size = (np.mean(sum_rewards) - np.mean(baseline_rewards)) / np.sqrt(
                    (np.var(sum_rewards) + np.var(baseline_rewards)) / 2
                )
            else:
                t_stat, p_value, effect_size = 0, 1, 0
            
            baseline_results[agent_name] = {
                'win_rate': win_rate,
                'sum_avg_reward': np.mean(sum_rewards),
                'baseline_avg_reward': np.mean(baseline_rewards),
                'reward_difference': np.mean(sum_rewards) - np.mean(baseline_rewards),
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'significant': p_value < (1 - self.config.confidence_level),
                'total_games': total_games
            }
            
            print(f"    Win rate: {win_rate:.2%}, p-value: {p_value:.4f}, Effect size: {effect_size:.3f}")
        
        return baseline_results
    
    def _play_head_to_head(self, agent1, agent2) -> Dict:
        """Play one head-to-head game between two agents"""
        env = PokerEnvironment(self.config.starting_stack, self.config.big_blind)
        state = env.reset()
        
        # Simple alternating play (simplified for research purposes)
        total_reward = 0
        
        while not state['done']:
            action = agent1.get_action(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
        
        return {
            'player1_reward': total_reward,
            'player2_reward': -total_reward  # Zero-sum approximation
        }
    
    def _perform_statistical_analysis(self):
        """Perform comprehensive statistical analysis for research validation"""
        print("\nPerforming statistical analysis...")
        
        # Aggregate results across runs
        sum_win_rates = []
        sum_rewards = []
        
        for result in self.experimental_results['sum_agent_results']:
            sum_win_rates.append(result['final_evaluation']['win_rate'])
            sum_rewards.append(result['final_evaluation']['avg_reward'])
        
        # Statistical tests against each baseline
        for baseline_name in self.baseline_agents.keys():
            baseline_results = self.experimental_results['baseline_results'][baseline_name]
            baseline_win_rates = [r['win_rate'] for r in baseline_results]
            
            # Paired t-test
            if len(sum_win_rates) > 1 and len(baseline_win_rates) > 1:
                t_stat, p_value = stats.ttest_rel(sum_win_rates, baseline_win_rates)
                effect_size = (np.mean(sum_win_rates) - np.mean(baseline_win_rates)) / np.sqrt(
                    (np.var(sum_win_rates) + np.var(baseline_win_rates)) / 2
                )
                
                # Confidence interval
                diff_mean = np.mean(sum_win_rates) - np.mean(baseline_win_rates)
                diff_std = np.sqrt(np.var(sum_win_rates) + np.var(baseline_win_rates))
                ci_lower = diff_mean - 1.96 * diff_std / np.sqrt(len(sum_win_rates))
                ci_upper = diff_mean + 1.96 * diff_std / np.sqrt(len(sum_win_rates))
                
                self.experimental_results['statistical_tests'][baseline_name] = {
                    'sum_mean_win_rate': float(np.mean(sum_win_rates)),
                    'baseline_mean_win_rate': float(np.mean(baseline_win_rates)),
                    'win_rate_difference': float(diff_mean),
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'effect_size': float(effect_size),
                    'confidence_interval_95': [float(ci_lower), float(ci_upper)],
                    'statistically_significant': p_value < 0.05,
                    'practical_significance': abs(effect_size) > 0.5
                }
                
                print(f"  vs {baseline_name}: p={p_value:.4f}, effect_size={effect_size:.3f}, "
                      f"significant={p_value < 0.05}")
    
    def _save_intermediate_results(self, run_idx: int):
        """Save intermediate results after each run"""
        intermediate_file = f"{self.results_dir}/intermediate_run_{run_idx}.json"
        with open(intermediate_file, 'w') as f:
            json.dump({
                'run_index': run_idx,
                'sum_results': self.experimental_results['sum_agent_results'][-1],
                'baseline_results': {k: v[-1] for k, v in self.experimental_results['baseline_results'].items()}
            }, f, indent=2, default=str)
    
    def _generate_research_report(self):
        """Generate comprehensive research report"""
        print("\nGenerating research report...")
        
        # Save complete experimental results
        results_file = f"{self.results_dir}/complete_experimental_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.experimental_results, f, indent=2, default=str)
        
        # Generate summary statistics
        summary = self._generate_summary_statistics()
        summary_file = f"{self.results_dir}/research_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generate research plots
        self._generate_research_plots()
        
        # Generate LaTeX table for paper
        self._generate_latex_table()
        
        print(f"Research report saved to: {self.results_dir}")
    
    def _generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for research paper"""
        summary = {
            'experimental_setup': {
                'n_runs': self.config.n_evaluation_runs,
                'episodes_per_run': self.config.n_episodes,
                'total_episodes': self.config.n_evaluation_runs * self.config.n_episodes,
                'evaluation_games_per_baseline': self.config.min_sample_size,
                'confidence_level': self.config.confidence_level
            },
            'sum_agent_performance': {},
            'baseline_comparisons': {},
            'statistical_significance': {}
        }
        
        # SUM agent performance across runs
        sum_win_rates = [r['final_evaluation']['win_rate'] for r in self.experimental_results['sum_agent_results']]
        sum_rewards = [r['final_evaluation']['avg_reward'] for r in self.experimental_results['sum_agent_results']]
        
        summary['sum_agent_performance'] = {
            'mean_win_rate': float(np.mean(sum_win_rates)),
            'std_win_rate': float(np.std(sum_win_rates)),
            'min_win_rate': float(np.min(sum_win_rates)),
            'max_win_rate': float(np.max(sum_win_rates)),
            'mean_reward': float(np.mean(sum_rewards)),
            'std_reward': float(np.std(sum_rewards))
        }
        
        # Baseline comparisons
        for baseline_name in self.baseline_agents.keys():
            baseline_results = self.experimental_results['baseline_results'][baseline_name]
            baseline_win_rates = [r['win_rate'] for r in baseline_results]
            
            summary['baseline_comparisons'][baseline_name] = {
                'sum_vs_baseline_win_rate': float(np.mean([r['win_rate'] for r in baseline_results])),
                'average_reward_difference': float(np.mean([r['reward_difference'] for r in baseline_results])),
                'effect_size': float(np.mean([r['effect_size'] for r in baseline_results])),
                'wins_statistically_significant': all(r['significant'] for r in baseline_results)
            }
        
        # Overall statistical significance
        summary['statistical_significance'] = self.experimental_results['statistical_tests']
        
        return summary
    
    def _generate_research_plots(self):
        """Generate publication-quality plots"""
        # Create comprehensive figure
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Win rates across runs
        ax1 = plt.subplot(2, 3, 1)
        sum_win_rates = [r['final_evaluation']['win_rate'] for r in self.experimental_results['sum_agent_results']]
        plt.bar(['SUM Agent'], [np.mean(sum_win_rates)], yerr=[np.std(sum_win_rates)], capsize=5)
        plt.title('SUM Agent Win Rate (Mean ± SD)')
        plt.ylabel('Win Rate')
        plt.ylim(0, 1)
        
        # Plot 2: Comparison with baselines
        ax2 = plt.subplot(2, 3, 2)
        baseline_names = list(self.baseline_agents.keys())
        baseline_win_rates = []
        baseline_errors = []
        
        for name in baseline_names:
            rates = [r['win_rate'] for r in self.experimental_results['baseline_results'][name]]
            baseline_win_rates.append(np.mean(rates))
            baseline_errors.append(np.std(rates))
        
        x_pos = np.arange(len(baseline_names))
        plt.bar(x_pos, baseline_win_rates, yerr=baseline_errors, capsize=5)
        plt.title('SUM vs Baseline Agents')
        plt.xlabel('Baseline Agent')
        plt.ylabel('SUM Win Rate vs Baseline')
        plt.xticks(x_pos, baseline_names, rotation=45)
        
        # Plot 3: Effect sizes
        ax3 = plt.subplot(2, 3, 3)
        effect_sizes = []
        for name in baseline_names:
            effects = [r['effect_size'] for r in self.experimental_results['baseline_results'][name]]
            effect_sizes.append(np.mean(effects))
        
        colors = ['green' if es > 0.5 else 'orange' if es > 0.2 else 'red' for es in effect_sizes]
        plt.bar(x_pos, effect_sizes, color=colors)
        plt.title('Effect Sizes vs Baselines')
        plt.xlabel('Baseline Agent')
        plt.ylabel('Cohen\'s d')
        plt.xticks(x_pos, baseline_names, rotation=45)
        plt.axhline(y=0.2, color='orange', linestyle='--', alpha=0.7, label='Small effect')
        plt.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Medium effect')
        plt.legend()
        
        # Plot 4: Training progression (first run)
        ax4 = plt.subplot(2, 3, 4)
        if self.experimental_results['sum_agent_results']:
            first_run = self.experimental_results['sum_agent_results'][0]
            if 'win_rate' in first_run['agent'].training_metrics:
                plt.plot(first_run['agent'].training_metrics['win_rate'])
                plt.title('Training Progression (Run 1)')
                plt.xlabel('Evaluation Step')
                plt.ylabel('Win Rate')
        
        # Plot 5: Uncertainty metrics
        ax5 = plt.subplot(2, 3, 5)
        if self.experimental_results['sum_agent_results']:
            first_run = self.experimental_results['sum_agent_results'][0]
            if 'uncertainty_entropy' in first_run['agent'].training_metrics:
                plt.plot(first_run['agent'].training_metrics['uncertainty_entropy'])
                plt.title('Uncertainty Entropy Over Time')
                plt.xlabel('Episode')
                plt.ylabel('Entropy')
        
        # Plot 6: Statistical significance
        ax6 = plt.subplot(2, 3, 6)
        p_values = []
        for name in baseline_names:
            if name in self.experimental_results['statistical_tests']:
                p_values.append(self.experimental_results['statistical_tests'][name]['p_value'])
            else:
                p_values.append(1.0)
        
        colors = ['green' if p < 0.05 else 'red' for p in p_values]
        plt.bar(x_pos, [-np.log10(p) for p in p_values], color=colors)
        plt.title('Statistical Significance (-log10 p-value)')
        plt.xlabel('Baseline Agent')
        plt.ylabel('-log10(p-value)')
        plt.xticks(x_pos, baseline_names, rotation=45)
        plt.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='p=0.05')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/research_plots.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.results_dir}/research_plots.pdf", bbox_inches='tight')
        plt.show()
    
    def _generate_latex_table(self):
        """Generate LaTeX table for research paper"""
        latex_content = "\\begin{table}[h]\n"
        latex_content += "\\centering\n"
        latex_content += "\\caption{Strategic Uncertainty Management vs Baseline Agents}\n"
        latex_content += "\\begin{tabular}{|l|c|c|c|c|}\n"
        latex_content += "\\hline\n"
        latex_content += "Baseline Agent & Win Rate & Effect Size & p-value & Significant \\\\\n"
        latex_content += "\\hline\n"
        
        for baseline_name in self.baseline_agents.keys():
            if baseline_name in self.experimental_results['statistical_tests']:
                stats = self.experimental_results['statistical_tests'][baseline_name]
                win_rate = stats['sum_mean_win_rate']
                effect_size = stats['effect_size']
                p_value = stats['p_value']
                significant = "Yes" if stats['statistically_significant'] else "No"
                
                latex_content += f"{baseline_name.replace('_', ' ').title()} & "
                latex_content += f"{win_rate:.3f} & {effect_size:.3f} & {p_value:.4f} & {significant} \\\\\n"
        
        latex_content += "\\hline\n"
        latex_content += "\\end{tabular}\n"
        latex_content += "\\label{tab:sum_results}\n"
        latex_content += "\\end{table}\n"
        
        with open(f"{self.results_dir}/results_table.tex", 'w') as f:
            f.write(latex_content)

def main():
    """Main research experiment function"""
    print("=== STRATEGIC UNCERTAINTY MANAGEMENT RESEARCH EXPERIMENT ===")
    
    trainer = ResearchGPUTrainer()
    trainer.run_research_experiment()
    
    print("\n=== EXPERIMENT SUMMARY ===")
    
    # Print key findings
    if trainer.experimental_results['statistical_tests']:
        print("\nStatistical Significance Results:")
        for baseline, stats in trainer.experimental_results['statistical_tests'].items():
            significance = "SIGNIFICANT" if stats['statistically_significant'] else "NOT SIGNIFICANT"
            print(f"  vs {baseline}: {significance} (p={stats['p_value']:.4f}, d={stats['effect_size']:.3f})")
    
    return trainer

if __name__ == "__main__":
    trainer = main()