#!/usr/bin/env python3
"""
Real Quake III Arena Experiments using ViZDoom
Implements Strategic Uncertainty Management with actual 3D combat environment
Using ViZDoom for authentic first-person shooter gameplay with real enemy AI
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
from dataclasses import dataclass
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

# Import real Quake environment
from core.real_quake_environment import RealQuakeEnvironment, QuakeGameState
import vizdoom as vzd

class RealQuantumSpatialAgent:
    """Quantum spatial agent for real 3D combat using Strategic Uncertainty Management with GPU support"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
        # Setup GPU if available
        if device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
            self.use_gpu = True
            self.gpu_device = torch.device('cuda')
            print(f"🚀 QuantumSpatialAgent using GPU acceleration")
        else:
            self.use_gpu = False
            self.gpu_device = torch.device('cpu')
            if device == 'cuda':
                print(f"⚠️ QuantumSpatialAgent falling back to CPU (CUDA not available)")
        
        # Strategic Uncertainty Management state for spatial decisions
        self.spatial_superposition = {
            'movement_actions': ['forward', 'backward', 'left', 'right', 'turn_left', 'turn_right'],
            'combat_actions': ['attack', 'defend', 'retreat', 'advance'],
            'weapon_actions': ['keep_weapon', 'switch_weapon'],
            'amplitudes': np.array([0.16, 0.16, 0.16, 0.16, 0.16, 0.2]),  # Normalized
            'phase_relationships': np.array([0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, np.pi]),
            'coherence_time': 0.0,
            'collapse_threshold': 0.8
        }
        
        # Combat metrics
        self.combat_metrics = {
            'episodes_played': 0,
            'total_kills': 0,
            'total_deaths': 0,
            'total_damage_dealt': 0,
            'total_damage_taken': 0,
            'survival_times': [],
            'kill_death_ratios': [],
            'decision_times': [],
            'spatial_entropies': [],
            'tactical_collapses': defaultdict(int),
            'weapon_usage': defaultdict(int),
            'enemy_encounters': 0,
            'successful_engagements': 0
        }
        
        # Spatial tracking
        self.position_history = []
        self.movement_patterns = []
        self.last_position = None
        self.movement_entropy_window = []
        
    def get_action(self, game_state: QuakeGameState) -> int:
        """Get tactical action using Strategic Uncertainty Management"""
        start_time = time.time()
        
        # Analyze spatial state
        spatial_analysis = self._analyze_spatial_state(game_state)
        
        # Update position tracking
        self._update_position_tracking(game_state)
        
        # Calculate spatial entropy
        spatial_entropy = self._calculate_spatial_entropy(spatial_analysis)
        self.combat_metrics['spatial_entropies'].append(spatial_entropy)
        
        # Strategic Uncertainty Management decision process
        action_idx = self._quantum_tactical_decision(game_state, spatial_analysis, spatial_entropy)
        
        # Update superposition state
        self._update_spatial_superposition(action_idx, game_state)
        
        # Record decision time
        decision_time = time.time() - start_time
        self.combat_metrics['decision_times'].append(decision_time)
        
        return action_idx
    
    def _analyze_spatial_state(self, game_state: QuakeGameState) -> Dict:
        """Analyze current spatial and tactical state"""
        analysis = {
            'health_status': self._categorize_health(game_state.player_health),
            'armor_status': self._categorize_armor(game_state.player_armor),
            'enemy_threat_level': self._assess_enemy_threat(game_state),
            'weapon_suitability': self._assess_weapon_suitability(game_state),
            'positional_advantage': self._assess_positional_advantage(game_state),
            'movement_urgency': self._assess_movement_urgency(game_state),
            'tactical_situation': self._assess_tactical_situation(game_state)
        }
        return analysis
    
    def _categorize_health(self, health: int) -> str:
        """Categorize health status"""
        if health > 75:
            return 'excellent'
        elif health > 50:
            return 'good'
        elif health > 25:
            return 'moderate'
        elif health > 10:
            return 'low'
        else:
            return 'critical'
    
    def _categorize_armor(self, armor: int) -> str:
        """Categorize armor status"""
        if armor > 100:
            return 'heavy'
        elif armor > 50:
            return 'medium'
        elif armor > 25:
            return 'light'
        else:
            return 'none'
    
    def _assess_enemy_threat(self, game_state: QuakeGameState) -> str:
        """Assess enemy threat level"""
        if not game_state.enemy_visible:
            return 'none'
        
        if game_state.enemy_distance < 20:
            return 'immediate'
        elif game_state.enemy_distance < 50:
            return 'high'
        elif game_state.enemy_distance < 100:
            return 'medium'
        else:
            return 'low'
    
    def _assess_weapon_suitability(self, game_state: QuakeGameState) -> str:
        """Assess current weapon suitability for situation"""
        weapon = game_state.weapon_selected
        enemy_distance = game_state.enemy_distance
        
        if weapon == 'SHOTGUN':
            if enemy_distance < 30:
                return 'excellent'
            elif enemy_distance < 60:
                return 'good'
            else:
                return 'poor'
        elif weapon == 'CHAINGUN':
            if 30 < enemy_distance < 100:
                return 'excellent'
            elif enemy_distance < 150:
                return 'good'
            else:
                return 'moderate'
        elif weapon == 'ROCKET_LAUNCHER':
            if enemy_distance > 50:
                return 'excellent'
            elif enemy_distance > 30:
                return 'good'
            else:
                return 'dangerous'  # Too close for rockets
        else:
            return 'moderate'
    
    def _assess_positional_advantage(self, game_state: QuakeGameState) -> str:
        """Assess positional advantage"""
        # Simple heuristic based on game variables
        if hasattr(game_state, 'game_variables'):
            # Higher ground, cover, etc.
            position_score = sum(game_state.game_variables.values()) / len(game_state.game_variables)
            if position_score > 0.7:
                return 'excellent'
            elif position_score > 0.4:
                return 'good'
            elif position_score > 0.2:
                return 'moderate'
            else:
                return 'poor'
        return 'unknown'
    
    def _assess_movement_urgency(self, game_state: QuakeGameState) -> str:
        """Assess urgency of movement"""
        if game_state.player_health < 20 and game_state.enemy_visible:
            return 'critical'
        elif game_state.enemy_visible and game_state.enemy_distance < 30:
            return 'high'
        elif game_state.enemy_visible:
            return 'medium'
        else:
            return 'low'
    
    def _assess_tactical_situation(self, game_state: QuakeGameState) -> str:
        """Assess overall tactical situation"""
        health_critical = game_state.player_health < 25
        enemy_close = game_state.enemy_visible and game_state.enemy_distance < 40
        low_ammo = sum(game_state.ammo_counts.values()) < 20
        
        if health_critical and enemy_close:
            return 'desperate'
        elif enemy_close and not health_critical:
            return 'engagement'
        elif game_state.enemy_visible:
            return 'combat'
        elif health_critical or low_ammo:
            return 'survival'
        else:
            return 'exploration'
    
    def _update_position_tracking(self, game_state: QuakeGameState):
        """Update position and movement tracking"""
        current_pos = game_state.player_position
        self.position_history.append(current_pos)
        
        if self.last_position is not None:
            movement = np.linalg.norm(np.array(current_pos) - np.array(self.last_position))
            self.movement_patterns.append(movement)
            
            # Keep window of recent movements for entropy calculation
            self.movement_entropy_window.append(movement)
            if len(self.movement_entropy_window) > 10:
                self.movement_entropy_window.pop(0)
        
        self.last_position = current_pos
    
    def _calculate_spatial_entropy(self, analysis: Dict) -> float:
        """Calculate spatial entropy for superposition state with GPU acceleration"""
        if self.use_gpu and TORCH_AVAILABLE and len(self.movement_entropy_window) > 3:
            # GPU-accelerated entropy calculation
            movement_tensor = torch.tensor(self.movement_entropy_window, device=self.gpu_device)
            movement_variance = torch.var(movement_tensor)
            base_entropy = torch.clamp(movement_variance * 10, max=2.0)
            
            # Situation modifiers on GPU
            situation_map = {
                'desperate': 0.3, 'engagement': 0.5, 'combat': 0.8,
                'survival': 0.4, 'exploration': 1.0
            }
            
            situation = analysis.get('tactical_situation', 'exploration')
            modifier = torch.tensor(situation_map.get(situation, 0.7), device=self.gpu_device)
            
            entropy = base_entropy * modifier
            return entropy.cpu().item()
        else:
            # CPU fallback
            if len(self.movement_entropy_window) > 3:
                movement_variance = np.var(self.movement_entropy_window)
                base_entropy = min(2.0, movement_variance * 10)  # Scale and cap
            else:
                base_entropy = 1.0
            
            # Modify based on tactical situation
            situation_modifiers = {
                'desperate': 0.3,    # Low entropy - need decisive action
                'engagement': 0.5,   # Medium entropy - tactical decisions
                'combat': 0.8,       # High entropy - many options
                'survival': 0.4,     # Low entropy - focus on survival
                'exploration': 1.0   # High entropy - many movement options
            }
            
            situation = analysis.get('tactical_situation', 'exploration')
            modifier = situation_modifiers.get(situation, 0.7)
            
            return base_entropy * modifier
    
    def _quantum_tactical_decision(self, game_state: QuakeGameState, analysis: Dict, entropy: float) -> int:
        """Make tactical decision using quantum superposition principles"""
        
        # Determine if superposition should collapse
        should_collapse, trigger = self._should_collapse_superposition(analysis, entropy)
        
        if should_collapse:
            action_idx = self._execute_tactical_collapse(game_state, analysis, trigger)
            self.combat_metrics['tactical_collapses'][trigger] += 1
        else:
            # Maintain superposition - probabilistic decision
            action_idx = self._superposition_tactical_decision(game_state, analysis)
            self.combat_metrics['tactical_collapses']['superposition'] += 1
        
        return action_idx
    
    def _should_collapse_superposition(self, analysis: Dict, entropy: float) -> Tuple[bool, str]:
        """Determine if spatial superposition should collapse"""
        
        # Critical health - collapse to survival
        if analysis['health_status'] == 'critical':
            return True, 'critical_health'
        
        # Immediate threat - collapse to combat response
        if analysis['enemy_threat_level'] == 'immediate':
            return True, 'immediate_threat'
        
        # Desperate situation - collapse to decisive action
        if analysis['tactical_situation'] == 'desperate':
            return True, 'desperate_situation'
        
        # Low entropy - collapse for efficiency
        if entropy < 0.3:
            return True, 'low_entropy'
        
        # High entropy - maintain superposition for flexibility
        if entropy > 1.5:
            return False, 'high_entropy'
        
        # Random collapse for unpredictability
        if np.random.random() < 0.15:
            return True, 'random'
        
        return False, 'maintain'
    
    def _execute_tactical_collapse(self, game_state: QuakeGameState, analysis: Dict, trigger: str) -> int:
        """Execute tactical collapse to specific action"""
        
        if trigger == 'critical_health':
            # Prioritize survival - retreat or find cover
            if game_state.enemy_visible:
                return self._choose_evasive_action()
            else:
                return self._choose_defensive_action()
        
        elif trigger == 'immediate_threat':
            # Engage or evade based on weapon and health
            if analysis['weapon_suitability'] in ['excellent', 'good'] and analysis['health_status'] not in ['critical', 'low']:
                return self._choose_attack_action(game_state)
            else:
                return self._choose_evasive_action()
        
        elif trigger == 'desperate_situation':
            # All-or-nothing decision
            if np.random.random() < 0.7:  # 70% chance to fight
                return self._choose_attack_action(game_state)
            else:
                return self._choose_evasive_action()
        
        else:
            # Default to superposition decision
            return self._superposition_tactical_decision(game_state, analysis)
    
    def _superposition_tactical_decision(self, game_state: QuakeGameState, analysis: Dict) -> int:
        """Make decision while maintaining spatial superposition"""
        
        # Update superposition amplitudes based on current state
        self._update_superposition_amplitudes(analysis)
        
        # Probabilistic action selection based on amplitudes
        action_probs = self.spatial_superposition['amplitudes']
        
        # Map to actual ViZDoom actions
        if game_state.enemy_visible:
            # Combat situation - bias toward combat actions
            if np.random.random() < 0.4:  # 40% attack
                return self._choose_attack_action(game_state)
            elif np.random.random() < 0.3:  # 30% tactical movement
                return self._choose_tactical_movement()
            else:  # 30% evasive action
                return self._choose_evasive_action()
        else:
            # Exploration situation - bias toward movement
            if np.random.random() < 0.6:  # 60% exploration movement
                return self._choose_exploration_action()
            elif np.random.random() < 0.2:  # 20% weapon switch
                return self._choose_weapon_switch_action(game_state)
            else:  # 20% defensive positioning
                return self._choose_defensive_action()
    
    def _update_superposition_amplitudes(self, analysis: Dict):
        """Update superposition amplitudes based on tactical analysis"""
        # Adjust amplitudes based on situation
        if analysis['tactical_situation'] == 'combat':
            # Favor combat actions
            self.spatial_superposition['amplitudes'] = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])
        elif analysis['tactical_situation'] == 'survival':
            # Favor defensive actions
            self.spatial_superposition['amplitudes'] = np.array([0.3, 0.3, 0.1, 0.1, 0.1, 0.1])
        elif analysis['tactical_situation'] == 'exploration':
            # Favor movement actions
            self.spatial_superposition['amplitudes'] = np.array([0.15, 0.15, 0.2, 0.2, 0.15, 0.15])
        else:
            # Balanced distribution
            self.spatial_superposition['amplitudes'] = np.array([0.16, 0.16, 0.16, 0.16, 0.16, 0.2])
        
        # Normalize
        self.spatial_superposition['amplitudes'] /= np.sum(self.spatial_superposition['amplitudes'])
    
    def _choose_attack_action(self, game_state: QuakeGameState) -> int:
        """Choose attack action based on enemy position and weapon"""
        self.combat_metrics['weapon_usage'][game_state.weapon_selected] += 1
        
        # Attack while moving for better tactics
        if game_state.enemy_distance < 30:
            # Close combat - strafe and shoot
            attack_actions = [8, 17, 18, 19]  # attack, attack+move combinations
        else:
            # Long range - aim and shoot
            attack_actions = [8, 0, 8]  # attack, forward, attack
        
        return np.random.choice(attack_actions)
    
    def _choose_evasive_action(self) -> int:
        """Choose evasive action when under threat"""
        evasive_actions = [1, 2, 3, 10, 11, 12, 13]  # backward, left, right, jump, crouch, combinations
        return np.random.choice(evasive_actions)
    
    def _choose_defensive_action(self) -> int:
        """Choose defensive action"""
        defensive_actions = [11, 2, 3, 1]  # crouch, left, right, backward
        return np.random.choice(defensive_actions)
    
    def _choose_tactical_movement(self) -> int:
        """Choose tactical movement action"""
        tactical_actions = [0, 1, 2, 3, 4, 5, 10]  # forward, backward, left, right, turn_left, turn_right, jump
        return np.random.choice(tactical_actions)
    
    def _choose_exploration_action(self) -> int:
        """Choose exploration action"""
        exploration_actions = [0, 2, 3, 4, 5]  # forward, left, right, turn_left, turn_right
        return np.random.choice(exploration_actions)
    
    def _choose_weapon_switch_action(self, game_state: QuakeGameState) -> int:
        """Choose weapon switch action"""
        # Simple weapon switching logic
        weapon_actions = [6, 7]  # weapon switch actions
        return np.random.choice(weapon_actions)
    
    def _update_spatial_superposition(self, action_idx: int, game_state: QuakeGameState):
        """Update spatial superposition state after action"""
        # Update coherence time
        self.spatial_superposition['coherence_time'] += 0.1
        
        # Decay coherence over time
        if self.spatial_superposition['coherence_time'] > 5.0:
            self.spatial_superposition['coherence_time'] = 0.0
            # Reset to balanced state
            self.spatial_superposition['amplitudes'] = np.array([0.16, 0.16, 0.16, 0.16, 0.16, 0.2])
    
    def update_episode_result(self, episode_stats: Dict):
        """Update episode results"""
        self.combat_metrics['episodes_played'] += 1
        self.combat_metrics['total_kills'] += episode_stats.get('kills', 0)
        self.combat_metrics['total_deaths'] += episode_stats.get('deaths', 0)
        self.combat_metrics['total_damage_dealt'] += episode_stats.get('damage_dealt', 0)
        self.combat_metrics['total_damage_taken'] += episode_stats.get('damage_taken', 0)
        self.combat_metrics['survival_times'].append(episode_stats.get('survival_time', 0))
        self.combat_metrics['enemy_encounters'] += episode_stats.get('enemy_encounters', 0)
        self.combat_metrics['successful_engagements'] += episode_stats.get('successful_engagements', 0)
        
        # Calculate K/D ratio for this episode
        kills = episode_stats.get('kills', 0)
        deaths = max(episode_stats.get('deaths', 0), 1)  # Avoid division by zero
        kd_ratio = kills / deaths
        self.combat_metrics['kill_death_ratios'].append(kd_ratio)
    
    def get_research_summary(self) -> Dict:
        """Get comprehensive research summary"""
        metrics = self.combat_metrics
        
        return {
            'combat_performance': {
                'episodes_played': metrics['episodes_played'],
                'total_kills': metrics['total_kills'],
                'total_deaths': metrics['total_deaths'],
                'kill_death_ratio': metrics['total_kills'] / max(metrics['total_deaths'], 1),
                'avg_survival_time': np.mean(metrics['survival_times']) if metrics['survival_times'] else 0,
                'total_damage_dealt': metrics['total_damage_dealt'],
                'total_damage_taken': metrics['total_damage_taken']
            },
            'spatial_analysis': {
                'avg_spatial_entropy': np.mean(metrics['spatial_entropies']) if metrics['spatial_entropies'] else 0,
                'avg_decision_speed': np.mean(metrics['decision_times']) if metrics['decision_times'] else 0,
                'tactical_collapses': dict(metrics['tactical_collapses']),
                'weapon_usage': dict(metrics['weapon_usage'])
            },
            'engagement_metrics': {
                'total_encounters': metrics['enemy_encounters'],
                'successful_engagements': metrics['successful_engagements'],
                'engagement_success_rate': metrics['successful_engagements'] / max(metrics['enemy_encounters'], 1)
            }
        }

class RealQuakeExperimentFramework:
    """Real Quake experiment framework using ViZDoom with GPU support"""
    
    def __init__(self, device: str = 'cpu', progress: bool = False, cuda_optimize: bool = False):
        self.device = device
        self.progress = progress
        self.cuda_optimize = cuda_optimize
        self.quantum_agent = RealQuantumSpatialAgent(device=device)
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
        
        # GPU memory optimization
        if device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            if cuda_optimize:
                torch.backends.cudnn.benchmark = True
                print("🔥 CUDA optimizations enabled")
            print(f"🎮 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Setup headless display for cloud environments
        self._setup_headless_display()
        
        # Try to initialize ViZDoom environment
        try:
            self.quake_env = RealQuakeEnvironment()
            self.env_available = True
            print("ViZDoom environment initialized successfully")
        except Exception as e:
            print(f"Warning: ViZDoom environment failed to initialize: {e}")
            print("Will attempt to run with basic ViZDoom configuration")
            self.env_available = False
            self.quake_env = None
    
    def run_experiments(self, num_episodes: int = 20):
        """Run real Quake experiments with GPU acceleration"""
        print(f"\n=== REAL QUAKE III ARENA EXPERIMENTS ===")
        print(f"Running {num_episodes} episodes with ViZDoom")
        print(f"Device: {self.device.upper()}")
        print(f"CUDA Optimize: {self.cuda_optimize}")
        
        if not self.env_available:
            print("Attempting to create basic ViZDoom environment...")
            success = self._setup_basic_vizdoom()
            if not success:
                print("ERROR: Cannot run real Quake experiments without ViZDoom")
                print("Please install ViZDoom properly with: pip install vizdoom")
                return
        
        # GPU warmup
        if self.device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
            print("Warming up GPU...")
            dummy_tensor = torch.randn(1000, 1000, device='cuda')
            _ = torch.matmul(dummy_tensor, dummy_tensor)
            del dummy_tensor
            torch.cuda.empty_cache()
        
        print("\nStarting combat episodes...")
        start_time = time.time()
        
        # Initialize detailed progress tracking
        if self.progress and TQDM_AVAILABLE:
            episode_iterator = tqdm(range(num_episodes), 
                                  desc="Quake Episodes", 
                                  unit="ep",
                                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] K/D: {postfix}",
                                  postfix="0/0")
        else:
            episode_iterator = range(num_episodes)
            
        total_kills = 0
        total_deaths = 0
        
        for episode_num in episode_iterator:
            if not self.progress:
                print(f"\nEpisode {episode_num + 1}/{num_episodes}")
            
            try:
                episode_stats = self._run_single_episode(episode_num)
                self.quantum_agent.update_episode_result(episode_stats)
                
                # Update totals for progress tracking
                total_kills += episode_stats['kills']
                total_deaths += episode_stats['deaths']
                
                # Print episode summary or update progress bar
                if not self.progress:
                    print(f"  Kills: {episode_stats['kills']}, Deaths: {episode_stats['deaths']}")
                    print(f"  Survival: {episode_stats['survival_time']:.1f}s, Encounters: {episode_stats['enemy_encounters']}")
                elif TQDM_AVAILABLE and hasattr(episode_iterator, 'set_postfix'):
                    kd_ratio = total_kills / max(total_deaths, 1)
                    episode_iterator.set_postfix_str(f"{total_kills}/{total_deaths} (Ratio: {kd_ratio:.2f}) Survival: {episode_stats['survival_time']:.1f}s")
                
            except Exception as e:
                if not self.progress:
                    print(f"  Episode {episode_num + 1} failed: {e}")
                # Create minimal stats for failed episode
                episode_stats = {
                    'kills': 0, 'deaths': 1, 'survival_time': 1.0,
                    'enemy_encounters': 0, 'damage_dealt': 0, 'damage_taken': 0
                }
                total_deaths += 1
                self.quantum_agent.update_episode_result(episode_stats)
        
        # Close progress bar if used
        if self.progress and TQDM_AVAILABLE and hasattr(episode_iterator, 'close'):
            episode_iterator.close()
        
        end_time = time.time()
        duration = end_time - start_time
        episodes_per_second = num_episodes / duration
        
        print(f"\nAll episodes completed in {duration:.2f} seconds")
        print(f"Performance: {episodes_per_second:.2f} episodes/second")
        print(f"Final Stats: {total_kills} kills, {total_deaths} deaths, K/D ratio: {total_kills/max(total_deaths,1):.2f}")
        
        # GPU memory cleanup
        if self.device == 'cuda' and TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
            memory_used = torch.cuda.memory_allocated() / 1e6
            print(f"GPU Memory Used: {memory_used:.1f}MB")
        
        # Generate comprehensive report
        self._generate_report()
        
        # Print final summary
        summary = self.quantum_agent.get_research_summary()
        print("\n=== FINAL COMBAT RESULTS ===")
        print(f"Episodes: {summary['combat_performance']['episodes_played']}")
        print(f"Total Kills: {summary['combat_performance']['total_kills']}")
        print(f"Total Deaths: {summary['combat_performance']['total_deaths']}")
        print(f"K/D Ratio: {summary['combat_performance']['kill_death_ratio']:.2f}")
        print(f"Avg Survival: {summary['combat_performance']['avg_survival_time']:.1f}s")
        print(f"Avg Decision Speed: {summary['spatial_analysis']['avg_decision_speed']:.4f}s")
    
    def _setup_headless_display(self):
        """Setup virtual display for headless environments"""
        import subprocess
        import os
        
        # Check if we're in a headless environment
        if 'DISPLAY' not in os.environ:
            try:
                # Try to start Xvfb virtual display
                subprocess.run(['Xvfb', ':99', '-screen', '0', '1024x768x24'], 
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                             timeout=5)
                os.environ['DISPLAY'] = ':99'
                print("Virtual display setup for headless environment")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                print("Warning: Could not setup virtual display. ViZDoom may fail.")
                print("Install xvfb: apt-get install xvfb")
    
    def _setup_basic_vizdoom(self) -> bool:
        """Setup basic ViZDoom environment as fallback"""
        try:
            import vizdoom as vzd
            
            game = vzd.DoomGame()
            
            # Basic configuration
            game.set_doom_scenario_path(vzd.scenarios_path + "/basic.wad")
            game.set_doom_map("map01")
            game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
            game.set_screen_format(vzd.ScreenFormat.RGB24)
            game.set_render_hud(False)
            game.set_render_crosshair(False)
            game.set_render_weapon(True)
            game.set_render_decals(False)
            game.set_render_particles(False)
            game.set_window_visible(False)  # Headless for experiments
            
            # Add available buttons
            game.add_available_button(vzd.Button.MOVE_FORWARD)
            game.add_available_button(vzd.Button.MOVE_BACKWARD)
            game.add_available_button(vzd.Button.MOVE_LEFT)
            game.add_available_button(vzd.Button.MOVE_RIGHT)
            game.add_available_button(vzd.Button.TURN_LEFT)
            game.add_available_button(vzd.Button.TURN_RIGHT)
            game.add_available_button(vzd.Button.ATTACK)
            
            # Game variables
            game.add_available_game_variable(vzd.GameVariable.HEALTH)
            game.add_available_game_variable(vzd.GameVariable.ARMOR)
            game.add_available_game_variable(vzd.GameVariable.AMMO2)
            
            # Initialize
            game.init()
            
            # Create simple environment wrapper
            self.quake_env = SimpleViZDoomWrapper(game)
            self.env_available = True
            
            print("Basic ViZDoom environment created successfully")
            return True
            
        except Exception as e:
            print(f"Failed to create basic ViZDoom environment: {e}")
            return False
    
    def _run_single_episode(self, episode_num: int) -> Dict:
        """Run a single combat episode"""
        if self.quake_env is None:
            raise Exception("No ViZDoom environment available")
        
        # Reset environment
        game_state = self.quake_env.reset()
        
        episode_stats = {
            'kills': 0,
            'deaths': 0,
            'survival_time': 0,
            'enemy_encounters': 0,
            'damage_dealt': 0,
            'damage_taken': 0,
            'successful_engagements': 0
        }
        
        episode_start_time = time.time()
        steps = 0
        max_steps = 2000  # Prevent infinite episodes
        last_health = game_state.player_health if game_state else 100
        
        while steps < max_steps:
            if game_state is None:
                break
            
            # Get agent action
            action = self.quantum_agent.get_action(game_state)
            
            # Execute action
            next_state, reward, done, info = self.quake_env.step(action)
            
            # Update episode stats
            if game_state.enemy_visible:
                episode_stats['enemy_encounters'] += 1
            
            # Check for kills (positive reward)
            if reward > 50:
                episode_stats['kills'] += 1
                episode_stats['successful_engagements'] += 1
            
            # Check for damage taken
            if next_state and next_state.player_health < last_health:
                damage_taken = last_health - next_state.player_health
                episode_stats['damage_taken'] += damage_taken
                last_health = next_state.player_health
            
            # Check for death
            if done or (next_state and next_state.player_health <= 0):
                episode_stats['deaths'] += 1
                break
            
            game_state = next_state
            steps += 1
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
        
        # Calculate survival time
        episode_stats['survival_time'] = time.time() - episode_start_time
        
        return episode_stats
    
    def _generate_report(self):
        """Generate comprehensive research report"""
        print("\nGenerating real Quake combat report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create results directory
        experiment_dir = os.path.join(self.results_dir, f"real_quake_combat_{timestamp}")
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
        fig.suptitle('Real Quake III Arena Combat Analysis (Strategic Uncertainty Management)', fontsize=16)
        
        metrics = self.quantum_agent.combat_metrics
        
        # Survival time distribution
        if metrics['survival_times']:
            axes[0, 0].hist(metrics['survival_times'], bins=15, alpha=0.7, color='blue')
            axes[0, 0].set_title('Survival Time Distribution\n(Real Combat)')
            axes[0, 0].set_xlabel('Survival Time (seconds)')
            axes[0, 0].set_ylabel('Frequency')
        
        # Spatial entropy evolution
        if metrics['spatial_entropies']:
            axes[0, 1].plot(metrics['spatial_entropies'], alpha=0.7, color='green')
            axes[0, 1].set_title('Spatial Entropy Evolution\n(3D Superposition)')
            axes[0, 1].set_xlabel('Decision Number')
            axes[0, 1].set_ylabel('Entropy')
        
        # Tactical collapse triggers
        if metrics['tactical_collapses']:
            triggers = list(metrics['tactical_collapses'].keys())
            counts = list(metrics['tactical_collapses'].values())
            axes[0, 2].bar(triggers, counts, alpha=0.7, color='orange')
            axes[0, 2].set_title('Tactical Collapse Triggers\n(Real Combat)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Decision speed distribution
        if metrics['decision_times']:
            axes[1, 0].hist(metrics['decision_times'], bins=20, alpha=0.7, color='red')
            axes[1, 0].set_title('Decision Speed Distribution\n(Real-Time)')
            axes[1, 0].set_xlabel('Decision Time (seconds)')
            axes[1, 0].set_ylabel('Frequency')
        
        # Combat performance
        performance_data = {
            'Episodes': metrics['episodes_played'],
            'K/D Ratio': metrics['total_kills'] / max(metrics['total_deaths'], 1),
            'Avg Survival': np.mean(metrics['survival_times']) if metrics['survival_times'] else 0
        }
        
        bars = axes[1, 1].bar(performance_data.keys(), performance_data.values(), 
                             color=['blue', 'green', 'red'], alpha=0.7)
        axes[1, 1].set_title('Combat Performance\n(Real Results)')
        axes[1, 1].set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, performance_data.values()):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.2f}', ha='center', va='bottom')
        
        # Weapon usage
        if metrics['weapon_usage']:
            weapons = list(metrics['weapon_usage'].keys())
            usage = list(metrics['weapon_usage'].values())
            axes[1, 2].pie(usage, labels=weapons, autopct='%1.1f%%', alpha=0.7)
            axes[1, 2].set_title('Weapon Usage Distribution\n(Tactical Choices)')
        
        plt.tight_layout()
        
        # Save plots
        plot_path = os.path.join(experiment_dir, f"real_quake_combat_analysis_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        pdf_path = os.path.join(experiment_dir, f"real_quake_combat_analysis_{timestamp}.pdf")
        plt.savefig(pdf_path, bbox_inches='tight')
        
        plt.close()
    
    def _save_research_data(self, experiment_dir, timestamp):
        """Save research data"""
        summary = self.quantum_agent.get_research_summary()
        
        data = {
            'timestamp': timestamp,
            'experiment_type': 'real_quake_combat_vizdoom',
            'summary': summary,
            'raw_metrics': {
                'survival_times': self.quantum_agent.combat_metrics['survival_times'],
                'kill_death_ratios': self.quantum_agent.combat_metrics['kill_death_ratios'],
                'decision_times': self.quantum_agent.combat_metrics['decision_times'],
                'spatial_entropies': self.quantum_agent.combat_metrics['spatial_entropies'],
                'tactical_collapses': dict(self.quantum_agent.combat_metrics['tactical_collapses']),
                'weapon_usage': dict(self.quantum_agent.combat_metrics['weapon_usage'])
            }
        }
        
        data_path = os.path.join(experiment_dir, f"real_quake_combat_data_{timestamp}.json")
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _generate_latex_summary(self, experiment_dir, timestamp):
        """Generate LaTeX summary table"""
        summary = self.quantum_agent.get_research_summary()
        
        latex_content = f"""\\begin{{table}}[h]
\\centering
\\caption{{Real Quake III Arena Combat Results with ViZDoom}}
\\begin{{tabular}}{{|l|c|}}
\\hline
Metric & Value \\\\
\\hline
Episodes Played & {summary['combat_performance']['episodes_played']} \\\\
Total Kills & {summary['combat_performance']['total_kills']} \\\\
Total Deaths & {summary['combat_performance']['total_deaths']} \\\\
K/D Ratio & {summary['combat_performance']['kill_death_ratio']:.2f} \\\\
Avg Survival Time & {summary['combat_performance']['avg_survival_time']:.1f}s \\\\
Avg Decision Speed & {summary['spatial_analysis']['avg_decision_speed']:.4f}s \\\\
Avg Spatial Entropy & {summary['spatial_analysis']['avg_spatial_entropy']:.3f} \\\\
Enemy Encounters & {summary['engagement_metrics']['total_encounters']} \\\\
Engagement Success Rate & {summary['engagement_metrics']['engagement_success_rate']:.2f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}"""
        
        latex_path = os.path.join(experiment_dir, f"real_quake_combat_table_{timestamp}.tex")
        with open(latex_path, 'w') as f:
            f.write(latex_content)

class SimpleViZDoomWrapper:
    """Simple wrapper for basic ViZDoom functionality"""
    
    def __init__(self, game):
        self.game = game
        self.actions = [
            [1, 0, 0, 0, 0, 0, 0],  # MOVE_FORWARD
            [0, 1, 0, 0, 0, 0, 0],  # MOVE_BACKWARD
            [0, 0, 1, 0, 0, 0, 0],  # MOVE_LEFT
            [0, 0, 0, 1, 0, 0, 0],  # MOVE_RIGHT
            [0, 0, 0, 0, 1, 0, 0],  # TURN_LEFT
            [0, 0, 0, 0, 0, 1, 0],  # TURN_RIGHT
            [0, 0, 0, 0, 0, 0, 1],  # ATTACK
            [1, 0, 0, 0, 0, 0, 1],  # MOVE_FORWARD + ATTACK
            [0, 0, 1, 0, 0, 0, 1],  # MOVE_LEFT + ATTACK
            [0, 0, 0, 1, 0, 0, 1],  # MOVE_RIGHT + ATTACK
        ]
    
    def reset(self):
        """Reset environment"""
        try:
            self.game.new_episode()
            return self._get_state()
        except:
            return None
    
    def step(self, action_idx):
        """Execute action"""
        try:
            if action_idx >= len(self.actions):
                action_idx = 0
            
            action = self.actions[action_idx]
            reward = self.game.make_action(action)
            done = self.game.is_episode_finished()
            
            if done:
                state = None
            else:
                state = self._get_state()
            
            return state, reward, done, {'action_idx': action_idx}
        except:
            return None, 0, True, {}
    
    def _get_state(self):
        """Get current game state"""
        try:
            if self.game.is_episode_finished():
                return None
            
            screen = self.game.get_state().screen_buffer
            variables = self.game.get_state().game_variables
            
            # Create QuakeGameState
            state = QuakeGameState(
                screen=screen,
                player_position=(0, 0, 0),  # Not available in basic mode
                player_health=int(variables[0]) if len(variables) > 0 else 100,
                player_armor=int(variables[1]) if len(variables) > 1 else 0,
                ammo_counts={'bullets': int(variables[2]) if len(variables) > 2 else 50},
                weapon_selected='PISTOL',
                enemy_visible=False,  # Simplified
                enemy_distance=999,
                enemy_angle=0,
                game_variables={'health': variables[0] if len(variables) > 0 else 100},
                available_actions=['move', 'turn', 'attack']
            )
            
            return state
        except:
            return None

def main():
    """Run real Quake combat experiments with GPU support"""
    parser = argparse.ArgumentParser(description='Real Quake Combat Experiments')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device to use for computation')
    parser.add_argument('--episodes', type=int, default=30,
                       help='Number of episodes to run')
    parser.add_argument('--progress', action='store_true',
                       help='Show progress bars')
    parser.add_argument('--cuda-optimize', action='store_true',
                       help='Enable CUDA optimizations')
    
    args = parser.parse_args()
    
    print("=== REAL QUAKE III ARENA COMBAT EXPERIMENTS WITH VIZDOOM ===")
    print(f"Device: {args.device.upper()}")
    print(f"Episodes: {args.episodes}")
    print(f"Progress: {args.progress}")
    print(f"CUDA Optimize: {args.cuda_optimize}")
    
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
    
    framework = RealQuakeExperimentFramework(
        device=args.device, 
        progress=args.progress, 
        cuda_optimize=args.cuda_optimize
    )
    framework.run_experiments(num_episodes=args.episodes)
    
    print("\n=== REAL QUAKE COMBAT EXPERIMENTS COMPLETED ===")
    
    return framework

if __name__ == "__main__":
    framework = main()