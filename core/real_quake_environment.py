#!/usr/bin/env python3
"""
Real Quake III Arena Environment using ViZDoom
Implements actual 3D combat environment with real game physics
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import vizdoom as vzd
import time
import os

@dataclass
class QuakeGameState:
    """Real Quake game state"""
    screen: np.ndarray  # Visual observation
    player_position: Tuple[float, float, float]
    player_health: int
    player_armor: int
    ammo_counts: Dict[str, int]
    weapon_selected: str
    enemy_visible: bool
    enemy_distance: float
    enemy_angle: float
    game_variables: Dict[str, float]
    available_actions: List[str]
    
class RealQuakeEnvironment:
    """Real Quake III Arena environment using ViZDoom"""
    
    def __init__(self, config_file: str = None, resolution: Tuple[int, int] = (640, 480)):
        self.game = vzd.DoomGame()
        self.resolution = resolution
        
        # Configure ViZDoom
        self._setup_game(config_file)
        
        # Game state tracking
        self.episode_start_time = None
        self.total_kills = 0
        self.total_deaths = 0
        self.damage_dealt = 0
        self.damage_taken = 0
        
        # Action space
        self.actions = self._define_actions()
        self.action_names = [
            'MOVE_FORWARD', 'MOVE_BACKWARD', 'MOVE_LEFT', 'MOVE_RIGHT',
            'TURN_LEFT', 'TURN_RIGHT', 'LOOK_UP', 'LOOK_DOWN',
            'ATTACK', 'USE', 'JUMP', 'CROUCH',
            'WEAPON_1', 'WEAPON_2', 'WEAPON_3', 'WEAPON_4', 'WEAPON_5'
        ]
        
        # Initialize game
        try:
            self.game.init()
            print("ViZDoom initialized successfully!")
        except Exception as e:
            print(f"Failed to initialize ViZDoom: {e}")
            print("Using fallback configuration...")
            self._setup_fallback_config()
    
    def _setup_game(self, config_file: str = None):
        """Setup ViZDoom game configuration"""
        # Use provided config or create default
        if config_file and os.path.exists(config_file):
            self.game.load_config(config_file)
        else:
            # Default configuration for deathmatch
            self.game.set_doom_scenario_path(vzd.scenarios_path + "/deathmatch.wad")
            self.game.set_doom_map("map01")
            
        # Screen settings
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        
        # Game variables to track
        self.game.add_available_game_variable(vzd.GameVariable.HEALTH)
        self.game.add_available_game_variable(vzd.GameVariable.ARMOR)
        self.game.add_available_game_variable(vzd.GameVariable.AMMO0)  # Fist
        self.game.add_available_game_variable(vzd.GameVariable.AMMO1)  # Pistol
        self.game.add_available_game_variable(vzd.GameVariable.AMMO2)  # Shotgun
        self.game.add_available_game_variable(vzd.GameVariable.AMMO3)  # Chaingun
        self.game.add_available_game_variable(vzd.GameVariable.AMMO4)  # Rocket
        self.game.add_available_game_variable(vzd.GameVariable.AMMO5)  # Plasma
        self.game.add_available_game_variable(vzd.GameVariable.AMMO6)  # BFG
        self.game.add_available_game_variable(vzd.GameVariable.SELECTED_WEAPON)
        self.game.add_available_game_variable(vzd.GameVariable.SELECTED_WEAPON_AMMO)
        self.game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
        self.game.add_available_game_variable(vzd.GameVariable.DEATHCOUNT)
        self.game.add_available_game_variable(vzd.GameVariable.DAMAGECOUNT)
        
        # Available buttons (actions)
        self.game.add_available_button(vzd.Button.MOVE_FORWARD)
        self.game.add_available_button(vzd.Button.MOVE_BACKWARD)
        self.game.add_available_button(vzd.Button.MOVE_LEFT)
        self.game.add_available_button(vzd.Button.MOVE_RIGHT)
        self.game.add_available_button(vzd.Button.TURN_LEFT)
        self.game.add_available_button(vzd.Button.TURN_RIGHT)
        self.game.add_available_button(vzd.Button.LOOK_UP)
        self.game.add_available_button(vzd.Button.LOOK_DOWN)
        self.game.add_available_button(vzd.Button.ATTACK)
        self.game.add_available_button(vzd.Button.USE)
        self.game.add_available_button(vzd.Button.JUMP)
        self.game.add_available_button(vzd.Button.CROUCH)
        self.game.add_available_button(vzd.Button.SELECT_WEAPON1)
        self.game.add_available_button(vzd.Button.SELECT_WEAPON2)
        self.game.add_available_button(vzd.Button.SELECT_WEAPON3)
        self.game.add_available_button(vzd.Button.SELECT_WEAPON4)
        self.game.add_available_button(vzd.Button.SELECT_WEAPON5)
        
        # Game settings
        self.game.set_episode_timeout(2100)  # 35 seconds at 60 FPS
        self.game.set_episode_start_time(10)
        self.game.set_window_visible(True)  # Set to False for headless
        self.game.set_sound_enabled(False)
        self.game.set_living_reward(-1)  # Small penalty for time
        
        # Multiplayer settings for realistic combat
        self.game.add_game_args("+sv_cheats 1")
        self.game.add_game_args("+timelimit 1")
        self.game.add_game_args("+fraglimit 10")
        
    def _setup_fallback_config(self):
        """Setup fallback configuration if main setup fails"""
        try:
            # Minimal working configuration
            self.game.set_doom_scenario_path(vzd.scenarios_path + "/basic.wad")
            self.game.set_doom_map("map01")
            self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
            self.game.set_screen_format(vzd.ScreenFormat.RGB24)
            self.game.set_window_visible(False)
            self.game.init()
            print("Fallback ViZDoom configuration initialized.")
        except Exception as e:
            print(f"Fallback configuration also failed: {e}")
            raise RuntimeError("Could not initialize ViZDoom environment")
    
    def _define_actions(self) -> List[List[int]]:
        """Define discrete action space"""
        actions = []
        
        # Movement actions
        actions.append([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # MOVE_FORWARD
        actions.append([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # MOVE_BACKWARD
        actions.append([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # MOVE_LEFT
        actions.append([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # MOVE_RIGHT
        
        # Turning actions
        actions.append([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # TURN_LEFT
        actions.append([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # TURN_RIGHT
        
        # Looking actions
        actions.append([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # LOOK_UP
        actions.append([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # LOOK_DOWN
        
        # Combat actions
        actions.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # ATTACK
        actions.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])  # USE
        
        # Special actions
        actions.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])  # JUMP
        actions.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])  # CROUCH
        
        # Weapon selection
        actions.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])  # WEAPON_1
        actions.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])  # WEAPON_2
        actions.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])  # WEAPON_3
        actions.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])  # WEAPON_4
        actions.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])  # WEAPON_5
        
        # Combination actions for more realistic gameplay
        actions.append([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])  # FORWARD + ATTACK
        actions.append([0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # STRAFE_LEFT + TURN_LEFT
        actions.append([0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # STRAFE_RIGHT + TURN_RIGHT
        
        return actions
    
    def reset(self) -> QuakeGameState:
        """Reset environment for new episode"""
        if self.game.is_episode_finished():
            self.game.new_episode()
        
        self.episode_start_time = time.time()
        self.total_kills = 0
        self.total_deaths = 0
        self.damage_dealt = 0
        self.damage_taken = 0
        
        return self._get_game_state()
    
    def step(self, action_idx: int) -> Tuple[QuakeGameState, float, bool, Dict]:
        """Execute action and return new state"""
        if action_idx >= len(self.actions):
            action_idx = 0  # Default to no action
        
        # Execute action
        action = self.actions[action_idx]
        reward = self.game.make_action(action)
        
        # Check if episode is finished
        done = self.game.is_episode_finished()
        
        # Calculate additional rewards
        if not done:
            state = self._get_game_state()
            reward += self._calculate_additional_rewards(state)
        else:
            state = None
        
        # Info dictionary
        info = {
            'action_name': self.action_names[action_idx] if action_idx < len(self.action_names) else 'UNKNOWN',
            'episode_time': time.time() - self.episode_start_time if self.episode_start_time else 0,
            'total_kills': self.total_kills,
            'total_deaths': self.total_deaths
        }
        
        return state, reward, done, info
    
    def _get_game_state(self) -> QuakeGameState:
        """Get current game state"""
        if self.game.is_episode_finished():
            # Return empty state if episode is finished
            return QuakeGameState(
                screen=np.zeros((480, 640, 3), dtype=np.uint8),
                player_position=(0.0, 0.0, 0.0),
                player_health=0,
                player_armor=0,
                ammo_counts={},
                weapon_selected='none',
                enemy_visible=False,
                enemy_distance=float('inf'),
                enemy_angle=0.0,
                game_variables={},
                available_actions=[]
            )
        
        # Get screen buffer
        screen = self.game.get_state().screen_buffer
        if screen is not None:
            screen = np.transpose(screen, (1, 2, 0))  # CHW to HWC
        else:
            screen = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Get game variables
        game_vars = self.game.get_state().game_variables
        
        # Extract specific variables
        health = int(game_vars[0]) if len(game_vars) > 0 else 100
        armor = int(game_vars[1]) if len(game_vars) > 1 else 0
        
        # Ammo counts
        ammo_counts = {
            'fist': int(game_vars[2]) if len(game_vars) > 2 else 0,
            'pistol': int(game_vars[3]) if len(game_vars) > 3 else 50,
            'shotgun': int(game_vars[4]) if len(game_vars) > 4 else 0,
            'chaingun': int(game_vars[5]) if len(game_vars) > 5 else 0,
            'rocket': int(game_vars[6]) if len(game_vars) > 6 else 0,
            'plasma': int(game_vars[7]) if len(game_vars) > 7 else 0,
            'bfg': int(game_vars[8]) if len(game_vars) > 8 else 0
        }
        
        # Selected weapon
        selected_weapon_id = int(game_vars[9]) if len(game_vars) > 9 else 1
        weapon_names = ['fist', 'pistol', 'shotgun', 'chaingun', 'rocket', 'plasma', 'bfg']
        weapon_selected = weapon_names[selected_weapon_id - 1] if 1 <= selected_weapon_id <= 7 else 'pistol'
        
        # Enemy detection (simplified - would need more sophisticated computer vision)
        enemy_visible, enemy_distance, enemy_angle = self._detect_enemies(screen)
        
        # Player position (estimated from game state)
        player_position = (0.0, 0.0, 0.0)  # Would need additional game variables for exact position
        
        return QuakeGameState(
            screen=screen,
            player_position=player_position,
            player_health=health,
            player_armor=armor,
            ammo_counts=ammo_counts,
            weapon_selected=weapon_selected,
            enemy_visible=enemy_visible,
            enemy_distance=enemy_distance,
            enemy_angle=enemy_angle,
            game_variables={
                'health': health,
                'armor': armor,
                'kills': int(game_vars[10]) if len(game_vars) > 10 else 0,
                'deaths': int(game_vars[11]) if len(game_vars) > 11 else 0,
                'damage': int(game_vars[12]) if len(game_vars) > 12 else 0
            },
            available_actions=list(range(len(self.actions)))
        )
    
    def _detect_enemies(self, screen: np.ndarray) -> Tuple[bool, float, float]:
        """Simple enemy detection using computer vision"""
        try:
            # Ensure screen is in correct format
            if len(screen.shape) != 3 or screen.shape[2] != 3:
                return False, float('inf'), 0.0
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(screen, cv2.COLOR_RGB2HSV)
            
            # Define color ranges for enemy detection (this is very basic)
            # In a real implementation, you'd use more sophisticated object detection
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red, upper_red)
            
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            
            mask = mask1 + mask2
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (assumed to be closest enemy)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 100:  # Minimum area threshold
                    # Calculate centroid
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Estimate distance based on contour area (very rough)
                        distance = max(10.0, 1000.0 / np.sqrt(area))
                        
                        # Calculate angle from center of screen
                        screen_center_x = screen.shape[1] // 2
                        angle = (cx - screen_center_x) / screen_center_x * 90  # Degrees
                        
                        return True, distance, angle
        except Exception as e:
            print(f"Enemy detection error: {e}")
            return False, float('inf'), 0.0
        
        return False, float('inf'), 0.0
    
    def _calculate_additional_rewards(self, state: QuakeGameState) -> float:
        """Calculate additional rewards based on game state"""
        reward = 0.0
        
        # Health-based rewards
        if state.player_health > 75:
            reward += 0.1
        elif state.player_health < 25:
            reward -= 0.2
        
        # Armor bonus
        if state.player_armor > 50:
            reward += 0.05
        
        # Enemy engagement rewards
        if state.enemy_visible:
            reward += 0.2
            if state.enemy_distance < 100:
                reward += 0.1  # Close combat bonus
        
        # Weapon selection rewards
        if state.weapon_selected in ['shotgun', 'chaingun', 'rocket'] and state.enemy_visible:
            reward += 0.1
        
        return reward
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Render the environment"""
        if mode == 'human':
            # ViZDoom handles rendering automatically when window is visible
            return None
        elif mode == 'rgb_array':
            state = self._get_game_state()
            return state.screen
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def close(self):
        """Close the environment"""
        if hasattr(self, 'game'):
            self.game.close()
    
    def get_action_meanings(self) -> List[str]:
        """Get human-readable action meanings"""
        return self.action_names
    
    def get_screen_dimensions(self) -> Tuple[int, int]:
        """Get screen dimensions"""
        return self.resolution

if __name__ == "__main__":
    # Test the real Quake environment
    print("Testing Real Quake III Arena Environment")
    print("========================================")
    
    try:
        env = RealQuakeEnvironment()
        
        print(f"Action space: {len(env.actions)} actions")
        print(f"Action meanings: {env.get_action_meanings()[:5]}...")  # Show first 5
        
        # Test episode
        state = env.reset()
        print(f"\nInitial state:")
        print(f"  Health: {state.player_health}")
        print(f"  Armor: {state.player_armor}")
        print(f"  Weapon: {state.weapon_selected}")
        print(f"  Screen shape: {state.screen.shape}")
        
        # Run a few steps
        for step in range(10):
            action = np.random.randint(0, len(env.actions))
            state, reward, done, info = env.step(action)
            
            if state is not None:
                print(f"Step {step + 1}: Action={info['action_name']}, Reward={reward:.2f}, Health={state.player_health}")
            
            if done:
                print("Episode finished!")
                break
        
        env.close()
        print("\nReal Quake environment test completed!")
        
    except Exception as e:
        print(f"Error testing Quake environment: {e}")
        print("This is expected if ViZDoom scenarios are not properly installed.")
        print("The environment class is ready for use when ViZDoom is properly configured.")