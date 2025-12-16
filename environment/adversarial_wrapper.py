import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

class AdversarialLanderWrapper(gym.Wrapper):
    def __init__(self, env, max_wind_force=5.0, max_budget=100.0, visible_wind=False):
        super().__init__(env)
        self.env = env
        
        # Configuration
        self.max_wind_force = max_wind_force
        self.max_budget = max_budget
        self.current_budget = max_budget
        self.visible_wind = visible_wind
        
        # State
        self.opponent_model = None 
        self.mode = "protagonist" 
        self.current_wind = np.zeros(2, dtype=np.float32)

        # Action Spaces
        self.lander_action_space = env.action_space
        self.wind_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # --- OBSERVATION SPACE UPDATE ---
        env_low = self.env.observation_space.low.astype(np.float32)
        env_high = self.env.observation_space.high.astype(np.float32)

        extra_dims = 1 + (2 if visible_wind else 0) # +1 for Budget
        
        low = np.concatenate([env_low, [0.0] * extra_dims]).astype(np.float32)
        high = np.concatenate([env_high, [1.0] * extra_dims]).astype(np.float32) # Budget is normalized 0-1
        
        if visible_wind:
            high[-2:] = float(max_wind_force) 
            low[-2:] = float(-max_wind_force)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def set_mode(self, mode):
        assert mode in ["protagonist", "adversary"], "Invalid mode"
        self.mode = mode
        self.action_space = self.lander_action_space if mode == "protagonist" else self.wind_action_space

    def set_opponent(self, model_or_path):
        if isinstance(model_or_path, str):
            # Multi CPU
            self.opponent_model = PPO.load(model_or_path, device="cpu")
        else:
            # Single core
            self.opponent_model = model_or_path

    def _get_obs(self, obs):
        """Constructs the observation vector: [State, Budget, (Wind)]"""
        # Normalize budget to 0.0 - 1.0 range so the neural net handles it better
        norm_budget = np.array([self.current_budget / self.max_budget], dtype=np.float32)
        
        # 1. Base State + Budget
        enhanced_obs = np.concatenate([obs, norm_budget])
        
        # 2. (Optional) + Wind Vector
        if self.visible_wind:
            enhanced_obs = np.concatenate([enhanced_obs, self.current_wind])
            
        return enhanced_obs.astype(np.float32)

    def reset(self, **kwargs):
        self.current_budget = self.max_budget
        self.current_wind = np.zeros(2, dtype=np.float32)
        
        obs, info = self.env.reset(**kwargs)
        self.last_obs = self._get_obs(obs)
        return self.last_obs, info

    def step(self, action):
        lander_action = None
        wind_raw = np.array([0.0, 0.0])

        if self.mode == "protagonist":
            lander_action = action
            if self.opponent_model and self.current_budget > 0:
                wind_raw, _ = self.opponent_model.predict(self.last_obs, deterministic=True)
                
        elif self.mode == "adversary":
            wind_raw = np.array(action, dtype=np.float32)
            if self.opponent_model:
                lander_action, _ = self.opponent_model.predict(self.last_obs, deterministic=True)
            else:
                lander_action = 0 

        # Calculate Wind & Budget
        wind_vector = wind_raw * self.max_wind_force
        
        # --- NON-LINEAR COST ---
        # Small puffs are cheap. Big blasts are expensive.
        # This encourages "efficiency" rather than just dumping.
        force_magnitude = np.linalg.norm(wind_vector)
        cost = (force_magnitude * 0.05) + (force_magnitude**2 * 0.005)
        
        self.current_budget -= cost
        if self.current_budget <= 0:
            self.current_budget = 0.0
            wind_vector = np.zeros(2)

        self.current_wind = wind_vector

        # Apply Physics
        try:
            lander = self.env.unwrapped.lander
            if lander:
                lander.ApplyForceToCenter((float(wind_vector[0]), float(wind_vector[1])), True)
        except AttributeError:
            pass

        # Step
        next_obs_raw, reward, terminated, truncated, info = self.env.step(lander_action)
        
        # Get Enhanced Obs
        next_obs = self._get_obs(next_obs_raw)
        self.last_obs = next_obs
        
        if self.mode == "adversary":
            reward = -reward 
            # Add penalty for using fuel (incentivize saving)
            reward -= (cost * 0.5) 

        info["budget"] = self.current_budget
        return next_obs, reward, terminated, truncated, info