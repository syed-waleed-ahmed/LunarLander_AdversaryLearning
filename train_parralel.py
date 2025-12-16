import gymnasium as gym
import os
import multiprocessing
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from environment.adversarial_wrapper import AdversarialLanderWrapper

# --- CONFIGURATION ---
ROUNDS = 10              
STEPS_PER_ROUND = 200_000  # Increased steps because we generate data much faster now!
MODELS_DIR = "models_parallel"
LOG_DIR = "logs_parallel"
VISIBLE_WIND = False
MAX_WIND_FORCE = 10.0

# Number of parallel games (Leave 1-2 cores free for the OS)
# If server has 32 cores, set this to 24 or 30.
NUM_CORES = multiprocessing.cpu_count() - 2 

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- FACTORY FUNCTION ---
# This function is needed to create a fresh environment copy for each CPU core
def make_env(rank: int, seed: int = 0):
    def _init():
        env = gym.make("LunarLander-v3", render_mode=None)
        # Use a separate log file for each core (0.monitor.csv, 1.monitor.csv...)
        env = Monitor(env, os.path.join(LOG_DIR, str(rank)))
        env = AdversarialLanderWrapper(
            env, 
            max_wind_force=MAX_WIND_FORCE, 
            visible_wind=VISIBLE_WIND
        )
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    print(f"Detected {multiprocessing.cpu_count()} CPUs. Launching {NUM_CORES} parallel environments...")
    
    # 1. Create the Vectorized Environment
    # SubprocVecEnv runs each env in a separate process
    env = SubprocVecEnv([make_env(i) for i in range(NUM_CORES)])

    # 2. Initialize Agents
    print("Initializing Agents...")
    
    # Init Protagonist
    # We use env_method to call 'set_mode' on all 32 copies of the environment at once
    env.env_method("set_mode", "protagonist")
    protagonist = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR, batch_size=64*NUM_CORES)
    
    # Init Adversary
    env.env_method("set_mode", "adversary")
    adversary = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR, batch_size=64*NUM_CORES)

    print("Agents Initialized. Starting Parallel Training Loop...")

    # 3. Training Loop
    for round_idx in range(1, ROUNDS + 1):
        print(f"\n{'='*20} ROUND {round_idx} / {ROUNDS} {'='*20}")

        # Curriculum: Update Budget on all cores
        current_budget = min(20.0 * round_idx, 100.0)
        # We manually set the attribute on all parallel environments
        env.set_attr("max_budget", current_budget)
        print(f"Adversary Budget set to: {current_budget:.1f}")

        # --- PHASE A: TRAIN PROTAGONIST ---
        print(f"> Training Protagonist (vs Fixed Adversary)...")
        
        # 1. Save Adversary to a temp file
        adversary.save("temp_adversary_for_workers")
        
        # 2. Tell workers to load this file
        env.env_method("set_mode", "protagonist")
        env.env_method("set_opponent", "temp_adversary_for_workers.zip") # Pass STRING, not OBJECT
        
        # 3. Train
        protagonist.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, tb_log_name="PPO_Protagonist")
        protagonist.save(f"{MODELS_DIR}/protagonist_round_{round_idx}")

        # --- PHASE B: TRAIN ADVERSARY ---
        print(f"> Training Adversary (vs Fixed Protagonist)...")
        
        # 1. Save Protagonist to a temp file
        protagonist.save("temp_protagonist_for_workers")
        
        # 2. Tell workers to load this file
        env.env_method("set_mode", "adversary")
        env.env_method("set_opponent", "temp_protagonist_for_workers.zip") # Pass STRING, not OBJECT
        
        # 3. Train
        adversary.learn(total_timesteps=STEPS_PER_ROUND, reset_num_timesteps=False, tb_log_name="PPO_Adversary")
        adversary.save(f"{MODELS_DIR}/adversary_round_{round_idx}")

    print("\nPARALLEL TRAINING COMPLETE!")
    env.close()

if __name__ == "__main__":
    # This check is MANDATORY for multiprocessing on Windows/Linux
    main()