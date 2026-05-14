"""
train.py - PPO Training Script for the Double Inverted Pendulum Environment.

Usage:
    python train.py --timesteps 200000 --reward_type shaped --save_path models/ppo_shaped.zip
    python train.py --timesteps 200000 --reward_type baseline --save_path models/ppo_baseline.zip
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import DoublePendulumEnv


def main():
    parser = argparse.ArgumentParser(description="Train a PPO agent on DoublePendulumEnv.")
    parser.add_argument("--timesteps", type=int, default=200000,
                        help="Total training timesteps.")
    parser.add_argument("--reward_type", type=str, default="shaped",
                        choices=["baseline", "shaped"],
                        help="Reward function to use: 'baseline' or 'shaped'.")
    parser.add_argument("--save_path", type=str, default="models/ppo_model.zip",
                        help="File path to save the trained model.")
    args = parser.parse_args()

    # Create output directories
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else "models", exist_ok=True)
    log_dir = os.path.join("logs", args.reward_type)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[train.py] Starting training:")
    print(f"  reward_type : {args.reward_type}")
    print(f"  timesteps   : {args.timesteps}")
    print(f"  save_path   : {args.save_path}")
    print(f"  log_dir     : {log_dir}")

    # Build the environment: Monitor wraps it for logging ep rewards/lengths
    def make_env():
        env = DoublePendulumEnv(reward_type=args.reward_type)
        env = Monitor(env, log_dir)
        return env

    vec_env = DummyVecEnv([make_env])

    # Create the PPO model
    model = PPO("MlpPolicy", vec_env, verbose=1)

    # Train
    model.learn(total_timesteps=args.timesteps)

    # Save the trained model
    model.save(args.save_path)
    print(f"[train.py] Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
