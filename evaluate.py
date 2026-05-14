"""
evaluate.py - Evaluation & GIF Generation Script for the Double Inverted Pendulum Environment.

Usage:
    # Run with visual window (requires a display):
    python evaluate.py --model_path models/ppo_shaped.zip

    # Headless evaluation + save GIF:
    python evaluate.py --model_path models/ppo_shaped.zip --gif_path media/agent_final.gif --steps 500
"""

import argparse
import os
import sys

import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from environment import DoublePendulumEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO model on DoublePendulumEnv.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to a saved .zip model file.")
    parser.add_argument("--gif_path", type=str, default=None,
                        help="If provided, save a GIF of the evaluation to this path.")
    parser.add_argument("--steps", type=int, default=500,
                        help="Number of evaluation steps to run.")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"[evaluate.py] ERROR: Model file not found: {args.model_path}")
        sys.exit(1)

    # Choose render mode based on whether we're saving a gif or showing a live window
    render_mode = "rgb_array" if args.gif_path else "human"

    print(f"[evaluate.py] Loading model: {args.model_path}")
    print(f"[evaluate.py] Render mode  : {render_mode}")
    print(f"[evaluate.py] Steps        : {args.steps}")

    env = DoublePendulumEnv(reward_type="shaped", render_mode=render_mode)
    vec_env = DummyVecEnv([lambda: env])

    model = PPO.load(args.model_path, env=vec_env)

    obs = vec_env.reset()
    images = []
    total_reward = 0.0

    for step in range(args.steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += float(reward[0])

        if render_mode == "rgb_array":
            img = vec_env.envs[0].render()
            if img is not None:
                images.append(img)
        else:
            vec_env.envs[0].render()

        if done[0]:
            obs = vec_env.reset()

    print(f"[evaluate.py] Evaluation complete. Total reward: {total_reward:.2f}")

    if args.gif_path and images:
        gif_dir = os.path.dirname(args.gif_path)
        if gif_dir:
            os.makedirs(gif_dir, exist_ok=True)
        imageio.mimsave(args.gif_path, images, fps=60)
        print(f"[evaluate.py] GIF saved to {args.gif_path}")

    vec_env.close()


if __name__ == "__main__":
    main()
