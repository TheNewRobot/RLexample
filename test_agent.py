"""
To run the following script with a specific experiment you can use this command (example)
    - python test_agent.py CartPole-v0 --experiment 20250112_192941
"""

import argparse
import gymnasium as gym
import numpy as np
import json
import os
from _policies import BinaryActionLinearPolicy

def load_policy(env_name, experiment=None):

    log_dir = os.path.join('logs', env_name)

    if experiment:
        log_path = os.path.join(log_dir,experiment,'best_experiment.json')
        if not os.path.exists(log_path):
            raise FileNotFoundError(f"No experiment found with ID {experiment}")
    else:
        experiment_files = [f for f in os.listdir(log_dir)]
        if not experiment_files:
            raise FileNotFoundError(f"No experiments found for {env_name}")
        latest_file = max(experiment_files)
        log_path = os.path.join(log_dir, latest_file, 'best_experiment.json')

    with open(log_path, 'r') as f:
        experiment_data = json.load(f)
    
    print(f"Loaded experiment from: {log_path}")
    print(f"Experiment timestamp: {experiment_data['timestamp']}")
    print(f"Achieved reward: {experiment_data['best_reward']}")
    
    return BinaryActionLinearPolicy(experiment_data['theta_params'])

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    parser.add_argument('game', nargs="?", default="CartPole-v0")
    parser.add_argument('--episodes',  type=int, default=20)
    parser.add_argument('--max_steps', type=int, default=200)
    parser.add_argument('--experiment', type=str, 
                        help='Specific experiment. If not specified, loads the latest experiment.')
    args = parser.parse_args()

    # 1. Create the environment
    env = gym.make(args.game, render_mode='human')

    # 2. Load the trained agent
    try:
        agent = load_policy(args.game, experiment=args.experiment)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)

    # 3. Run episodes with the trained policy
    episode_rewards = []

    for i_episode in range(args.episodes):
        observation, _ = env.reset()
        episode_reward = 0

        for t in range(args.max_steps):
            env.render()
            action = agent.act(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            
            done = np.logical_or(terminated, truncated)
            episode_reward += reward 

            if done:
                break
    
        episode_rewards.append(episode_reward)
        print(f'###################################')
        print(f'\nEpisode {i_episode+1} finished after {t+1} steps')
        print(f'Total episode reward: {episode_reward}\n')
    
    # Print final statistics
    print(f'###################################')
    print("\nExperiment Summary:")
    print(f"Average reward over {args.episodes} episodes: {np.mean(episode_rewards):.2f}")
    print(f"Standard deviation: {np.std(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")

    env.close()