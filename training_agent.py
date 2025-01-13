from __future__ import print_function

import argparse
import gymnasium as gym
import numpy as np
import json 
import os
from datetime import datetime
from _policies import BinaryActionLinearPolicy
from  my_learning_agent import cem, do_rollout

def save_experiment(env_name, it, iteration_data, theta, reward, timestamp):
    log_dir = f"logs/{env_name}/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    experiment_data = {
        'environment' : env_name,
        'timestamp': timestamp,
        'iteration': it,
        'best_reward': reward,
        'theta_params': theta.tolist(),
        'mean_reward': iteration_data['y_mean'],
        'theta_mean': iteration_data['theta_mean'].tolist()
    }

    best_exp_file = os.path.join(log_dir, 'best_experiment.json' )\
    
    should_save = True
    if os.path.exists(best_exp_file):
        with open(best_exp_file, 'r') as f:
            previous_best = json.load(f)
            if previous_best['best_reward'] >= reward:
                should_save = False
    
    # Save if it's the best so far
    if should_save:
        with open(best_exp_file, 'w') as f:
            json.dump(experiment_data, f, indent=4)
        print(f"Saved new best experiment with reward: {reward}")
        return True
    return False

def train_environment(env_name, params, num_steps, save_experiment_log):
    # 1. Create the environment
    env = gym.make(env_name, render_mode ="human")
    np.random.seed(0)

    def noisy_evaluation(theta):
        agent = BinaryActionLinearPolicy(theta)
        rew, T = do_rollout(agent, env, num_steps)
        return rew

    # 2. Experiments buffers
    best_reward = float('-inf')
    best_theta = None

    # 3. Initialize parameters
    theta_init = np.zeros(env.observation_space.shape[0] + 1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 4. Train the agent 
    for (i, iterdata) in enumerate(cem(noisy_evaluation, theta_init, **params)):
        current_reward = iterdata['y_mean']
        print('Iteration %2i. Episode mean reward: %7.3f' % (i, current_reward))

        if save_experiment_log:
            save_experiment(
                env_name,
                i,
                iterdata,
                iterdata['theta_mean'],
                current_reward,
                timestamp
            )   

        if current_reward > best_reward:
            best_reward = current_reward
            best_theta = iterdata['theta_mean']

    print('Demonstrating best policy...')
    agent = BinaryActionLinearPolicy(best_theta)
    final_reward, _ = do_rollout(agent, env, num_steps, render=True)
    print(f'Final demonstration reward: {final_reward}')

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    parser.add_argument('target', nargs="?", default="CartPole-v1")
    args = parser.parse_args()
    
    # Training parameters
    ###############################
    params = dict(
        n_iter=100,
        batch_size=10,
        elite_frac=0.2
    )
    num_steps = 200
    save_experiment_log = True
    ##############################

    if save_experiment_log:
        assert any(substring in args.target for substring in ['CartPole', 'Acrobot'])

    train_environment(args.target, params, num_steps, save_experiment_log)