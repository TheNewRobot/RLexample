from __future__ import print_function

import argparse

import gymnasium as gym
import numpy as np
# from gymnasium import logger

from _policies import BinaryActionLinearPolicy  # Different file so it can be unpickled


def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function

    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size * elite_frac))
    th_std = np.ones_like(th_mean) * initial_std
    
    for _ in range(n_iter):
        # Generates samples
        ths = np.array([th_mean + dth for dth in th_std[None, :] * np.random.randn(batch_size, th_mean.size)])
        # Generate scores for the samples in batches (for loop)
        ys = np.array([f(th) for th in ths])

        # Selects the elite performers 
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]

        # Updates the distribution parameters 
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        # Dictionary
        yield {'ys': ys, 'theta_mean': th_mean, 'y_mean': ys.mean()}


def do_rollout(agent, env, num_steps, render=False):
    """
    Executes one episode
    """
    total_rew = 0
    ob, _ = env.reset()
    for t in range(num_steps):
        a = agent.act(ob)
        (ob, reward, terminated, truncated, _info) = env.step(a)
        done = np.logical_or(terminated, truncated)  # here use the logical or, one can use terminal
        total_rew += reward
        if render and t % 3 == 0: env.render()
        if done: break
    return total_rew, t + 1


if __name__ == '__main__':
    # logger.set_level(logger.INFO)
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--display', action='store_true')
    parser.add_argument('target', nargs="?", default="CartPole-v0")
    args = parser.parse_args()
    # 1. Create the environment 
    env = gym.make(args.target, render_mode='human')
    np.random.seed(0)
    # 2. Experiment hyperparamenters 
    params = dict(n_iter=100, 
                  batch_size=10, 
                  elite_frac=0.2)
    num_steps = 200
    # 3. Local functions 
    def noisy_evaluation(theta):
        agent = BinaryActionLinearPolicy(theta)
        rew, T = do_rollout(agent, env, num_steps)
        return rew
    # 4. Train the agent, and snapshot each stage
    for (i, iterdata) in enumerate(cem(noisy_evaluation, np.zeros(env.observation_space.shape[0] + 1), **params)):
        print('Iteration %2i. Episode mean reward: %7.3f' % (i, iterdata['y_mean']))
        # Best agent for demonstration
        print('Creating agent for demonstration!')
        agent = BinaryActionLinearPolicy(iterdata['theta_mean'])
        do_rollout(agent, env, 200, render=True)

    env.close()
