"""
Training loop for the Coma framework on Switch2-v0 the ma_gym
"""

import torch
import gym
import ma_gym
import matplotlib.pyplot as plt
import numpy as np

from COMA import COMA


def moving_average(x, N):
    return np.convolve(x, np.ones((N,)) / N, mode='valid')

def euclidean(a, b):
    return np.linalg.norm(np.array(a)-np.array(b))

def modify_obs(obs, threshold = 100):
    a = obs[0]
    b = obs[1]
    dist = euclidean(a, b)
    if dist < threshold:
        return([a + b, b + a])
    else:
        return([a + [0, 0], b + [0, 0]])

if __name__ == "__main__":
    # Hyperparameters
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    agent_num = 2

    state_dim = 4
    action_dim = 5

    gamma = 0.99
    lr_a = 0.0001
    lr_c = 0.005

    target_update_steps = 10

    # agent initialisation

    agents = COMA(agent_num, state_dim, action_dim, lr_c, lr_a, gamma, target_update_steps).to(device)

    env = gym.make("Switch2-v0")

    episodes_reward = []

    # training loop

    n_episodes = 100000
    episode = 0

    while episode < n_episodes:
        episode_reward = 0
        obs = env.reset()
        obs = modify_obs(obs)
        obs = torch.tensor(obs).to(device)

        done_n = [False]

        while not all(done_n):

            actions = torch.tensor(agents.get_actions(obs)).to(device)
            next_obs, reward, done_n, _ = env.step(actions)
            next_obs = modify_obs(next_obs)
            next_obs = torch.tensor(next_obs).to(device)

            agents.memory.reward.append(reward)
            for i in range(agent_num):
                agents.memory.done[i].append(done_n[i])

            episode_reward += sum(reward)

            obs = next_obs

        episodes_reward.append(episode_reward)

        episode += 1

        agents.train(device)

        avg_reward = np.mean(episodes_reward[-100:])
        if avg_reward > 2:
            torch.save(agents, "saved_models/coma.pth")
            break
        if episode % 100 == 0:
            print(f"episode: {episode}, average reward: {avg_reward}")
