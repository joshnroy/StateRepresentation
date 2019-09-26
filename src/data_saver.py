import gym
import numpy as np
from tqdm import trange
import sys

num_episodes = 100

env = gym.make("CartPole-v1")
for e in trange(num_episodes):
    states = []
    actions = []
    rewards = []
    observation = env.reset()
    states.append(observation)
    for _ in range(1000):

        action = env.action_space.sample() # your agent here (this takes random actions)
        actions.append(action)

        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        states.append(observation)

        if done:
            states = np.array(states)
            actions = np.array(actions)
            rewards = np.array(rewards)
            np.savez_compressed("data/offline_random_episodes/episode" + str(e), states=states, actions=actions, rewards=rewards)

            states = []
            actions = []
            rewards = []

            break

env.close()
