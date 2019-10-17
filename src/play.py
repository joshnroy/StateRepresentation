import gym 
import numpy as np
import torch
from dqn import DQN # pylint: disable=unused-import

EPISODES = 10


def init_agent(path='dqn.pth'):
    dqn = torch.load(path)
    return dqn

def plan(agent, state):
    max_qval = -np.inf
    best_action = None
    for action in [0,1]:
        state_action_tuple = torch.from_numpy(
                np.append(state, action).astype('float32'))
        q_value = agent.forward(state_action_tuple)[action].item()
        if max_qval < q_value:
            best_action = action
            max_qval = q_value
    return best_action, max_qval
    
def main():
    env = gym.make('CartPole-v0')
    state = env.reset()
    agent = init_agent('dqn.pth')
    for episode in range(EPISODES):
        cumulative_reward = 0
        done = False
        while not done:
            env.render()
            action, _ = plan(agent, state)
            state, reward, done, _ = env.step(action)
            cumulative_reward += reward
        state = env.reset() 
        print("Episode:", episode, cumulative_reward) 
    env.close()

if __name__ == '__main__':
    main()
