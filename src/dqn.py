import torch
import random
from model import Net  # pylint: disable=unused-import
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from collections import namedtuple

from tqdm import trange

import matplotlib.pyplot as plt

import gym
env = gym.make('CartPole-v0')

# Constants
EPISODES = 500
GAMMA = 0.95
EPSILON = 0.1
UPDATE_INTERVAL = 200
BATCHSIZE = 50

TEST_EPISODES = 10

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_action', 'next_state', 'reward', 'not_done'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(5, 48)
        self.fc2 = nn.Linear(48, 48)
        self.fc3 = nn.Linear(48, 48)
        self.fc4 = nn.Linear(48, 2)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def best_action(self, x):
        q_values = torch.stack([
            self.forward(torch.cat((x, torch.FloatTensor([i]))))[i] \
            for i in range(2)
        ])
        action = torch.argmax(q_values).unsqueeze(dim=0).type(torch.FloatTensor)
        return action

    def random_policy(self):
        """ policy for dqn """
        return np.random.randint(0, 1, size=(1, ))


# Globals
dream = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN()
target_net = DQN()
target_net.load_state_dict(dqn.state_dict())
target_net.eval()
optimizer = optim.Adam(dqn.parameters(), lr=5e-4)


def world_model(state_action_batch, world_net):
    """ trained world model """
    return world_net(state_action_batch)


def get_action(state, eps):
    # Get action
    if np.random.random_sample((1, ))[0] < eps:
        action = torch.FloatTensor([np.random.choice([0, 1])])
    else:
        action = dqn.best_action(state)
    return action


def get_init_state():
    if dream:
        init_state = torch.from_numpy(((np.random.random_sample(
            (4, )) * 0.1) - 0.05).astype('float32'))
    else:
        init_state = torch.FloatTensor(env.reset())
    return init_state


def training_loop(memory, world_net):
    global_step = 0
    episode_rewards = []
    episode_losses = []
    for episode in trange(EPISODES):
        episode_reward = 0
        episode_loss = 0
        episode_len = 0

        # Make init transition
        state = get_init_state()
        action = get_action(state, EPSILON)

        while True:
            state_action = torch.cat((state, action))
            # Get transition
            if dream:
                simulated_transition = world_model(state_action, world_net)
            else:
                observation, reward, done, _ = env.step(int(action.item()))
                transition = np.append(observation, [reward, done])
                simulated_transition = torch.FloatTensor(transition)

            episode_reward += simulated_transition[4]
            # Push into memory
            next_action = dqn.best_action(simulated_transition[:4])
            memory.push(state.unsqueeze(1), action.unsqueeze(0),
                        next_action.unsqueeze(0),
                        simulated_transition[:4].unsqueeze(1),
                        simulated_transition[4].unsqueeze(0), not (done))

            state = simulated_transition[:4]
            action = get_action(state, EPSILON)

            if done:
                if len(memory) >= BATCHSIZE:
                    loss, predicted_batch, reward_batch = optimize_model(
                        memory)
                    episode_loss += loss
                break
            else:
                if len(memory) >= BATCHSIZE:
                    loss, predicted_batch, reward_batch = optimize_model(
                        memory)
                    episode_loss += loss
            global_step += 1
            episode_len += 1.

            # sync policy_net and dqn
            if global_step % UPDATE_INTERVAL == 0:
                target_net.load_state_dict(dqn.state_dict())

        if episode % 1 == 0:
            episode_rewards.append(episode_reward)
            episode_losses.append(episode_loss / episode_len)

    return episode_rewards, episode_losses

def testing_loop():
    total_reward = 0.
    for episode in trange(TEST_EPISODES):
        episode_reward = 0.

        # Make init transition
        state = get_init_state()
        action = get_action(state, 0.)

        while True:
            state_action = torch.cat((state, action))
            # Get transition

            observation, reward, done, _ = env.step(int(action.item()))
            transition = np.append(observation, [reward, done])
            simulated_transition = torch.FloatTensor(transition)

            episode_reward += simulated_transition[4]

            state = simulated_transition[:4]
            action = get_action(state, 0.)

            if done:
                total_reward += episode_reward
                break

    return total_reward / TEST_EPISODES


def optimize_model(memory):
    # Get batched transitions
    transitions = memory.sample(BATCHSIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state,
                            dim=1).transpose(0, 1).type(torch.FloatTensor)
    next_state_batch = torch.cat(batch.next_state,
                                 dim=1).transpose(0, 1).type(torch.FloatTensor)
    next_action_batch = torch.cat(batch.next_action,
                                  dim=1).transpose(0,
                                                   1).type(torch.FloatTensor)
    action_batch = torch.cat(batch.action,
                             dim=1).transpose(0, 1).type(torch.FloatTensor)
    reward_batch = torch.cat(batch.reward, dim=0).type(torch.FloatTensor)
    not_final_state_mask = batch.not_done

    state_action = torch.cat((state_batch, action_batch), dim=1)
    next_state_action = torch.cat((next_state_batch, next_action_batch), dim=1)

    predicted_batch = dqn(state_action).gather(
        1, action_batch.type(torch.LongTensor))
    expected_batch = reward_batch
    expected_batch[not_final_state_mask] += (
        GAMMA * target_net(next_state_action).max(1)[0])
    expected_batch = expected_batch.unsqueeze(1)

    # Compute loss
    # loss = F.smooth_l1_loss(predicted_batch, expected_batch)
    loss = F.mse_loss(predicted_batch, expected_batch)
    # loss = torch.clamp(loss, 0, 1)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, predicted_batch, reward_batch


def main():
    # world_net = torch.load("net.pth")
    memory = ReplayMemory(10000)
    rewards, losses = training_loop(memory, None)
    plt.plot(rewards)
    plt.title("rewards")
    plt.show()
    plt.plot(losses)
    plt.title("losses")
    plt.show()
    torch.save(dqn, "dqn.pth")
    print("Test Reward", testing_loop())


if __name__ == '__main__':
    main()
