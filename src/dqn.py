import torch
import random
from model import Net  # pylint: disable=unused-import
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from collections import namedtuple

import gym
env = gym.make('CartPole-v0')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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
        return torch.argmax(q_values).unsqueeze(dim=0).type(torch.FloatTensor)

    def random_policy(self):
        """ policy for dqn """
        return np.random.randint(0, 1, size=(1, ))


# Constants
SEED = 5
EPOCHS = 200
EPISODES = 500
GAMMA = 0.95
EPSILON = 0.1
UPDATE_INTERVAL = 2
BATCHSIZE = 128

# Globals
dream = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN()
target_net = DQN()
target_net.load_state_dict(dqn.state_dict())
target_net.eval()
optimizer = optim.Adam(dqn.parameters())


def world_model(state_action_batch, world_net):
    """ trained world model """
    return world_net(state_action_batch)


def get_action(state):
    # Get action
    if np.random.random_sample((1, ))[0] < EPSILON:
        action = torch.FloatTensor([np.random.choice([0, 1])])
    else:
        action = target_net.best_action(state)
    return action


def get_init_state():
    if dream:
        init_state = torch.from_numpy(((np.random.random_sample(
            (4, )) * 0.1) - 0.05).astype('float32'))
    else:
        init_state = torch.FloatTensor(env.reset())
    return init_state


def training_loop(memory, world_net):
    for epoch in range(EPOCHS):
        epoch_reward = 0
        epoch_value = 0
        epoch_loss = 0

        # Make init transition
        state = get_init_state()
        action = get_action(state)

        for _ in range(EPISODES):
            state_action = torch.cat((state, action))
            # Get transition
            if dream:
                simulated_transition = world_model(state_action, world_net)
            else:
                observation, reward, done, _ = env.step(int(action.item()))
                transition = np.append(observation, [reward, done])
                simulated_transition = torch.FloatTensor(transition)

            # Push into memory
            memory.push(state.unsqueeze(1), action.unsqueeze(0),
                        simulated_transition[:4].unsqueeze(1),
                        simulated_transition[4].unsqueeze(0))

            state = simulated_transition[:4]
            action = get_action(state)

            if done:
                if len(memory) >= BATCHSIZE:
                    loss, predicted_batch, reward_batch = optimize_model(
                        memory)
                    epoch_reward += reward_batch.sum().item()
                    epoch_value += predicted_batch.mean().item()
                    epoch_loss += loss.item()
                break
            else:
                if len(memory) < BATCHSIZE:
                    continue
                else:
                    loss, predicted_batch, reward_batch = optimize_model(
                        memory)
                    epoch_reward += reward_batch.sum().item()
                    epoch_value += predicted_batch.mean().item()
                    epoch_loss += loss.item()


        # sync policy_net and dqn
        if epoch % UPDATE_INTERVAL == 0:
            target_net.load_state_dict(dqn.state_dict())
        if epoch % 1 == 0:
            print("Epoch", epoch, epoch_loss / EPISODES,
                  epoch_reward / EPISODES, epoch_value / EPISODES)


def optimize_model(memory):
    # Get batched transitions
    transitions = memory.sample(BATCHSIZE)
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state,
                            dim=1).transpose(0, 1).type(torch.FloatTensor)
    next_state_batch = torch.cat(batch.next_state,
                                 dim=1).transpose(0, 1).type(torch.FloatTensor)
    action_batch = torch.cat(batch.action,
                             dim=1).transpose(0, 1).type(torch.FloatTensor)
    reward_batch = torch.cat(batch.reward, dim=0).type(torch.FloatTensor)

    state_action = torch.cat((state_batch, action_batch), dim=1)
    next_state_action = torch.cat((next_state_batch, action_batch), dim=1)

    predicted_batch = dqn(state_action).gather(
        1, action_batch.type(torch.LongTensor))
    expected_batch = reward_batch + (GAMMA *
                                     target_net(next_state_action).max(1)[0])
    expected_batch = expected_batch.unsqueeze(1)

    # Compute loss
    loss = F.smooth_l1_loss(predicted_batch, expected_batch)
    loss = torch.clamp(loss, 0, 1)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, predicted_batch, reward_batch


def main():
    world_net = torch.load("net.pth")
    memory = ReplayMemory(10000)
    training_loop(memory, world_net)
    torch.save(dqn, "dqn.pth")


if __name__ == '__main__':
    main()
