import torch
from model import Net  # pylint: disable=unused-import
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import gym 
env = gym.make('CartPole-v0')


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
            self.forward(torch.cat((x, torch.FloatTensor([i]))))[i] for i in range(2)
        ])
        return torch.argmax(q_values).unsqueeze(dim=0).type(torch.FloatTensor)

    def random_policy(self):
        """ policy for dqn """
        return np.random.randint(0, 1, size=(1, ))


# Constants
SEED = 5
EPOCHS = 100
EPISODES = 200
GAMMA = 0.95

# Globals
dream = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dqn = DQN()
policy_net = DQN()
policy_net.load_state_dict(dqn.state_dict())
policy_net.eval()
optimizer = optim.Adam(dqn.parameters())


def world_model(state_action_batch, world_net):
    """ trained world model """
    return world_net(state_action_batch)

def optimize_model(world_net):
    for epoch in range(EPOCHS):
        epoch_reward = 0
        for episode in range(EPISODES):
            done = False
            if dream:
                init_state = torch.from_numpy(((np.random.random_sample(
                    (4, )) * 0.1) - 0.05).astype('float32'))
            else:
                init_state = torch.FloatTensor(env.reset())
            action = policy_net.best_action(init_state)
            state_action = torch.cat(
                (init_state, action))
            cumulative_reward = 0
            predicted_q_values = []
            expected_q_values = []
            while not done:
                if dream:
                    simulated_transition = world_model(state_action,
                                                       world_net)
                else:
                    observation, reward, done, _ = env.step(int(action.item()))
                    transition = np.append(observation, [reward, done])
                    simulated_transition = torch.FloatTensor(transition)
                next_state = simulated_transition[0:4]
                reward = simulated_transition[4]
                done = int(np.round(np.abs(simulated_transition[-1].detach().numpy())))
                action = policy_net.best_action(next_state)
                next_state_action = torch.cat((next_state, action))
                predicted_q_values.append(reward + (GAMMA *
                                               dqn.forward(state_action)))
                expected_q_values.append(policy_net.forward(next_state_action))
                cumulative_reward += reward.item()
                state_action = next_state_action

            predicted_batch = torch.stack(predicted_q_values, dim=1)
            expected_batch = torch.stack(expected_q_values, dim=1)

            # Compute Huber loss
            loss = F.smooth_l1_loss(predicted_batch, expected_batch)
            
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_reward += cumulative_reward

        # sync policy_net and dqn
        policy_net.load_state_dict(dqn.state_dict())
        if epoch % 1 == 0:
            print("Epoch", epoch, episode, loss.item(),
                  epoch_reward / EPISODES)
        if loss.item() < 1e-9:
            print("Stopping early, loss", loss.item())
            break


def main():
    world_net = torch.load("net.pth")
    optimize_model(world_net)
    torch.save(dqn, "dqn.pth")


if __name__ == '__main__':
    main()
