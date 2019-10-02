import torch
from model import Net # pylint: disable=unused-import
from torch import nn, optim
from torch.nn import functional as F
import numpy as np


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def random_policy(self):
        """ policy for dqn """
        return np.random.randint(0, 1, size=(1, ))


# Constants
SEED = 5
EPOCHS = 300
EPISODES = 100
GAMMA = 0.95

# Globals
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
world_net = torch.load("net.pth")
dqn = DQN()
target = DQN()
target.load_state_dict(dqn.state_dict())
target.eval()
optimizer = optim.Adam(dqn.parameters())


def world_model(state_action_batch):
    """ trained world model """
    state_action_batch = torch.from_numpy(state_action_batch.astype('float32'))
    return world_net(state_action_batch).detach().numpy()


def optimize_model():
    for epoch in range(EPOCHS):
        for episode in range(EPISODES):
            done = False
            init_state = (np.random.random_sample((4, )) * 0.1) - 0.05
            state_action = np.append(init_state, dqn.random_policy())
            episodic_batch = []
            expected_values = []

            while not done:
                simulated_transition = world_model(state_action)
                done = int(simulated_transition[-1])
                state_action = np.append(simulated_transition[0:4],
                        dqn.random_policy()).astype('float32')
                episodic_batch.append(state_action)
                exp_val = target.forward(torch.from_numpy(
                    state_action)) * GAMMA + simulated_transition[4]
                expected_values.append(exp_val.detach().numpy())

            # SA_S'R batched
            input_tensor = torch.from_numpy(np.array(episodic_batch))
            label_tensor = torch.from_numpy(np.array(expected_values))
            output_tensor = dqn.forward(input_tensor)

            # Compute Huber loss
            loss = F.smooth_l1_loss(output_tensor, label_tensor)
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # sync target and dqn
        target.load_state_dict(dqn.state_dict())
        if epoch % 10 == 0:
            print("Epoch", epoch, episode, loss)


def main():
    optimize_model()
    torch.save(dqn, "dqn.pth")

if __name__ == '__main__':
    main()
