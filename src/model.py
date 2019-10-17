import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000
train = False
BATCH_SIZE = 10
WEIGHTS = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.5]).astype('float32')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 48)
        self.fc3 = nn.Linear(48, 48)
        self.fc4 = nn.Linear(48, 48)
        self.fc5 = nn.Linear(48, 6)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


def weightedL1Loss(output, label, weights):
    return torch.mean(torch.mul(torch.abs(output - label), weights))


def main():
    net = Net()
    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criterion = weightedL1Loss

    if train:
        # Batch up the data
        dataset = []
        for ep_filepath in glob.glob("./data/**/episode*.npz"):
            data = np.load(ep_filepath)
            states = data['states']
            actions = data['actions']
            rewards = data['rewards']
            terminal = data['terminal']
            episode_batch = []
            for i in range(len(actions)):
                transition = np.append(states[i], actions[i])
                transition = np.append(transition, states[i + 1])
                transition = np.append(transition, rewards[i])
                transition = np.append(transition, terminal[i])
                episode_batch.append(transition.transpose())
            dataset.append(np.array(episode_batch))
        dataset = np.vstack(dataset)

        for epoch in range(EPOCHS):
            np.random.shuffle(dataset)
            cumulative_loss = 0
            for i in range(0, dataset.shape[0] - BATCH_SIZE, BATCH_SIZE):
                batch = np.take(dataset, np.arange(i, i + BATCH_SIZE), axis=0)
                optimizer.zero_grad()
                input_tensor = torch.from_numpy(
                    batch[:, 0:5].astype('float32'))
                output = net(input_tensor)
                label = torch.from_numpy(batch[:, 5:].astype('float32'))
                label = label.to(device)
                loss = criterion(output, label, torch.from_numpy(WEIGHTS))
                cumulative_loss += loss.item()
                loss.backward()
                optimizer.step()

            print("epoch: {:04d}, loss: {:02.6f}".format(
                epoch, cumulative_loss))
            if loss < 1e-6:
                break

        torch.save(net, "net.pth")
    else:
        net = torch.load("net.pth")

    data = np.load("data/offline_random_episodes/episode11.npz")
    states = data['states']
    actions = data['actions']
    rewards = data['rewards']
    terminal = data['terminal']

    input_tensor = torch.from_numpy(
        np.append(states[-2, :], actions[-1]).astype(np.float32)).to(device)
    print(net(input_tensor))
    print(states[-1, :], rewards[-1], terminal[-1])

    input_tensor = torch.from_numpy(
        np.append(states[-3, :], actions[-2]).astype(np.float32)).to(device)
    print(net(input_tensor))
    print(states[-2, :], rewards[-2], terminal[-2])


if __name__ == '__main__':
    main()
