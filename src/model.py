import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import glob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 6)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
net.to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-3)

criterion = nn.L1Loss()
train = False
if train:
    epochs = 300
    for epoch in range(epochs):
        for ep_filepath in glob.glob("./data/**/episode*.npz"):
            data = np.load(ep_filepath)

            states = data['states']
            actions = data['actions']
            rewards = data['rewards']
            terminal = data['terminal']

            optimizer.zero_grad()
            input_tensor = torch.from_numpy(
                np.append(states[:-1, :],
                          np.expand_dims(actions, axis=1),
                          axis=1).astype(np.float32))
            input_tensor = input_tensor.to(device)
            output = net(input_tensor)

            label = torch.from_numpy(
                np.append(np.append(states[1:, :],
                                    np.expand_dims(rewards, axis=1),
                                    axis=1).astype(np.float32),
                          np.expand_dims(terminal, axis=1),
                          axis=1).astype(np.float32))
            label = label.to(device)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        print("loss", loss)
        if loss < 0.0005:
            break

    torch.save(net, "net.pth")
else:
    net = torch.load("net.pth")

data = np.load("data/offline_random_episodes/episode0.npz")
states = data['states']
actions = data['actions']
rewards = data['rewards']
terminal = data['terminal']
input_tensor = torch.from_numpy(
    np.append(states[-2, :], actions[-1]).astype(np.float32)).to(device)
print(net(input_tensor))
print(states[-1, :], rewards[-1], terminal[-1])
