import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import sys

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()

optimizer = optim.Adam(net.parameters(), lr=1e-3)

criterion = nn.MSELoss()

data = np.load("data/offline_random_episodes/episode0.npz")

states = data["states"]
actions = data["actions"]
rewards = data["rewards"]

epochs = 300
for epoch in range(epochs):
    for i in range(len(states)-1):
        optimizer.zero_grad()
        output = net(torch.from_numpy(np.append(states[i], actions[i]).astype(np.float32)))

        loss = criterion(output, torch.from_numpy(np.append(states[i+1], rewards[i]).astype(np.float32)))
        if i == 0:
            print("loss", loss)
            print("true label", np.append(states[i+1], rewards[i]))
            print("predicted label", output)
        loss.backward()
        optimizer.step()
