import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import sys
from tqdm import tqdm
from tqdm import trange
import glob


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

states = np.empty((1,4))
actions = np.empty((1))
rewards = np.empty((1))

for ep_filepath in tqdm(glob.glob("./data/**/episode*.npz")):
    data = np.load(ep_filepath)
    states = np.append(states, data['states'][:-1], axis=0)
    actions = np.append(actions, data['actions'], axis=0)
    rewards = np.append(rewards, data['rewards'], axis=0)

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

