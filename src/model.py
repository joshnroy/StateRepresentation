from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from tqdm import tqdm
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

for ep_filepath in tqdm(glob.glob("./data/episode*.npz")):
    data = np.load(ep_filepath)

for i in len(states)-1:
    optimizer.zero_grad()
    output = net(np.append(states[i], actions[i]))

    loss = criterion(output, np.append(states[i+1], rewards[i]))
    print("loss", loss)
    loss.backward()
    optimizer.step()
