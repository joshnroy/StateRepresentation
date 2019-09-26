from torch import nn 
from torch.nn import functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.fc1 = nn.Linear(5, 5)  # 6*6 from image dimension
        self.fc2 = nn.Linear(5, 5)  # 6*6 from image dimension

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = Net()
print(net)
