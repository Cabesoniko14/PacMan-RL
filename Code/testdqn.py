import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        conv_out_size = self._get_conv_out(input_shape)

        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_out(self, shape):
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, x):
        x = x.float() / 255
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.fc2(x)
        return x
    

# Create a NumPy array with shape (4, 84, 84, 1)
observation = np.zeros((4, 84, 84, 1), dtype=np.uint8)

# Convert the array to a PyTorch tensor and add an extra dimension at the beginning
observation_tensor = torch.tensor(observation).unsqueeze(0)  # shape: (1, 4, 84, 84, 1)
observation_tensor = observation_tensor.squeeze(-1)         

model = DQN((4, 84, 84), 9)
output = model(observation_tensor)
print(output)


