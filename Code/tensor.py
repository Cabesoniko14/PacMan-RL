import numpy as np
import torch

# Create a NumPy array with shape (4, 84, 84, 1)
observation = np.zeros((4, 84, 84, 1), dtype=np.uint8)

# Convert the array to a PyTorch tensor and add an extra dimension at the beginning
observation_tensor = torch.tensor(observation).unsqueeze(0)  # shape: (1, 4, 84, 84, 1)
observation_tensor = observation_tensor.squeeze(-1)         # shape: (1, 4, 84, 84)


# Print the shape of the tensor
print(observation_tensor.shape)  # Output: torch.Size([1, 4, 84, 84])