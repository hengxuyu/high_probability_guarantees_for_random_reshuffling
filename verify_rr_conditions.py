import torch
import random
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.autograd.functional import vhp

# Hyperparameters
LEARNING_RATE = 0.05
NUM_EPOCHS = 1
BATCH_SIZE = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the results folder with hyperparameters
results_folder = f"estimate_lipschitz_parameter"
os.makedirs(results_folder, exist_ok=True)

# Load the MNIST dataset
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())

# Define a simple Multi-Layer Perceptron (MLP) model
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(50, 50)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        x = self.fc3(x)
        return x

# Function to create partitions of the dataset (data points to form a f_i)
def create_partitions(dataset_size, batch_size):
    indices = list(range(dataset_size))
    random.shuffle(indices)
    return [indices[i:i + batch_size] for i in range(0, dataset_size, batch_size)]

# Function to reconstruct a model from a flattened parameter vector
def reconstruct_model_from_flat_params(flat_params):
    reconstructed_model = SimpleMLP().to(DEVICE)
    current_position = 0
    for p in reconstructed_model.parameters():
        flat_size = p.numel()
        p.data = flat_params[current_position:current_position + flat_size].view(p.size())
        current_position += flat_size
    return reconstructed_model

# Function to compute the vector-Hessian product (vhp)
def compute_vhp(loss_fn, flat_params, inputs, targets, v):
    def forward_fn(flat_params):
        temp_model = reconstruct_model_from_flat_params(flat_params)
        outputs = temp_model(inputs)
        loss = loss_fn(outputs, targets)
        return loss

    _, vhp_result = vhp(forward_fn, flat_params, v)
    return vhp_result


dataset_size = len(train_set)
partitions = create_partitions(dataset_size, BATCH_SIZE)
model = SimpleMLP().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
init_model = SimpleMLP().to(DEVICE)

lipschitz_grad_estimate_trajectory = []
hessian_lip_estimate_trajectory = []

for epoch in range(NUM_EPOCHS):
    epoch_starting_point = torch.cat([p.flatten() for p in model.parameters()])
    model_at_epoch_starting_point = model
    shuffled_partitions = create_partitions(dataset_size, BATCH_SIZE)

    inner_iterate_count = 0
    for partition_indices in shuffled_partitions:
        inner_iterate_count += 1

        # create a mini-batch from the partition
        mini_batch = Subset(train_set, partition_indices)
        mini_batch_loader = DataLoader(mini_batch, batch_size=len(mini_batch), shuffle=False)

        # train on the mini-batch
        images, labels = next(iter(mini_batch_loader))
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = nn.CrossEntropyLoss(reduction='mean')(outputs, labels)
        loss.backward()

        if inner_iterate_count > 1:
            # component grad on the mini-batch
            inner_iterate_grad = torch.cat([p.grad.flatten() for p in model.parameters()])
            inner_iterate_params = torch.cat([p.flatten() for p in model.parameters()])

        optimizer.step()

        if inner_iterate_count > 1:
            # compute mini-batch gradient at the epoch starting point
            for p in model_at_epoch_starting_point.parameters():
                if p.grad is not None:
                    p.grad.zero_()
            component_loss_at_epoch_starting_point = nn.CrossEntropyLoss(reduction='mean')(model_at_epoch_starting_point(images), labels)
            component_loss_at_epoch_starting_point.backward()
            grad_epoch_starting_point = torch.cat([p.grad.flatten() for p in model_at_epoch_starting_point.parameters()])

            vhp_component = compute_vhp(loss_fn=nn.CrossEntropyLoss(reduction='mean'), flat_params=epoch_starting_point, inputs=images, targets=labels, v=inner_iterate_params - epoch_starting_point)

            lipschitz_grad_estimate = torch.linalg.norm(inner_iterate_grad - grad_epoch_starting_point) / torch.linalg.norm(inner_iterate_params - epoch_starting_point)
            hessian_lip_estimate = torch.linalg.norm(inner_iterate_grad - grad_epoch_starting_point - vhp_component) / torch.linalg.norm(inner_iterate_params - epoch_starting_point) ** 2 * 2

            lipschitz_grad_estimate_trajectory.append(lipschitz_grad_estimate.item())
            hessian_lip_estimate_trajectory.append(hessian_lip_estimate.item())

            # Save the trajectories on the fly
            np.save(os.path.join(results_folder, f'lipschitz_grad_estimate_trajectory.npy'), np.array(lipschitz_grad_estimate_trajectory))
            np.save(os.path.join(results_folder, f'hessian_lip_estimate_trajectory.npy'), np.array(hessian_lip_estimate_trajectory))

            print(f'Inner iterate: {inner_iterate_count}, Lipschitz grad estimate: {lipschitz_grad_estimate.item()}, Hessian Lipschitz estimate: {hessian_lip_estimate.item()}.')

