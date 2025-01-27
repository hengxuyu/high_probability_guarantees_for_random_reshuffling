import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from datetime import datetime
import os
import yaml
import numpy as np
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load configuration from a YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
LEARNING_RATE = config['learning_rate']
GRAD_NORM_THRESHOLD = config['grad_norm_threshold']
NUM_EPOCHS = config['num_epochs']
BATCH_SIZE = config['batch_size']
lr_decay_rate = config['gamma']
results_folder = config['results_folder']

paralell = int(os.environ.get('PARALELL', 0))
repetition = int(os.environ.get('REPETITION', 0))


train_set = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transforms.ToTensor())
os.makedirs(results_folder, exist_ok=True)
full_grad_bs = 256
full_data_size = len(train_set)
full_train_loader = torch.utils.data.DataLoader(train_set, batch_size=full_grad_bs, shuffle=False)

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

def compute_full_gradient_norm(model, full_data_loader, full_data_size=full_data_size):
    model.zero_grad()
    for images, labels in full_data_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = nn.CrossEntropyLoss(reduction='sum')(outputs, labels)
        loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm**2

    total_norm = total_norm**0.5 / full_data_size
    return total_norm


def sgd_train(model, train_loader, lr, num_epochs, grad_threshold, device, full_grad_loader, lr_schedule, seed):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    grad_norm_record = []
    loss_record = []

    if lr_schedule == 'cosine':
        def lambda_lr(epoch): return 0.5 * \
            (1 + math.cos(math.pi * epoch / num_epochs))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    elif lr_schedule == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=lr_decay_rate)
    else:
        raise ValueError(f"Invalid learning rate schedule: {lr_schedule}. Must be 'cosine' or 'step'.")

    for epoch in range(num_epochs):
        for x, y in train_loader:
            x, y = x.reshape(-1, 28*28).to(device), y.to(device)
            optimizer.zero_grad()
            z = model(x)
            loss = nn.CrossEntropyLoss(reduction='mean')(z, y)
            loss.backward()
            optimizer.step()

        loss_record.append(loss.item())

        full_grad_norm = compute_full_gradient_norm(model, full_grad_loader)
        grad_norm_record.append(full_grad_norm)
        if full_grad_norm < grad_threshold:
            print(f"SGD: Stop early at epoch {epoch} with gradient norm {full_grad_norm}.")

        scheduler.step()

    np.save(os.path.join(results_folder, f'sgd_grad_norm_{seed}.npy'), np.array(grad_norm_record))
    np.save(os.path.join(results_folder, f'sgd_loss_{seed}.npy'), np.array(loss_record))

    return epoch

def rr_train(model, dataset, train_loader, lr, num_epochs, grad_threshold, device, full_grad_loader, lr_schedule, seed):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    grad_norm_record = []
    loss_record = []

    if lr_schedule == 'cosine':
        def lambda_lr(epoch): return 0.5 * \
            (1 + math.cos(math.pi * epoch / num_epochs))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
    elif lr_schedule == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=lr_decay_rate)
    else:
        raise ValueError(f"Invalid learning rate schedule: {lr_schedule}. Must be 'cosine' or 'step'.")

    for epoch in range(num_epochs):
        for x, y in train_loader:
            x, y = x.reshape(-1, 28*28).to(device), y.to(device)
            optimizer.zero_grad()
            z = model(x)
            loss = nn.CrossEntropyLoss(reduction='mean')(z, y)
            loss.backward()
            optimizer.step()
        loss_record.append(loss.item())

        full_grad_norm = compute_full_gradient_norm(model, full_grad_loader)
        grad_norm_record.append(full_grad_norm)
        if full_grad_norm < grad_threshold:
            print(f"RR: Stop early at epoch {epoch} with gradient norm {full_grad_norm}.")

        scheduler.step()

    np.save(os.path.join(results_folder, f'rr_grad_norm_{seed}.npy'), np.array(grad_norm_record))
    np.save(os.path.join(results_folder, f'rr_loss_{seed}.npy'), np.array(loss_record))

    return epoch


sgd_iter_counts = []
rr_iter_counts = []
random_seeds = [i+paralell * 4 for i in range(1, repetition+1)]

for seed in random_seeds:
    torch.manual_seed(seed)
    random.seed(seed)

    train_rr = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    train_sgd = torch.utils.data.DataLoader(dataset=train_set, batch_size=BATCH_SIZE, sampler=torch.utils.data.RandomSampler(train_set, replacement=True))

    sgd_iters_count = sgd_train(model=SimpleMLP().to(DEVICE), train_loader=train_sgd, lr=LEARNING_RATE, num_epochs=NUM_EPOCHS, grad_threshold=GRAD_NORM_THRESHOLD, device=DEVICE, full_grad_loader=full_train_loader, lr_schedule=config['lr_schedule'], seed=seed)
    sgd_iter_counts.append(sgd_iters_count)

    rr_iters_count = rr_train(model=SimpleMLP().to(DEVICE), train_loader=train_rr, lr=LEARNING_RATE, num_epochs=NUM_EPOCHS, grad_threshold=GRAD_NORM_THRESHOLD, device=DEVICE, full_grad_loader=full_train_loader, lr_schedule=config['lr_schedule'], seed=seed)
    rr_iter_counts.append(rr_iters_count)

    np.save(os.path.join(results_folder, f'sgd_iter_counts_{seed}.npy'), np.array(sgd_iter_counts))
    np.save(os.path.join(results_folder, f'rr_iter_counts_{seed}.npy'), np.array(rr_iter_counts))

