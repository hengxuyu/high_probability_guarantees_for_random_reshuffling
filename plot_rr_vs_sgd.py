import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import glob
import sys

if len(sys.argv) != 2:
    print("Usage: python plot_rr_vs_sgd.py <directory>")
    sys.exit(1)

directory = sys.argv[1]

# Get all .npy files in the specified directory
rr_npy_files = glob.glob(os.path.join(directory, "results/*rr_grad_norms*.npy"))

# Load the last elements of each file
rr_last_elements = []
for file in rr_npy_files:
    data = np.load(file)
    rr_last_elements.append(data.flatten()[-1])

# Convert to numpy array and filter values <= 1
rr_last_elements = np.array(rr_last_elements)

sgd_npy_files = glob.glob(os.path.join(directory, "results/*sgd_grad_norms*.npy"))
sgd_last_elements = []
for file in sgd_npy_files:
    data = np.load(file)
    sgd_last_elements.append(data.flatten()[-1])
sgd_last_elements = np.array(sgd_last_elements)


plt.hist(rr_last_elements, bins=50, alpha=0.5)
plt.hist(sgd_last_elements, bins=50, alpha=0.5)
plt.legend(['RR', 'SGD'])
plt.xlabel('Full gradient norm at epoch 100')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig(f"histogram_full_gradient_norm_epoch_100.pdf")

plt.gcf().set_size_inches(12, 9)
letter_size = 28
number_size = 28

rr_stop_at = []
for file in rr_npy_files:
    data = np.load(file)
    indices = np.where(data < 1e-3)[0]
    if indices.size > 0:
        rr_stop_at.append(indices[0])


sgd_stop_at = []
for file in sgd_npy_files:
    data = np.load(file)
    indices = np.where(data < 1e-3)[0]
    if indices.size > 0:
        sgd_stop_at.append(indices[0])


min_epoch = min(min(rr_stop_at), min(sgd_stop_at))
max_epoch = max(max(rr_stop_at), max(sgd_stop_at))
bins = np.arange(min_epoch, max_epoch + 2) - 0.5  # +2 to include the last value, -0.5 for proper bin centering


plt.hist(rr_stop_at, bins=bins, alpha=0.5, label='RR')
plt.hist(sgd_stop_at, bins=bins, alpha=0.5, label='SGD')

plt.xticks(sorted(list(set(rr_stop_at + sgd_stop_at)))[::5], fontsize=number_size, fontfamily='serif')

max_count = max(
    max(np.histogram(rr_stop_at, bins=bins)[0]),
    max(np.histogram(sgd_stop_at, bins=bins)[0])
)
plt.yticks(np.arange(0, max_count + 1, 1), fontsize=number_size, fontfamily='serif')

plt.legend(fontsize=letter_size, loc='upper right')
plt.xlabel('Number of iterations / epochs', fontfamily='serif', fontsize=letter_size)
plt.ylabel('Count', fontfamily='serif', fontsize=letter_size)
plt.grid(True, alpha=0.1)

plt.savefig(f"histogram_1e-3.pdf")

