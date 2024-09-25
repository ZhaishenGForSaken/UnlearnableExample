import medmnist
from medmnist import INFO
import numpy as np
import os

# Define dataset
DataClass = getattr(medmnist, "OrganMNIST3D")
info = INFO["organmnist3d"]
task = info["task"]
n_channels = info["n_channels"]
n_classes = len(info["label"])

# Load dataset
train_dataset = DataClass(split='train', download=True)
test_dataset = DataClass(split='test', download=True)

# Create directories for saving images
os.makedirs('data/datasets/classification/organmnist3d_numpy/train', exist_ok=True)
os.makedirs('data/datasets/classification/organmnist3d_numpy/test', exist_ok=True)
for i in range(n_classes):
    os.makedirs(f'data/datasets/classification/organmnist3d_numpy/train/class_[{i}]', exist_ok=True)
    os.makedirs(f'data/datasets/classification/organmnist3d_numpy/test/class_[{i}]', exist_ok=True)

# Function to save images
def save_images_numpy(dataset, split):
    for i in range(len(dataset)):
        img, target = dataset[i]
        label = target
        img = np.array(img)  # Convert PIL image to NumPy array if needed
        np.save(f'data/datasets/classification/organmnist3d_numpy/{split}/class_{label}/{i}.npy', img)

# Save training images
save_images_numpy(train_dataset, 'train')

# Save testing images
save_images_numpy(test_dataset, 'test')

print("Images have been saved and classified as NumPy arrays.")