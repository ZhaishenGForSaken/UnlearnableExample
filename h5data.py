import h5py
import numpy as np
import os
from PIL import Image

with h5py.File('camelyonpatch_level_2_split_train_y.h5', 'r') as hy, h5py.File('camelyonpatch_level_2_split_train_x.h5', 'r') as hx:
    labels = hy['y'][:]
    images = hx['x'][:]

for unique_label in np.unique(labels):
    directory = f'class_{int(unique_label)}'
    if not os.path.exists(directory):
        os.makedirs(directory)

for idx, (image, label) in enumerate(zip(images, labels)):
    img = Image.fromarray(image.astype('uint8'))
    label = int(label[0])
    img.save(f'class_{label}/image_{idx}.png')

