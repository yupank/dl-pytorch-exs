import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# import numpy as np
from random import randrange
import matplotlib.pyplot as plt

# for the first time, download should be = True
trainset = datasets.CIFAR10(root='./data', download=False, transform= transforms.ToTensor())
im_size = len(trainset)
print(im_size)

# viewing examples of images
show_rows = 3
show_cols = 8
fig, axs = plt.subplots(show_rows, show_cols, squeeze=False, figsize=(show_cols, show_rows*2) )

# iteration via DataLoader
train_loader = torch.utils.data.DataLoader(trainset, batch_size=show_cols, shuffle=True)
data_iterator = iter(train_loader)
images, labels = next(data_iterator)
print(labels[0:])
print(images.size())

for ax_row in axs:
    images, labels = next(data_iterator)
    for i in range(show_cols):
        tr_image = images[i, :]
        np_image = tr_image.permute(1,2,0)
        ax_row[i].set_xlabel('')
        ax_row[i].set_xticks([])
        ax_row[i].set_ylabel('')
        ax_row[i].set_yticks([])
        ax_row[i].imshow(np_image)
        ax_row[i].set_title(f'_ {labels[i]} _')
plt.show()