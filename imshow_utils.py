import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def path_imshow(path, title=None):
    plt.figure()
    image = Image.open(path)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    else:
        plt.title(path)
    plt.pause(0.001)  # pause a bit so that plots are updated


# path_imshow('image_tutorial-1.png', 21312)


# ndarr.shape[0] is on y-axis (i.e. height) in the plot
def numpy_imshow(ndarr, title=None):
    plt.figure()
    plt.imshow(ndarr)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# numpy_imshow(np.random.rand(256, 512, 3), 'ndarr test')


# pytorch tensor have the number of channel in tensor.size(0)
def tensor_imshow(tensor, title=None):
    inp = tensor.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


# import torch
# tensor_imshow(torch.randn(3, 256, 512), 'pytorch tensor')