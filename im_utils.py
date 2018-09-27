import PIL
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as transforms

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
    if plt.isinteractive():
        plt.ioff()
    plt.show()
# import torch
# tensor_imshow(torch.randn(3, 256, 512), 'pytorch tensor')


def load_img(path, new_size):
    img = Image.open(path).convert(mode='RGB')
    if new_size:
        # for fixed-size squared resizing, leave only the following line uncommented in this if statement
        # img = transforms.resize(img, (new_size, new_size), PIL.Image.BICUBIC)
        width, height = img.size
        max_dim_ix = np.argmax(img.size)
        if max_dim_ix == 0:
            new_shape = (int(new_size * (height / width)), new_size)
            img = transforms.resize(img, new_shape, PIL.Image.BICUBIC)
        else:
            new_shape = (new_size, int(new_size * (width / height)))
            img = transforms.resize(img, new_shape, PIL.Image.BICUBIC)
    return transforms.to_tensor(img)
