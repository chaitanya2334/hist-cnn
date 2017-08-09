import torch
from torch.autograd import Variable
from torch.nn.functional import softmax
import numpy as np
import os

def argmax(var):
    assert isinstance(var, Variable)
    var = softmax(var)
    _, preds = torch.max(var.data, 1)

    preds = preds.squeeze().cpu().numpy().tolist()
    return preds


def crop(image, m):
    image = np.copy(image[:m[0], :m[1]])
    return image


def pad(image, m):
    if m[0] > image.shape[0] or m[1] > image.shape[1]:
        image = np.lib.pad(image, ((0, m[0] - image.shape[0]), (0, m[1] - image.shape[1]), (0, 0)),
                           'constant')
    return image


def fit_image(image, m=(299, 299)):
    assert isinstance(image, np.ndarray)
    # crop if bigger
    image = crop(image, m)

    # pad if smaller
    image = pad(image, m)

    assert image.shape[0] == m[0] and image.shape[1] == m[1]

    return image


def get_path(path):
    current = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(current, path)
