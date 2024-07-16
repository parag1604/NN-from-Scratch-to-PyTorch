import numpy as np
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils

from typing import Tuple


def plot_decision_boundary(clf, X, Y, cmap='Paired_r'):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 10 * h, X[:, 0].max() + 10 * h
    y_min, y_max = X[:, 1].min() - 10 * h, X[:, 1].max() + 10 * h
    xs, ys = np.meshgrid(
        np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = []
    for x, y in zip(xs.ravel(), ys.ravel()):
        Z.append(clf([x, y]) > 0.5)
    Z = np.array(Z).reshape(xs.shape)

    plt.figure(figsize=(5, 5))
    plt.contourf(xs, ys, Z, cmap=cmap, alpha=0.25)
    plt.contour(xs, ys, Z, colors='k', linewidths=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cmap, edgecolors='k')
    plt.show()


def get_data(
        dataset_name: str,
        batch_size: int = 64,
        transform: transforms = transforms.ToTensor(), # look into this
) -> Tuple[DataLoader, DataLoader]:
    if dataset_name == 'mnist':
        train_data = datasets.MNIST(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST(
            root='data', train=False, download=True, transform=transform)
    elif dataset_name == 'fashion_mnist':
        train_data = datasets.FashionMNIST(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.FashionMNIST(
            root='data', train=False, download=True, transform=transform)
    elif dataset_name == 'cifar10':
        train_data = datasets.CIFAR10(
            root='data', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(
            root='data', train=False, download=True, transform=transform)
    else:
        raise ValueError('Dataset not supported')

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)  # yield, generator in python
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    return train_loader, test_loader
