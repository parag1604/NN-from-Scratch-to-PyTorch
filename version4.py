import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import utils as vutils
from torch.utils.data import DataLoader

from tqdm import tqdm
from typing import Tuple
from matplotlib import pyplot as plt

from utils import get_data


class Network(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(Network, self).__init__()
        self.hidden_layer = nn.Linear(784, 100)
        self.output_layer = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.hidden_layer(x))
        x = self.output_layer(x)  # logits
        x = F.log_softmax(x, dim=1)  # more stable
        return x


def train_epoch(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    for batch_idx, (data, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)):
        # data batch already provided by the datalaoder
        data, target = data.to(device), target.to(device)

        # forward pass
        output = model(data)

        # compute loss
        loss = F.nll_loss(output, target)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        total_loss += loss.item() * data.size(0)
        total_correct += (output.argmax(dim=1) == target).sum().item()
        total_samples += data.size(0)
    return total_loss / total_samples, total_correct / total_samples


def train(
        model: nn.Module,
        data_loader: DataLoader,
        optimizer: optim.Optimizer,
        epochs: int=10,
        device: torch.device=torch.device('cpu'), # 'cuda' in case we want to use gpu
) -> None:
    print("Training...")
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, data_loader, optimizer, device)
        # Exercise: Plot some randomly chosen datapoints along with the labels
        print(
                f"Epoch: {epoch + 1}/{epochs}, "
                f"Train loss: {train_loss:.4f}, "
                f"Train accuracy: {train_acc:.4f}"
        )
    print("Done!")


def evaluate(
        model: nn.Module,
        data_loader: DataLoader,
        device: torch.device=torch.device('cpu'),
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    for batch_idx, (data, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)):
        # data batch already provided by the datalaoder
        data, target = data.to(device), target.to(device)

        # forward pass
        output = model(data)

        # compute loss
        loss = F.nll_loss(output, target)

        # logging
        total_loss += loss.item() * data.size(0)
        total_correct += (output.argmax(dim=1) == target).sum().item()
        total_samples += data.size(0)
    return total_loss / total_samples, total_correct / total_samples


def request_handler(model, data):
    return model(data)


def main() -> None:
    # Load data
    train_loader, test_loader = get_data('mnist', batch_size=64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # # Visualize data
    # images, labels = next(iter(train_loader))
    # print(images.shape)
    # plt.imshow(vutils.make_grid(images, padding=2, normalize=True).permute(1, 2, 0).numpy())
    # plt.show()

    # Create model
    model = Network().to(device)  # Easy mistake: model and data on different devices.

    # Create optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01) # try different optimizers

    # Train model
    train(model, train_loader, optimizer, epochs=5, device=device)

    # Evaluate model
    test_loss, test_acc = evaluate(model, test_loader, device=device)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # 95% acc on MNIST in 25 epochs

    # save model
    torch.save(model.state_dict(), 'mnist_model.pt')

    # load model
    model.load_state_dict(torch.load('mnist_model.pt'))
