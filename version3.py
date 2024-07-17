import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Network(nn.Module):
    def __init__(
            self,
            num_inputs: int,
            num_hidden: int,
            num_outputs: int,
    ) -> None:
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # 2 -> 10 -> 1: 2 -> (2 x 10) -> 10 -> (10 x 1) -> 1
        self.hidden_layer = nn.Linear(num_inputs, num_hidden)
        self.output_layer = nn.Linear(num_hidden, num_outputs)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.hidden_layer.forward(x)
        x = torch.tanh(x)
        x = self.output_layer.forward(x)
        return x


def main():
    X = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]).float()

    Y = torch.tensor([
        [0],
        [1],
        [1],
        [0]
    ]).float()

    # hyper-parameters
    lr = 0.01
    num_iters = 5000
    batch_size = 2

    model = Network(2, 3, 1)
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # training loop
    for i in range(num_iters + 1):
        # get data batch
        idxs = np.random.choice(len(X), batch_size, replace=False) # sampling without replacement
        batch_x = X[idxs]
        batch_y = Y[idxs]

        # forward pass
        y_pred = model(batch_x)

        # calculate the loss
        # loss = F.mse_loss(y_pred, batch_y)
        loss = F.binary_cross_entropy_with_logits(y_pred, batch_y)

        # backward pass
        optimizer.zero_grad()  # Caution: why need to do this
        loss.backward()

        # update the model
        optimizer.step()

        # logging
        if i % 100 == 0:
            print(f"Iter: {i}, Loss: {loss:.4f}")

    print("Final Predictions")
    for x, y in zip(X, Y):
        y_pred = model(x)
        print(f"x: {x}, y: {y}, y_pred: {int(y_pred>0.5)}")


if __name__ == '__main__':
    main()
