import math
import torch
import numpy as np
import torch.nn as nn


class Layer(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            gain: float = 5/3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = nn.Parameter(torch.randn(input_dim, output_dim))
        self.weights.data *= gain / math.sqrt(input_dim)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights + self.bias

    def update(
            self,
            lr: float,
    ) -> None:
        self.weights.data -= lr * self.weights.grad
        self.bias.data -= lr * self.bias.grad

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
        self.hidden_layer = Layer(num_inputs, num_hidden)
        self.output_layer = Layer(num_hidden, num_outputs)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.hidden_layer.forward(x)
        x = torch.tanh(x)
        x = self.output_layer.forward(x)
        return x

    def zero_grad(self):
        self.hidden_layer.weights.grad = None
        self.hidden_layer.bias.grad = None
        self.output_layer.weights.grad = None
        self.output_layer.bias.grad = None

    def update(
            self,
            lr: float,
    ) -> None:
        self.hidden_layer.update(lr)
        self.output_layer.update(lr)


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

    # model = Perceptron(input_dim=2)
    # model.fit(X, Y, epochs=10000, learning_rate=0.1)

    # hyper-parameters
    lr = 0.01
    num_iters = 5000
    batch_size = 2

    model = Network(2, 3, 1)

    # training loop
    for i in range(num_iters + 1):
        # get data batch
        idxs = np.random.choice(len(X), batch_size, replace=False) # sampling without replacement
        batch_x = X[idxs]
        batch_y = Y[idxs]

        # forward pass
        y_pred = model(batch_x)

        # calculate the loss
        loss = ((y_pred - batch_y) ** 2).mean()  # MSE Loss

        # backward pass
        model.zero_grad()  # Caution: why need to do this
        loss.backward()

        # update the model
        model.update(lr)

        # logging
        if i % 100 == 0:
            print(f"Iter: {i}, Loss: {loss:.4f}")

    print("Final Predictions")
    for x, y in zip(X, Y):
        y_pred = model(x)
        print(f"x: {x}, y: {y}, y_pred: {int(y_pred>0.5)}")


if __name__ == '__main__':
    main()
