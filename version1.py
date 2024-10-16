import math
import numpy as np

from typing import Union

from utils import plot_decision_boundary


class Layer:
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            gain: float = 5/3,
    ) -> None:
        self.data = np.array([0.0])  # temporary variable
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim)  # this is a matrix
        # kaiming init: w * gain / sqrt(fan_in)
        self.weights *= gain / math.sqrt(input_dim)
        self.bias = np.zeros(output_dim)  # this is a vector
        self.grad_w = np.zeros_like(self.weights)
        self.grad_b = np.zeros_like(self.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.data = x
        return np.dot(x, self.weights) + self.bias  # vectorized operations

    def backward(
            self,
            upstream_grad: np.ndarray,
    ) -> np.ndarray:
        # o = x * w
        # do/dx = d(x*w)/dx = w
        # do/dw = d(x*w)/dw = x

        self.grad_w += self.data.T @ upstream_grad
        self.grad_b += upstream_grad.sum(axis=0)
        grad_x = upstream_grad @ self.weights.T
        return grad_x

    def update(
            self,
            lr: float,
    ) -> None:
        self.weights -= lr * self.grad_w  # GD 
        self.bias -= lr * self.grad_b


def tanh(x: Union[float, np.ndarray]) -> np.ndarray:
    return np.tanh(x)


def sigmoid(x: Union[float, np.ndarray]) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


class Network:
    def __init__(
            self,
            num_inputs: int,
            num_hidden: int,
            num_outputs: int,
    ) -> None:
        self.tanh_x = None  # temporary variable
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs

        # 2 -> 10 -> 1: 2 -> (2 x 10) -> 10 -> (10 x 1) -> 1
        self.hidden_layer = Layer(num_inputs, num_hidden)
        self.output_layer = Layer(num_hidden, num_outputs, gain=1)

    def forward(
            self,
            x: np.ndarray,
    ) -> np.ndarray:
        x = self.hidden_layer.forward(x)
        x = tanh(x)
        self.tanh_x = x
        x = self.output_layer.forward(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def backward(
            self,
            grad_L: np.ndarray,
    ) -> None:
        # calculate gradient for the output layer
        grad_o = self.output_layer.backward(grad_L)

        # calculate the gradient for the non-linearity
        grad_tanh = (1 - self.tanh_x ** 2) * grad_o  # local grad * upstream grad

        # calculate the gradient for the hidden layer
        self.hidden_layer.backward(grad_tanh)

    def zero_grad(self):
        self.hidden_layer.grad_w.fill(0)
        self.hidden_layer.grad_b.fill(0)
        self.output_layer.grad_w.fill(0)
        self.output_layer.grad_b.fill(0)

    def update(
            self,
            lr: float,
    ) -> None:
        self.hidden_layer.update(lr)
        self.output_layer.update(lr)


def main():
    np.random.seed(2023)
    
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    Y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

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
        grad_L = 2 * (y_pred - batch_y) / batch_size  # local gradient

        # backward pass
        model.zero_grad()  # Caution: why need to do this?
        model.backward(grad_L)

        # update the model
        model.update(lr)

        # logging
        if i % 100 == 0:
            print(f"Iter: {i}, Loss: {loss:.4f}")

    print("Final Predictions")
    for x, y in zip(X, Y):
        y_pred = model(x)
        print(f"x: {x}, y: {y}, y_pred: {int(y_pred>0.5)}")

    plot_decision_boundary(model, X, Y)


if __name__ == "__main__":
    main()
