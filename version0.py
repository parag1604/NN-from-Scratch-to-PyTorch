import math
import numpy as np

from utils import plot_decision_boundary


class Perceptron:
    def __init__(self, input_dim: int) -> None:
        self.input_dim = input_dim
        self.weights = np.zeros(input_dim)  # this is a vector
        self.bias = 0

    def predict(self, x: np.ndarray) -> int:
        weighted_sum = np.dot(self.weights, x) + self.bias  # w.T * x + b
        return int(weighted_sum > 0)

    def __call__(self, x: np.ndarray) -> int:
        return self.predict(x)

    def fit(
            self,
            X: np.ndarray,
            Y: np.ndarray,
            epochs: int = 10,
            learning_rate: float = 0.1
    ) -> None:
        # epoch is basically how many times we are visiting all the datapoints
        for epoch in range(epochs):
            for x, y in zip(X, Y):
                y_pred = self.predict(x)
                update = learning_rate * (y - y_pred)  # this is the update rule for y = {0, 1}
                self.weights += update * x
                self.bias += update


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

    model = Perceptron(input_dim=2)
    model.fit(X, Y, epochs=10000, learning_rate=0.1)

    print("Final Predictions")
    for x, y in zip(X, Y):
        y_pred = model(x)
        print(f"x: {x}, y: {y}, y_pred: {int(y_pred>0.5)}")

    plot_decision_boundary(model, X, Y)


if __name__ == '__main__':
    main()
