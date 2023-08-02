import numpy as np


class NN:
    def __init__(self, arch):
        self.arch = arch

        self._weights = []
        self._biases = []

        for i in range(len(self.arch) - 1):
            self._weights.append(np.random.random(size=(self.arch[i], self.arch[i+1])))
            self._biases.append(np.random.random(size=self.arch[i+1]))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _loss(self, pred, label):
        n = len(pred)
        return 1/n * np.sum((pred - label) ** 2)

    def _finite_diff(self, in_data, out_data, eps):
        w_grads = [np.zeros_like(w_vec) for w_vec in self._weights]
        b_grads = [np.zeros_like(b_vec) for b_vec in self._biases]

        curr_loss = self._loss(self.forward(in_data), out_data)
        print(f"{curr_loss=}")

        for i in range(len(self._weights)):
            for j in range(self._weights[i].shape[0]):
                for k in range(self._weights[i].shape[1]):
                    saved = self._weights[i][j][k].item()
                    self._weights[i][j][k] += eps

                    new_loss = self._loss(self.forward(in_data), out_data)
                    w_grads[i][j][k] = (new_loss - curr_loss) / eps

                    self._weights[i][j][k] = saved

        for i in range(len(self._biases)):
            for j in range(self._biases[i].shape[0]):
                saved = self._biases[i][j].item()
                self._biases[i][j] += eps

                new_loss = self._loss(self.forward(in_data), out_data)
                b_grads[i][j] = (new_loss - curr_loss) / eps

                self._biases[i][j] = saved

        return w_grads, b_grads

    def forward(self, in_data):
        x = in_data.copy()
        for w, b in zip(self._weights, self._biases):
            x = self._sigmoid(x @ w + b)

        return x

    def train(self, data, eps=1e-1, lr=1e-3, epoch=10000):
        in_data = data[:, :-1]
        out_data = data[:, -1]

        for _ in range(epoch):
            w_grads, b_grads = self._finite_diff(in_data, out_data, eps)

            for i in range(len(self._weights)):
                self._weights[i] -= lr * w_grads[i]

            for i in range(len(self._biases)):
                self._biases[i] -= lr * b_grads[i]


def main():
    data = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0],
    ])

    nn = NN([2, 2, 1])
    nn.train(data, epoch=100000)

    for i in data:
        print(f"{i[0]} ^ {i[1]} = {nn.forward(i[:2]).item():.3f}")


if __name__ == '__main__':
    main()
