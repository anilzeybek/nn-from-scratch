import numpy as np


class NN:
    def __init__(self, arch):
        self.arch = arch

        self._weights = []
        self._biases = []

        for i in range(len(self.arch) - 1):
            self._weights.append(np.random.rand(self.arch[i], self.arch[i + 1]))
            self._biases.append(np.random.rand(self.arch[i + 1]))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, in_data):
        x = in_data.copy()
        for w, b in zip(self._weights, self._biases):
            x = self._sigmoid(x @ w + b)

        return x.squeeze()

    def _finite_diff(self, in_data, out_data, loss_fn, eps):
        w_grads = [np.zeros_like(w_vec) for w_vec in self._weights]
        b_grads = [np.zeros_like(b_vec) for b_vec in self._biases]

        curr_loss = loss_fn(self.forward(in_data), out_data)
        for i in range(len(self._weights)):
            for j in range(self._weights[i].shape[0]):
                for k in range(self._weights[i].shape[1]):
                    saved = self._weights[i][j][k].item()
                    self._weights[i][j][k] += eps

                    new_loss = loss_fn(self.forward(in_data), out_data)
                    w_grads[i][j][k] = (new_loss - curr_loss) / eps

                    self._weights[i][j][k] = saved

        for i in range(len(self._biases)):
            for j in range(self._biases[i].shape[0]):
                saved = self._biases[i][j].item()
                self._biases[i][j] += eps

                new_loss = loss_fn(self.forward(in_data), out_data)
                b_grads[i][j] = (new_loss - curr_loss) / eps

                self._biases[i][j] = saved

        return curr_loss, w_grads, b_grads

    def step(self, inp, label, loss_fn, lr=1e-4, eps=1e-4):
        loss, w_grads, b_grads = self._finite_diff(inp, label, loss_fn, eps)

        for i in range(len(self._weights)):
            self._weights[i] -= lr * w_grads[i]

        for i in range(len(self._biases)):
            self._biases[i] -= lr * b_grads[i]

        return loss


def main():
    np.random.seed(0)
    data = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.float32,
    )

    model = NN([2, 2, 1])
    in_data = data[:, :-1]
    label = data[:, -1]

    def loss_fn(pred, label):
        n = pred.shape[0]
        return 1 / n * np.sum((pred - label) ** 2)

    for i in range(2000):
        loss = model.step(in_data, label, loss_fn, lr=1)

        if (i + 1) % 100 == 0:
            print(f"epoch: {i+1} loss: {loss}")

    for i in data:
        print(f"{i[0]} ^ {i[1]} = {model.forward(np.expand_dims(i[:2], 0)).item():.3f}")


if __name__ == "__main__":
    main()
