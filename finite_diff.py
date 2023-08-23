import numpy as np


class NN:
    def __init__(self, arch):
        self.arch = arch

        self._weights = []
        self._biases = []

        for i in range(len(self.arch) - 1):
            self._weights.append(np.random.random(size=(self.arch[i], self.arch[i + 1])))
            self._biases.append(np.random.random(size=self.arch[i + 1]))

    def _relu(self, x):
        return np.maximum(0, x)

    def forward(self, in_data):
        x = in_data.copy()
        for w, b in zip(self._weights, self._biases):
            x = self._relu(x @ w + b)

        return x

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

    def step(self, inp, label, loss_fn, lr=1e-4, eps=1e-1):
        loss, w_grads, b_grads = self._finite_diff(inp, label, loss_fn, eps)

        for i in range(len(self._weights)):
            self._weights[i] -= lr * w_grads[i]

        for i in range(len(self._biases)):
            self._biases[i] -= lr * b_grads[i]

        return loss


def generate_adder_data(n_samples=32, max_value=50):
    a = np.random.randint(0, max_value, n_samples)
    b = np.random.randint(0, max_value, n_samples)
    sum = a + b

    return np.column_stack((a, b)), sum


def main():
    model = NN([2, 16, 16, 1])

    def loss_fn(pred, label):
        n = len(pred)
        return 1 / n * np.sum((pred - label) ** 2)

    for i in range(10000):
        inp, label = generate_adder_data(n_samples=32)
        loss = model.step(inp, label, loss_fn, lr=1e-4, eps=1e-2)

        if (i + 1) % 100 == 0:
            print(f"epoch: {i+1} loss: {loss}")

    print("==========")


if __name__ == "__main__":
    main()
