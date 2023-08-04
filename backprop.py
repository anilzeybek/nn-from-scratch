import numpy as np


class Var:
    def __init__(
        self, val, requires_grad=False, prev_var1=None, prev_var2=None, prev_op=None
    ):
        self.val = val
        self.requires_grad = requires_grad
        if requires_grad:
            self.grad = 0

        self.prev_var1 = prev_var1
        self.prev_var2 = prev_var2
        self.prev_op = prev_op

    def __mul__(self, other):
        if not isinstance(other, Var):
            other = Var(other)

        return Var(self.val * other.val, prev_var1=self, prev_var2=other, prev_op="mul")

    def __rmul__(self, other):
        return self.__mul__(other)

    def square(self):
        return Var(self.val**2, prev_var1=self, prev_op="square")

    def __add__(self, other):
        if not isinstance(other, Var):
            other = Var(other)

        return Var(self.val + other.val, prev_var1=self, prev_var2=other, prev_op="add")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        if not isinstance(other, Var):
            other = Var(other)

        return Var(other.val - self.val, prev_var1=self, prev_var2=other, prev_op="sub")

    def __neg__(self):
        return Var(-self.val, prev_var1=self, prev_op="neg")

    def sigmoid(self):
        return Var(1 / (1 + np.exp(-self.val)), prev_var1=self, prev_op="sigmoid")

    def backward(self, current_grad=1):
        if self.prev_op == "square":
            self.prev_var1.backward(current_grad * 2 * self.prev_var1.val)
        elif self.prev_op == "add":
            self.prev_var1.backward(current_grad)
            self.prev_var2.backward(current_grad)
        elif self.prev_op == "sub":
            self.prev_var1.backward(-current_grad)
            self.prev_var2.backward(current_grad)
        elif self.prev_op == "neg":
            self.prev_var1.backward(-current_grad)
        elif self.prev_op == "mul":
            self.prev_var1.backward(current_grad * self.prev_var2.val)
            self.prev_var2.backward(current_grad * self.prev_var1.val)
        elif self.prev_op == "sigmoid":
            self.prev_var1.backward(
                current_grad * self.sigmoid().val * (1 - self.sigmoid()).val
            )
        elif self.prev_op is None:
            pass
        else:
            assert False, "No such operation"

        if self.requires_grad:
            self.grad += current_grad

    def __repr__(self):
        return str(self.val)


class NN:
    def __init__(self, arch):
        self.arch = arch

        self._weights = []
        self._biases = []

        for i in range(len(self.arch) - 1):
            weights = np.random.random(size=(self.arch[i], self.arch[i + 1]))
            biases = np.random.random(size=self.arch[i + 1])

            self._weights.append(
                [[0 for _ in range(self.arch[i + 1])] for _ in range(self.arch[i])]
            )
            self._biases.append([0 for _ in range(self.arch[i + 1])])

            for j in range(self.arch[i]):
                for k in range(self.arch[i + 1]):
                    self._weights[i][j][k] = Var(
                        weights[j][k].item(), requires_grad=True
                    )

            for j in range(self.arch[i + 1]):
                self._biases[i][j] = Var(biases[j].item(), requires_grad=True)

    def _loss(self, pred, label):
        loss = np.sum(np.square(label - pred))
        loss *= 1 / (len(pred))
        return loss

    def _mat_sigmoid(self, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] = matrix[i][j].sigmoid()

        return matrix

    def forward(self, in_data):
        x = in_data.copy()
        for w, b in zip(self._weights, self._biases):
            x = self._mat_sigmoid(x @ w + b)

        return x

    def zero_grad(self):
        for i in range(len(self._weights)):
            for j in range(len(self._weights[i])):
                for k in range(len(self._weights[i][j])):
                    self._weights[i][j][k].grad = 0

        for i in range(len(self._biases)):
            for j in range(len(self._biases[i])):
                self._biases[i][j].grad = 0

    def train(self, data, lr=1e-1, epoch=10000):
        in_data = data[:, :-1]
        out_data = data[:, -1]

        for _ in range(epoch):
            pred = self.forward(in_data).squeeze()
            loss = self._loss(pred, out_data)
            print(loss)

            self.zero_grad()
            loss.backward()

            for i in range(len(self.arch) - 1):
                for j in range(self.arch[i]):
                    for k in range(self.arch[i + 1]):
                        self._weights[i][j][k].val -= lr * self._weights[i][j][k].grad

                for j in range(self.arch[i + 1]):
                    self._biases[i][j].val -= lr * self._biases[i][j].grad


def main():
    data = np.array(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.float32,
    )

    nn = NN([2, 2, 1])
    nn.train(data, lr=1e-1, epoch=100000)

    for i in data:
        print(f"{i[0]} ^ {i[1]} = {nn.forward(np.expand_dims(i[:2], 0))[0][0].val:.3f}")


if __name__ == "__main__":
    main()
