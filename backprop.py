import numpy as np


class Var:
    def __init__(self, val, requires_grad=False, prev_var1=None, prev_var2=None, prev_op=None):
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
            self.prev_var1.backward(current_grad * self.sigmoid().val * (1 - self.sigmoid()).val)
        elif self.prev_op is None:
            pass
        else:
            assert False, "No such operation"

        if self.requires_grad:
            self.grad += current_grad

    def grad_desc(self, lr):
        self.val -= lr * self.grad
        self.grad = 0

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

            np_weights = np.array(
                [
                    [Var(weights[j][k].item(), requires_grad=True) for k in range(self.arch[i + 1])]
                    for j in range(self.arch[i])
                ],
            )
            self._weights.append(np_weights)

            np_biases = np.array(
                [Var(biases[j].item(), requires_grad=True) for j in range(self.arch[i + 1])],
            )
            self._biases.append(np_biases)

    def forward(self, x):
        def mat_sigmoid(matrix):
            vfunc = np.vectorize(lambda var: var.sigmoid())
            return vfunc(matrix)

        for w, b in zip(self._weights, self._biases):
            x = mat_sigmoid(x @ w + b)

        return x

    def step(self, lr=1e-3):
        vfunc = np.vectorize(lambda var: var.grad_desc(lr))
        for i in range(len(self._weights)):
            vfunc(self._weights[i])
            vfunc(self._biases[i])


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

    for i in range(20000):
        pred = model.forward(in_data).squeeze()

        n = pred.shape[0]
        loss = 1 / n * np.sum(np.square(pred - label))
        loss.backward()

        model.step(lr=1)

        if (i + 1) % 100 == 0:
            print(f"epoch: {i+1} loss: {loss}")

    for i in data:
        print(f"{i[0]} ^ {i[1]} = {model.forward(np.expand_dims(i[:2], 0)).item().val:.3f}")


if __name__ == "__main__":
    main()
