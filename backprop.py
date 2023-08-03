import numpy as np
 

# class Tensor:
#     def __init__(self, arr):
#         self.arr = arr


#     def __matmul__(self, other):
#         result = [[0 for _ in range(len(other.arr[0]))] for _ in range(len(self.arr))]

#         # perform matrix multiplication
#         for i in range(len(self.arr)):
#             for j in range(len(other.arr[0])):
#                 for k in range(len(other.arr)):
#                     result[i][j] += self.arr[i][k] * other.arr[k][j]

#         return Tensor(result)

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

        return Var(self.val * other.val, prev_var1=self, prev_var2=other, prev_op='mul')

    def __rmul__(self, other):
        return self.__mul__(other)

    def square(self):
        return Var(self.val ** 2, prev_var1=self, prev_op='square')

    def __add__(self, other):
        if not isinstance(other, Var):
            other = Var(other)

        return Var(self.val + other.val, prev_var1=self, prev_var2=other, prev_op='add')

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self.__add__(-other)
    
    def __rsub__(self, other):
        if not isinstance(other, Var):
            other = Var(other)

        return Var(other.val - self.val, prev_var1=self, prev_var2=other, prev_op='sub')

    def __neg__(self):
        return Var(-self.val, prev_var1=self, prev_op='neg')

    def sigmoid(self):
        return Var(1/(1+np.exp(-self.val)), prev_var1=self, prev_op='sigmoid')

    def backward(self, current_grad=1):
        if self.prev_op == 'square':
            self.prev_var1.backward(current_grad*2*self.prev_var1.val)
        elif self.prev_op == 'add':
            self.prev_var1.backward(current_grad)
            self.prev_var2.backward(current_grad)
        elif self.prev_op == 'sub':
            self.prev_var1.backward(-current_grad)
            self.prev_var2.backward(current_grad)
        elif self.prev_op == 'neg':
            self.prev_var1.backward(-current_grad)
        elif self.prev_op == 'mul':
            self.prev_var1.backward(current_grad*self.prev_var2.val)
            self.prev_var2.backward(current_grad*self.prev_var1.val)
        elif self.prev_op == 'sigmoid':
            self.prev_var1.backward(current_grad*self.sigmoid().val*(1-self.sigmoid()).val)
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
            weights = np.random.random(size=(self.arch[i], self.arch[i+1]))
            biases = np.random.random(size=self.arch[i+1])

            self._weights.append([[0]*self.arch[i+1]]*self.arch[i])
            self._biases.append([0]*self.arch[i+1])

            for j in range(self.arch[i]):
                for k in range(self.arch[i+1]):
                    self._weights[i][j][k] = Var(weights[j][k].item(), requires_grad=True)

            for j in range(self.arch[i+1]):
                self._biases[i][j] = Var(biases[j].item(), requires_grad=True)

    def _loss(self, pred, label):
        n = len(pred)
        loss = 0
        for i in range(n):
            loss += (pred[i][0] - label[i].item()).square()

        loss = 1/n* loss
        return loss

    def _matmul(self, A, B):
        num_rows_A, num_cols_A = len(A), len(A[0])
        num_rows_B, num_cols_B = len(B), len(B[0])

        if num_cols_A != num_rows_B:
            print("Matrix multiplication is not possible.")
            return None

        result = [[0 for _ in range(num_cols_B)] for _ in range(num_rows_A)]

        for i in range(num_rows_A):
            for j in range(num_cols_B):
                for k in range(num_cols_A):  # or num_rows_B, as they are equal
                    result[i][j] += A[i][k] * B[k][j]
        return result

    def _mat_vec_add(self, matrix, vector):
        # "Broadcast" the vector to match the shape of the matrix
        vector_broadcasted = [vector[i % len(vector)] for i in range(len(matrix))]

        # Perform row-wise addition
        result = [[matrix[i][j] + vector_broadcasted[i] for j in range(len(matrix[0]))] for i in range(len(matrix))]
        return result

    def _mat_sigmoid(self, matrix):
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                matrix[i][j] = matrix[i][j].sigmoid()

        return matrix

    def forward(self, in_data):
        x = in_data.copy()
        for w, b in zip(self._weights, self._biases):
            x = self._mat_sigmoid(self._mat_vec_add(self._matmul(x, w), b))

        return x

    def train(self, data, lr=1e-1, epoch=10000):
        in_data = data[:, :-1]
        out_data = data[:, -1]

        for _ in range(epoch):
            pred = self.forward(in_data)
            loss = self._loss(pred, out_data)
            print(loss)

            loss.backward()

            for i in range(len(self.arch) - 1):
                for j in range(self.arch[i]):
                    for k in range(self.arch[i+1]):
                        self._weights[i][j][k].val -= lr * self._weights[i][j][k].grad

                for j in range(self.arch[i+1]):
                    self._biases[i][j].val -= lr * self._biases[i][j].grad


def main():
    data = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ])

    nn = NN([2, 2, 1])
    nn.train(data, lr=1e-1, epoch=100)

    for i in data:
        print(f"{i[0]} ^ {i[1]} = {nn.forward(np.expand_dims(i[:2], 1)).item():.3f}")



if __name__ == "__main__":
    main()
