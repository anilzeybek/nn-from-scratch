import math


class Var:
    def __init__(self, val, requires_grad=False, prev_var1=None, prev_var2=None, prev_op=None):
        self.val = val
        self.requires_grad = requires_grad

        self.prev_var1 = prev_var1
        self.prev_var2 = prev_var2
        self.prev_op = prev_op

    def mul(self, other):
        return Var(self.val * other.val, prev_var1=self, prev_var2=other, prev_op='mul')

    def square(self):
        return Var(self.val ** 2, prev_var1=self, prev_op='square')

    def add(self, other):
        return Var(self.val + other.val, prev_var1=self, prev_var2=other, prev_op='add')

    def sigmoid(self):
        return Var(1/(1+math.exp(-self.val)), prev_var1=self, prev_op='sigmoid')

    def backward(self, current_grad=1):
        if self.prev_op == 'square':
            self.prev_var1.backward(current_grad*2*self.prev_var1.val)
        elif self.prev_op == 'add':
            self.prev_var1.backward(current_grad)
            self.prev_var2.backward(current_grad)
        elif self.prev_op == 'mul':
            self.prev_var1.backward(current_grad*self.prev_var2.val)
            self.prev_var2.backward(current_grad*self.prev_var1.val)
        elif self.prev_op == 'sigmoid':
            assert False, "Not implemented yet"
        elif self.prev_op is None:
            pass
        else:
            assert False, "No such operation"

        if self.requires_grad:
            self.grad = current_grad

    def __str__(self):
        return str(self.val)


def main():
    x = Var(4, requires_grad=True)
    y = Var(3, requires_grad=True)

    result = x.square().add(Var(-3)).mul(y)
    result.backward()

    print(x.grad)
    # x' = 24

    print(y.grad)
    # y' = 13


if __name__ == "__main__":
    main()
