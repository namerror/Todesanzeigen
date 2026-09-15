import math
class Value:
    def __init__(self, data, children=(), op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(children)
        self._op = op
        self.label = label
        self._backward = lambda: None

    def __add__(self, other: Value):
        # TODO: coerce scalar, create output, attach closure
        out = Value(self.data + other.data, (self, other), "+")
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out

    def __mul__(self, other):
        # TODO
        out = Value(self.data * other.data, (self, other), "*")
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out

    def __neg__(self):
        out = Value(self.data * -1, (self,), "-")
        def backward():
            self.grad += -1 * out.grad
        out._backward = backward
        return out
    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other), "-")
        def backward():
            self.grad += out.grad
            other.grad += -1 * out.grad
        out._backward = backward
        return out
    def __truediv__(self, other):
        out = Value(self.data / other.data, (self, other), "/")
        def backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad
        out._backward = backward
        return out
    def __pow__(self, exponent):
        out = Value(self.data ** exponent, (self,), f"**{exponent}")
        def backward():
            self.grad += (exponent * self.data ** (exponent - 1)) * out.grad
        out._backward = backward
        return out
    def tanh(self):
        out = Value((math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1), (self,), "tanh")
        def backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = backward
        return out
    def exp(self):
        out = Value(math.exp(self.data), (self,), "exp")
        def backward():
            self.grad += out.data * out.grad
        out._backward = backward
        return out
    def log(self):
        out = Value(math.log(self.data), (self,), "log")
        def backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = backward
        return out