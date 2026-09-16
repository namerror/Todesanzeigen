import math
class Value:
    def __init__(self, data, children=(), op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(children)
        self._op = op
        self.label = label
        self._backward = lambda: None

    def __add__(self, other):
        # TODO: coerce scalar, create output, attach closure
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        # TODO
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = backward
        return out
    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        out = Value(self.data * -1, (self,), "-")
        def backward():
            self.grad += -1 * out.grad
        out._backward = backward
        return out
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other), "-")
        def backward():
            self.grad += out.grad
            other.grad += -1 * out.grad
        out._backward = backward
        return out
    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other - self
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data / other.data, (self, other), "/")
        def backward():
            self.grad += (1 / other.data) * out.grad
            other.grad += (-self.data / (other.data ** 2)) * out.grad
        out._backward = backward
        return out
    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other / self
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

    def backward(self):
        '''
        Build topological order of the graph, then go one variable at a time and apply the chain rule to get its gradient.
        ''' 

        # the reason for topological order is that we make sure no variable is backpropagated before all of its children have been backpropagated
        from collections import deque
        visited = set()
        topo = deque()
        def dfs(v):
            nonlocal visited
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    dfs(child)
                topo.appendleft(v)

        dfs(self)
        self.grad = 1.0
        for v in topo:
            v._backward()
