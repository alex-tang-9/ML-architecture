import random


class Base:
    def __init__(self):
        self.layers = []

    def Linear(self, in_features, out_features):
        layer = Linear(in_features, out_features)
        self.layers.append(layer)
        return layer

    def parameters(self):
        for layer in self.layers:
            layer.parameters()

    def __call__(self, x):
        return self.forward(x)


class Linear:
    def __init__(self, in_features, out_features, bias=False):
        self.weight = [
            [Tensor(random.random()) for _ in range(in_features)] for _ in range(out_features)
        ]
        print(self.weight)
        if bias == True:
            self.bias = [Tensor(random.random() for _ in range(out_features))]
        else:
            self.bias = None

    def forward(self, x):
        if self.bias != None:
            return [
                sum(i * j for i, j in zip(x, w_col)) + b
                for w_col, b in zip(self.weight, self.bias)
            ]
        else:
            return [sum(i * j for i, j in zip(x, w_col)) for w_col in self.weight]

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        if self.bias != None:
            print(self.weight)
            print(self.bias)
        else:
            print(self.weight)


import math


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        if (isinstance(data, list)):
            return [Tensor(i) for i in data]
        self.data = data
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.grad = 0.0

    def __repr__(self):
        return f"Tensor[data = {self.data}]"

    def __add__(self, other):
        if isinstance(other, Tensor) == False:
            other = Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        if isinstance(other, Tensor) == False:
            other = Tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Tensor(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def tanh(self):
        x = self.data
        out = (math.exp(2.0 * x) - 1) / (math.exp(2.0 * x) + 1)
        out = Tensor(out, (self,), "tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Tensor(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        self.grad = 1.0

        nodes = []
        added_nodes = set()

        def get_child_nodes(v):
            if v not in added_nodes:
                added_nodes.add(v)
                for child in v._prev:
                    get_child_nodes(child)
                nodes.append(v)

        get_child_nodes(self)

        for node in reversed(nodes):
            node._backward()
            
    def shape():
        return None