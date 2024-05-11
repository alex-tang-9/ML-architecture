import random
import math


class Base:
    def __init__(self):
        self.layers = []

    def Linear(self, in_features, out_features, bias=True):
        layer = Linear(in_features, out_features, bias=bias)
        self.layers.append(layer)
        return layer

    def parameters(self):
        return [
            layer_parameters
            for layer in self.layers
            for layer_parameters in layer.parameters()
        ]

    def __call__(self, x):
        return self.forward(x)

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.grad = 0


class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.neurons = [Neuron(in_features, bias) for _ in range(out_features)]
        print(self.neurons)
        self.bias = bias

    def forward(self, x):
        if self.bias != False:
            if len(self.neurons) == 1:
                return (
                    sum(i * j for i, j in zip(x, self.neurons[0].weights))
                    + self.neurons[0].bias
                )

            else:
                return [
                    sum(i * j for i, j in zip(x, neuron.weights)) + neuron.bias
                    for neuron in self.neurons
                ]
        else:
            if len(self.neurons) == 1:
                return sum(i * j for i, j in zip(x, self.neurons[0].weights))
            else:
                return [
                    sum(i * j for i, j in zip(x, neuron.weights))
                    for neuron in self.neurons
                ]

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        # n_weights = f"[Parameter containing:\n{[weight.item() for neuron in self.neurons for weight in neuron.weights]}]"
        # n_bias = (
        #     f",\n[Parameter containing:\n{[neuron.bias for neuron in self.neurons]}]"
        #     if self.neurons[0].bias != False
        #     else ""
        # )
        # return n_weights + n_bias
        return [
            parameter for neuron in self.neurons for parameter in neuron.parameters()
        ]


class Neuron:
    def __init__(self, in_features, bias=False):
        self.weights = [tensor(random.random()) for _ in range(in_features)]
        self.bias = tensor(random.random()) if bias == True else None

    def __call__(self):
        return self.weights

    def __repr__(self):
        return f"Neuron[weights = {self.weights}]"
        if self.bias != None:
            print(f"Neuron bias = {self.bias}")

    def parameters(self):
        return self.weights + [self.bias]


class tensor:
    def __init__(self, data, _children=(), _op=""):
        assert isinstance(
            data, (int, float, list)
        ), "Data must be an int, float, or list"
        if isinstance(data, (int, float)):
            self.data = data
        else:
            self.data = [tensor(d) for d in data]
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.grad = 0.0
        self.shape = self.shape()

    def __repr__(self):
        return f"tensor({self.data})"

    def __add__(self, other):
        if isinstance(self.data, list) and isinstance(other.data, list):
            out = [s_d + o_d for s_d, o_d in zip(self.data, other.data)]
            return out

        if isinstance(other, tensor) == False:
            other = tensor(other)
        out = tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __call__(self):
        return self.data

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __mul__(self, other):
        if isinstance(other, tensor) == False:
            other = tensor(other)
        out = tensor(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = tensor(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        out = (math.exp(2.0 * x) - 1) / (math.exp(2.0 * x) + 1)
        out = tensor(out, (self,), "tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = tensor(math.exp(x), (self,), "exp")

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

    def shape(self):
        shape = []
        inner_content = self.data
        while True:
            if isinstance(inner_content, list):
                shape.append(len(inner_content))
                inner_content = inner_content[0]
            elif isinstance(inner_content, tensor) and isinstance(
                inner_content.data, list
            ):
                shape.append(len(inner_content.data))
                inner_content = inner_content.data[0]
            else:
                # shape.append(1)
                break
        return shape

    def item(self):
        return self.data




class Embedding:
    def __init__(self):
        pass


class LayerNorm:
    def __init__(self):
        pass


class BatchNorm1d:
    def __init__(self):
        pass


class CrossEntropyLoss:
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def forward(self, x, y):
        # x = x.data
        # y = y.data
        if self.reduction == "mean":
            return sum(-math.log(x[i][y[i]]) for i in range(len(x))) / len(x)
        elif self.reduction == "sum":
            return sum(-math.log(x[i][y[i]]) for i in range(len(x)))

    def __call__(self, x, y):
        return self.forward(x, y)

import numpy as np


def cpmp_qwk(a1, a2, max_rat=3) -> float:
    """

    :param a1: The predicted labels
    :param a2: The ground truth labels
    :param max_rat: The maximum target value

    return: A floating point number with the QWK score
    """
    assert len(a1) == len(a2)
    a1 = np.asarray(a1, dtype=int)
    a2 = np.asarray(a2, dtype=int)

    hist1 = np.zeros((max_rat + 1,))
    hist2 = np.zeros((max_rat + 1,))

    o = 0
    for k in range(a1.shape[0]):
        i, j = a1[k], a2[k]
        hist1[i] += 1
        hist2[j] += 1
        o += (i - j) ** 2 / (max_rat**2)

    e = 0
    for i in range(max_rat + 1):
        for j in range(max_rat + 1):
            e += (hist1[i] / a1.shape[0]) * hist2[j] * ((i - j) ** 2 / (max_rat**2))

    return 1 - o / e
