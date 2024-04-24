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
                return sum(i * j for i, j in zip(x, self.neurons[0].weights)) + self.neurons[0].bias
            
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
        self.weights = [Tensor(random.random()) for _ in range(in_features)]
        self.bias = Tensor(random.random()) if bias == True else None

    def __call__(self):
        return self.weights

    def __repr__(self):
        return f"Neuron[weights = {self.weights}]"
        if self.bias != None:
            print(f"Neuron bias = {self.bias}")

    def parameters(self):
        return self.weights + [self.bias]


class Tensor:
    def __init__(self, data, _children=(), _op=""):
        if isinstance(data, list):
            return [Tensor(i) for i in data]
        self.data = data
        self._prev = set(_children)
        self._backward = lambda: None
        self._op = _op
        self.grad = 0.0

    def __repr__(self):
        return f"tensor({self.data})"

    def __add__(self, other):
        if isinstance(other, Tensor) == False:
            other = Tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

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

    def __rsub__(self, other):
        return (-self) + other

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

    def item(self):
        return self.data
