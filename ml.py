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
            [random.random() for _ in range(in_features)] for _ in range(out_features)
        ]
        if bias == True:
            self.bias = [random.random() for _ in range(out_features)]
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
