import torch


class Conv2d:
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        # Initialize convolution parameters
        self.kernel_size = kernel_size
        self.kernel = torch.randn(
            output_channels, input_channels, kernel_size, kernel_size
        )
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.padding = padding
        self.bias = torch.randn(output_channels) if use_bias else None

    def forward(self, x):
        n, c, h, w = x.shape
        # Padding for top and bottom
        verticalPadding = torch.zeros((n, c, self.padding, w))
        x = torch.cat(
            (verticalPadding, x, verticalPadding), dim=2
        )  # Concatenate along height dimension
        # Padding for left and right
        horizontalPadding = torch.zeros((n, c, (h + self.padding * 2), self.padding))
        x = torch.cat(
            (horizontalPadding, x, horizontalPadding), dim=3
        )  # Concatenate along width dimension
        out_width = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_heigh = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        out = torch.zeros((n, self.out_channels, out_heigh, out_width))
        for ni in range(n):
            for hi in range(out_heigh):
                hi_i = hi * self.stride
                for wi in range(out_width):
                    wi_i = wi * self.stride
                    for ci in range(self.out_channels):
                        out[ni, ci, hi, wi] = (
                            x[
                                ni,
                                :,
                                hi_i : hi_i + self.kernel_size,
                                wi_i : wi_i + self.kernel_size,
                            ]
                            * self.kernel[ci]
                        ).sum()
                        if self.bias is not None:
                            out[ni, ci, hi, wi] += self.bias[ci]
        return out

    def __call__(self, x):
        return self.forward(x)


class RNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        nonlinearity="tanh",
        bias=True,
        batch_first=False,
    ):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first
        self.bias = bias

        # Weight matrices and bias vectors
        self.weights_ih = nn.ParameterList(
            [
                nn.Parameter(torch.randn(hidden_size, input_size))
                for layer in range(num_layers)
            ]
        )
        self.weights_hh = nn.ParameterList(
            [
                nn.Parameter(torch.randn(hidden_size, hidden_size))
                for _ in range(num_layers)
            ]
        )
        if bias:
            self.bias_ih = nn.ParameterList(
                [nn.Parameter(torch.randn(hidden_size)) for _ in range(num_layers)]
            )
            self.bias_hh = nn.ParameterList(
                [nn.Parameter(torch.randn(hidden_size)) for _ in range(num_layers)]
            )
        else:
            self.bias_ih = [None] * num_layers
            self.bias_hh = [None] * num_layers

        # Non-linearity
        if nonlinearity == "tanh":
            self.activation = torch.tanh
        elif nonlinearity == "relu":
            self.activation = torch.relu
        else:
            raise ValueError("Unsupported nonlinearity")

    def forward(self, x, h_0=None):
        # Handle batch_first
        if self.batch_first:
            x = x.transpose(
                0, 1
            )  # Convert batch_size x seq_len x features to seq_len x batch_size x features

        seq_len, batch_size, _ = x.size()

        if h_0 is None:
            h_0 = self.init_hidden(batch_size)
        h_t_minus_1 = h_0
        h_t = h_0
        output = []
        for t in range(seq_len):
            current_input = x[t]
            for layer in range(self.num_layers):
                combined = (
                    current_input @ self.weights_ih[layer].T
                    + (self.bias_ih[layer] if self.bias else 0)
                    + h_t_minus_1[layer] @ self.weights_hh[layer].T
                    + (self.bias_hh[layer] if self.bias else 0)
                )
                h_t[layer] = self.activation(combined)

            output.append(h_t[-1])
            h_t_minus_1 = h_t

        # Stack outputs to make them seq_len x batch_size x hidden_size
        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(
                0, 1
            )  # Convert back to batch_size x seq_len x features
        return output, h_t

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class BatchNorm1d:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True
        # parameters (trained with backprop)
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # buffers (trained with a running 'momentum update')
        self.running_mean = torch.zeros(dim)
        self.running_var = torch.ones(dim)

    def __call__(self, x):
        # calculate the forward pass
        if self.training:
            xmean = x.mean(0, keepdim=True)  # batch mean
            xvar = x.var(0, keepdim=True)  # batch variance
        else:
            xmean = self.running_mean
            xvar = self.running_var
        xhat = (x - xmean) / (xvar + self.eps) ** 0.5
        self.out = self.gamma * xhat + self.beta
        # update the buffers
        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    self.momentum * xmean + (1 - self.momentum) * self.running_mean
                )
                self.running_var = (
                    self.momentum * xvar + (1 - self.momentum) * self.running_var
                )
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class LayerNorm:

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        last_dim = x.ndim - 1
        # calculate the forward pass
        xmean = x.mean(last_dim, keepdim=True)  # batch mean
        xvar = x.var(last_dim, keepdim=True)  # batch variance
        xhat = (x - xmean) / (xvar + self.eps) ** 0.5
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
