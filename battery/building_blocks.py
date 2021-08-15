import torch
import torch.nn as nn
import torch.distributions as D
# from torch.nn.utils import weight_norm

class MLP(nn.Module):
    def __init__(self, layer_sizes: list):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes)-1):
            layers += [nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.LeakyReLU()]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CGMM(nn.Module):
    # Conditional Gaussian Mixture Model
    # Yield an Gaussian mixture distribution conditioned on the input.
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int, mixture_num: int):
        super().__init__()
        self.feature = MLP([input_dim] + [hidden_dim] * num_hidden_layers)
        self.mixture = nn.Sequential(nn.Linear(hidden_dim, mixture_num))
        self.mean = nn.Linear(hidden_dim, mixture_num * output_dim)
        self.std = nn.Sequential(nn.Linear(hidden_dim, mixture_num * output_dim), nn.Softplus())

        self.mixture_num = mixture_num
        self.output_dim = output_dim
    
    def forward(self, inputs: torch.Tensor) -> D.Distribution:
        features = self.feature(inputs)
        mixtures = self.mixture(features)
        means = self.mean(features).reshape((features.shape[:-1] + (self.mixture_num, self.output_dim)))
        stds = self.std(features).reshape((features.shape[:-1] + (self.mixture_num, self.output_dim)))

        return D.MixtureSameFamily(
            D.Categorical(logits=mixtures),
            D.Independent(D.Normal(means, stds), 1)
        )

class CGM(nn.Module):
    # Conditional Gaussian Model
    # Yield an Gaussian distribution conditioned on the input.
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_hidden_layers: int):
        super().__init__()
        self.feature = MLP([input_dim] + [hidden_dim] * num_hidden_layers)
        self.mean = nn.Linear(hidden_dim, output_dim)
        self.std = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Softplus())
        self.output_dim = output_dim
    
    def forward(self, inputs: torch.Tensor) -> D.Distribution:
        features = self.feature(inputs)
        means = self.mean(features)
        stds = self.std(features).exp()
        return D.Independent(D.Normal(means, stds), 1)

class LGM(nn.Module):
    # Linear Gaussian Model
    # Yield an Gaussian distribution conditioned on the input.
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.mean = nn.Linear(input_dim, output_dim)
        self.std = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Softplus())
        self.output_dim = output_dim
    
    def forward(self, inputs: torch.Tensor) -> D.Distribution:
        means = self.mean(inputs)
        stds = self.std(inputs)
        return D.Independent(D.Normal(means, stds), 1)