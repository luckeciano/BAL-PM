from torch import nn
import torch
import torch.nn.functional as F
import ast

def get_activation_function(name):
    """Return the specified activation function from the torch.nn module."""
    if name.lower() == 'relu':
        return nn.ReLU()
    elif name.lower() == 'sigmoid':
        return nn.Sigmoid()
    elif name.lower() == 'tanh':
        return nn.Tanh()
    elif name.lower() == 'leakyrelu':
        return nn.LeakyReLU()
    else:
        raise ValueError(f'Unknown activation function: {name}')
    
class FreezableWeight(nn.Module):
    def __init__(self):
        super().__init__()
        self.unfreeze()

    def unfreeze(self):
        self.register_buffer("frozen_weight", None)

    def is_frozen(self):
        """Check if a frozen weight is available."""
        return isinstance(self.frozen_weight, torch.Tensor)

    def freeze(self):
        """Sample from the parameter distribution and freeze."""
        raise NotImplementedError()
    
def DropoutLinear_forward(self, input):
    if not self.is_frozen():
      drop_inp = F.dropout(input, self.p)
      return F.linear(drop_inp, self.weight, self.bias)

    return F.linear(input, self.frozen_weight, self.bias)

def DropoutLinear_freeze(self):
    """Apply dropout with rate `p` to columns of `weight` and freeze it."""
    with torch.no_grad():
        prob = torch.full_like(self.weight[:1, :], 1 - self.p)
        feature_mask = torch.bernoulli(prob) / prob

        frozen_weight = self.weight * feature_mask

    # and store it
    self.register_buffer("frozen_weight", frozen_weight)

class DropoutLinear(nn.Linear, FreezableWeight):
    """Linear layer with dropout on inputs."""
    def __init__(self, in_features, out_features, bias=True, p=0.5):
        super().__init__(in_features, out_features, bias=bias)

        self.p = p

    forward = DropoutLinear_forward

    freeze = DropoutLinear_freeze

class MCDropoutRewardMLP(nn.Module):
    def __init__(self, input_size, layers, activation_fn='tanh', init_func='normal', weight_init=0.01, p=0.5):
        super(MCDropoutRewardMLP, self).__init__()
        self.layers = nn.ModuleList()
        layers = ast.literal_eval(layers)
        
        # Input layer
        self.layers.append(nn.Linear(input_size, layers[0]))
        
        # Hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(get_activation_function(activation_fn))
            self.layers.append(DropoutLinear(layers[i], layers[i+1], p=p))
        
        self.layers.append(get_activation_function(activation_fn))
        # Output layer
        self.layers.append(DropoutLinear(layers[-1], 1, p=p))

        # Weight initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear) or isinstance(layer, DropoutLinear):
                if init_func == 'normal':
                    nn.init.normal_(layer.weight, std=weight_init)
                elif init_func == 'xavier_normal':
                    nn.init.xavier_normal_(layer.weight)
                elif init_func == 'uniform':
                    nn.init.uniform_(layer.weight, a=-weight_init, b=weight_init)
                elif init_func == 'kaiming_normal':
                    nn.init.kaiming_normal_(layer.weight, a=weight_init)
                elif init_func == 'xavier_uniform':
                    nn.init.xavier_uniform_(layer.weight)
                elif init_func == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(layer.weight, a=weight_init)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def unfreeze(self):
        for layer in self.modules():
            if isinstance(layer, FreezableWeight):
                layer.unfreeze()
    
    def freeze(self):
        for layer in self.modules():
            if isinstance(layer, FreezableWeight):
                layer.freeze()

    def sample_function(self, dataset, n_samples=1):
        """Draw a realization of a random function."""

        realizations = []

        for i in range(n_samples):
            self.freeze()
            output = self.forward(dataset)
            realizations.append(output)
            self.unfreeze()

        return torch.stack(realizations, dim=0)
    
    def get_output_with_rep(self, x):
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx == len(self.layers) - 2:
                rep = x.detach()
        return x, rep

