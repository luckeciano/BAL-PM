import torch
from torch import nn
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
    
def weight_init_func(layer, init_func, weight_init):
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


class VariationalEncoder(nn.Module):

    def __init__(self, input_size, layers, activation_fn='tanh', init_func='normal', weight_init=0.01):
        super(VariationalEncoder, self).__init__()
        self.layers = nn.ModuleList()
        layers = ast.literal_eval(layers)
        
        # Input layer
        self.layers.append(nn.Linear(input_size, layers[0]))
        
        # Hidden layers
        for i in range(len(layers) - 1):
            self.layers.append(get_activation_function(activation_fn))
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
        
        self.layers.append(get_activation_function(activation_fn))
        # Output layer
        self.mean = nn.Linear(layers[-1], 1)
        self.log_var = nn.Linear(layers[-1], 1)

        # Weight initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                weight_init_func(layer, init_func, weight_init)
                

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        mean = self.mean(x)
        log_var = self.log_var(x)

        var = torch.exp(0.5 * log_var)
        eps = torch.randn_like(var)

        z = mean + var * eps

        return z, mean, log_var


# Usage:
# model = MLP(input_size=10, output_size=2, layers=[64, 32, 16], weight_init=0.1)

