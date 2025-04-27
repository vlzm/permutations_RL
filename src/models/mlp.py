import torch
import torch.nn as nn
import numpy as np

class PermutationMLP(nn.Module):
    """
        layer_sizes (list of int): 
            A list defining the number of neurons in each hidden layer. The length of the list determines 
            the number of hidden layers, and each element specifies the size of that layer.
            Example: [32, 16, 8] creates 3 hidden layers with 32, 16, and 8 neurons respectively and nn.ReLU() activations between.
        """
    def __init__(self, input_size, hidden_dims, num_classes_for_one_hot):
        super(PermutationMLP, self).__init__()
        self.num_classes_for_one_hot = num_classes_for_one_hot
        self.input_layer_size_for_one_hot = input_size * num_classes_for_one_hot
        
        layers = []
        in_features = self.input_layer_size_for_one_hot
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):            
        x = torch.nn.functional.one_hot(x.long(), num_classes=self.num_classes_for_one_hot) \
        .float().flatten(start_dim=-2)
        return self.layers(x)