from collections import OrderedDict
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layers, scale=1.0, activation='SiLU'):
        super(MLP, self).__init__()
        self.depth = len(layers) - 1
        self.activation = getattr(torch.nn, activation) 
        self.scale = scale

        layer_list = []
        for i in range(self.depth - 1):
            layer_list.append(
                ("layer_%d" % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(("activation_%d" % i, self.activation()))
        layer_list.append(
            ("layer_%d" % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layer_dict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layer_dict)

    def forward(self, x, label=None):
        if label is not None:
            label = label.reshape(-1, 1) * self.scale
            state = torch.cat((x, label), dim=1)
        else:
            state = x
        out = self.layers(state)
        return out
