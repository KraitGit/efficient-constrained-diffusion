from collections import OrderedDict
import torch
import torch.nn as nn
from src.utils import Kabsch


class EMLP(nn.Module):
    """
    SE(3) Equivariant Multi-Layer Perceptron.

    This model can handle both:
    1. Overdamped systems (position-only input, shape [B, 30]).
    2. Underdamped systems (position and momentum input, shape [B, 60]).

    The output is always the score on the position coordinates.
    """
    def __init__(self, layers, xref, scale=1.0, activation='SiLU'):
        """
        Initializes the EMLP.

        Args:
            layers (list of int): Defines the architecture of the neural network.
                The output size must be natom * 3 (e.g., 30).
                The input size must be set by the user based on the system:
                - For Overdamped: input_dim = (natom * 3) + 1
                - For Underdamped: input_dim = (natom * 3 * 2) + 1
            xref (Tensor): The reference structure for Kabsch alignment. Shape: [natom, 3].
            scale (float): A scaling factor for the time/label input.
            activation (str): The name of the activation function to use.
        """
        super(EMLP, self).__init__()
        self.depth = len(layers) - 1
        self.xref = xref
        self.natom = xref.shape[0]
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

    def forward(self, x, label):
        """
        Forward pass for the EMLP.

        It automatically detects if the input 'x' is for an overdamped or
        underdamped system based on its last dimension.
        """
        is_underdamped = (x.shape[-1] == self.natom * 3 * 2)
        
        if is_underdamped:
            pos = x[:, :self.natom * 3].reshape(x.shape[0], self.natom, 3)
            mom = x[:, self.natom * 3:].reshape(x.shape[0], self.natom, 3)
        else:
            pos = x.reshape(x.shape[0], self.natom, 3)

        label = label.reshape(-1, 1) * self.scale
        
        R, b = Kabsch(pos, self.xref)
        aligned_pos = torch.matmul(pos - b, R.transpose(1, 2))
        
        if is_underdamped:
            aligned_mom = torch.matmul(mom, R.transpose(1, 2))
            state = torch.cat([
                torch.flatten(aligned_pos, start_dim=1),
                torch.flatten(aligned_mom, start_dim=1),
                label
            ], dim=1)
        else:
            state = torch.cat([
                torch.flatten(aligned_pos, start_dim=1),
                label
            ], dim=1)

        nn_output = self.layers(state)

        out_pos_aligned = nn_output.reshape(x.shape[0], self.natom, 3)
        out_pos = torch.matmul(out_pos_aligned, R)
        return torch.flatten(out_pos, start_dim=1)
