import torch
import torch.nn as nn
import math


class ImplicitNetwork(nn.Module):

    def __init__(self, hidden_layers, beta, skip_in=[], weight_norm=False,
            geometric_init=True, bias=1.0):
        super().__init__()
        
        d_in, d_out = 3, 1
        dims = [d_in] + hidden_layers + [d_out]
        self.pc_dim = d_in
        self.d_in = d_in
        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=math.sqrt(math.pi) / math.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, math.sqrt(2) / math.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)
        self.softplus = nn.Softplus(beta=beta)

    def forward(self, input):
        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, input], 1) / math.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x
