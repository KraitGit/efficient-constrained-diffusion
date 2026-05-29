import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, kernel_size=3, activation='SiLU'):
        super().__init__()
        self.activation = getattr(nn, activation)()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        
        time_emb = self.time_mlp(self.activation(t))
        h += time_emb.unsqueeze(-1)
        
        h = self.activation(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        
        return h + self.residual_conv(x)

class TemporalUNet(nn.Module):
    def __init__(self, layers, input_state_dim=14, output_state_dim=None, scale=1.0, activation='SiLU'):
        """
        Args:
            layers (list): [input_channels, h1, h2, ..., output_channels].
                           Example for Robot: [14, 64, 128, 256, 14].
                           Hidden values define the channel sizes for the encoder.
            input_state_dim (int): Number of channels per timestep expected in the input (e.g., 14 or 28 when concatenating position/velocity).
            output_state_dim (int): Number of channels per timestep produced by the network. If None, defaults to input_state_dim.
            scale (float): Scale factor for label (timestep).
            activation (str): Activation function name.
        """
        super().__init__()
        self.scale = scale
        self.input_state_dim = input_state_dim
        self.output_state_dim = input_state_dim if output_state_dim is None else output_state_dim

        channel_mults = layers[1:-1]
        if not channel_mults:
            raise ValueError("TemporalUNet requires at least one hidden channel in `layers`.")
        
        time_dim = channel_mults[0] * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(channel_mults[0]),
            nn.Linear(channel_mults[0], time_dim),
            getattr(nn, activation)(),
            nn.Linear(time_dim, time_dim),
        )

        self.downs = nn.ModuleList()
        in_ch = self.input_state_dim
        for out_ch in channel_mults:
            self.downs.append(
                ResidualBlock(in_ch, out_ch, time_dim, activation=activation)
            )
            in_ch = out_ch
            
        mid_ch = channel_mults[-1]
        self.mid_block1 = ResidualBlock(mid_ch, mid_ch, time_dim, activation=activation)
        self.mid_block2 = ResidualBlock(mid_ch, mid_ch, time_dim, activation=activation)
        
        self.ups = nn.ModuleList()
        decoder_channels = list(reversed(channel_mults))

        for out_ch in decoder_channels:
            self.ups.append(
                ResidualBlock(in_ch + out_ch, out_ch, time_dim, activation=activation)
            )
            in_ch = out_ch

        self.final_conv = nn.Conv1d(in_ch, self.output_state_dim, 1)

    def forward(self, x, label):
        """
        x: (Batch, flattened_dim) -> e.g., (B, 140) for T=10, D=14
        label: (Batch, 1) or (Batch,)
        """
        B, flat_dim = x.shape
        
        time_steps, remainder = divmod(flat_dim, self.input_state_dim)
        if remainder != 0:
            raise ValueError(f"Input dimension {flat_dim} is not divisible by input_state_dim {self.input_state_dim}.")
        x = x.view(B, time_steps, self.input_state_dim).transpose(1, 2)
        
        label = label.reshape(-1) * self.scale
        t = self.time_mlp(label)
        
        skips = []
        h = x
        
        for block in self.downs:
            h = block(h, t)
            skips.append(h)
            
        h = self.mid_block1(h, t)
        h = self.mid_block2(h, t)
        
        for skip, block in zip(reversed(skips), self.ups):
            h = torch.cat([h, skip], dim=1)
            h = block(h, t)
            
        out = self.final_conv(h)
        return out.transpose(1, 2).reshape(B, -1)
