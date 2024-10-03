import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from magnet.utils.data_classes import Status
from magnet.utils.prism.models.cmamba.data_classes import CMambaArgs
from datetime import datetime
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ChannelMixup(nn.Module):
    def __init__(self, sigma=0.5):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            B, V, L = x.shape
            perm = torch.randperm(V)
            lambda_ = torch.normal(mean=0, std=self.sigma, size=(V,)).to(x.device)
            x_mixed = x + lambda_.unsqueeze(1) * x[:, perm]
            return x_mixed
        return x

class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(num_channels, num_channels // reduction_ratio)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(num_channels // reduction_ratio, num_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x).squeeze(-1))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x).squeeze(-1))))
        out = self.sigmoid(avg_out + max_out)
        return out.unsqueeze(-1)

class PatchMamba(nn.Module):
    def __init__(self, args: CMambaArgs):
        super().__init__()
        self.args = args
        self.layers = nn.ModuleList([MambaBlock(args) for _ in range(args.n_layer)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x

class CMambaBlock(nn.Module):
    def __init__(self, args: CMambaArgs):
        super().__init__()
        self.args = args
        self.patch_mamba = PatchMamba(args)
        self.channel_attention = ChannelAttention(args.d_model, args.reduction_ratio)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        x = self.patch_mamba(x)
        attn = self.channel_attention(x.permute(0, 2, 1))
        x = x * attn.permute(0, 2, 1)
        return self.norm(x)

class CMamba(nn.Module):
    def __init__(self, args: CMambaArgs, magnet):
        super().__init__()
        self.args = args
        self.magnet = magnet  # Pass in the magnet object for status updates
        self.channel_mixup = ChannelMixup(args.sigma)
        self.patch_embedding = nn.Linear(args.patch_len * args.num_channels, args.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, args.num_patches, args.d_model))
        
        self.c_mamba_blocks = nn.ModuleList([CMambaBlock(args) for _ in range(args.n_layer)])
        
        self.norm_f = RMSNorm(args.d_model)
        self.output_layer = nn.Linear(args.d_model * args.num_patches, args.num_channels * args.forecast_len)

    def forward(self, input_ids, return_embeddings=False):
        self.magnet.status_callback(Status(datetime.now(), "info", f"Input IDs shape: {input_ids.shape}"))
        
        x = self.channel_mixup(input_ids)
        self.magnet.status_callback(Status(datetime.now(), "info", f"After channel mixup: {x.shape}"))
        
        # Patching
        B, V, L = x.shape
        P = self.args.patch_len
        S = self.args.stride

        # Manual patching
        patches = []
        for i in range(0, L - P + 1, S):
            patch = x[:, :, i:i+P].reshape(B, -1)
            patches.append(patch)
        num_patches = (L - P) // S + 1
        self.magnet.status_callback(Status(datetime.now(), "info", f"Calculated number of patches: {num_patches}"))

        x = torch.stack(patches, dim=1)  # (B, num_patches, V*P)
        self.magnet.status_callback(Status(datetime.now(), "info", f"After patching: {x.shape}"))

        # Patch embedding
        x = self.patch_embedding(x)  # (B, num_patches, d_model)
        self.magnet.status_callback(Status(datetime.now(), "info", f"After patch embedding: {x.shape}"))

        # Adjust positional encoding
        pos_encoding = self.pos_encoding[:, :x.size(1), :]
        self.magnet.status_callback(Status(datetime.now(), "info", f"Positional encoding shape: {pos_encoding.shape}"))

        # Add positional encoding
        x = x + pos_encoding
        self.magnet.status_callback(Status(datetime.now(), "info", f"After positional encoding: {x.shape}"))

        # Apply C-Mamba blocks
        for block in self.c_mamba_blocks:
            x = block(x)
        self.magnet.status_callback(Status(datetime.now(), "info", f"After C-Mamba blocks: {x.shape}"))

        x = self.norm_f(x)
        if return_embeddings:
            return x  # Return the embeddings before the final layer
        self.magnet.status_callback(Status(datetime.now(), "info", f"After norm_f: {x.shape}"))

        # Output layer
        x = x.reshape(x.shape[0], -1)
        self.magnet.status_callback(Status(datetime.now(), "info", f"Before output layer: {x.shape}"))

        logits = self.output_layer(x)
        self.magnet.status_callback(Status(datetime.now(), "info", f"After output layer: {logits.shape}"))

        logits = logits.reshape(-1, self.args.num_channels, self.args.forecast_len)
        self.magnet.status_callback(Status(datetime.now(), "info", f"Final logits shape: {logits.shape}"))

        return logits

class MambaBlock(nn.Module):
    def __init__(self, args: CMambaArgs):
        super().__init__()
        self.args = args

        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)

        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )

        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape
        
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)

        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        
        x = F.silu(x)

        y = self.ssm(x)
        
        y = y * F.silu(res)
        
        output = self.out_proj(y)

        return output

    def ssm(self, x):
        """Runs the SSM."""
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D)
        
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        
        # Discretize continuous parameters (A, B)
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []    
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        
        y = y + u * D
    
        return y

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output
