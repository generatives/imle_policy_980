
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleActionGenerator(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        noise_dim=32,
        hidden_dim=256,
        horizon=16,
        n_groups=8
    ):
        """
        Simple network that generates action trajectories from state and noise.

        Args:
            state_dim: Dimension of the state input
            action_dim: Dimension of each action
            noise_dim: Dimension of the noise input vector
            hidden_dim: Size of hidden layers
            horizon: Number of timesteps in the output trajectory
            n_groups: Number of groups for GroupNorm
        """
        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim

        # Initial processing of state and noise inputs
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        self.noise_encoder = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
        )

        # Combined processing
        self.combined_processor = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        # Output layers - generate full trajectory at once
        self.trajectory_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim*2, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, horizon * action_dim)
        )

        print("number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self, state, noise):
        """
        Generate action trajectory from state and noise inputs.

        Args:
            state: (batch_size, state_dim) tensor of state features
            noise: (batch_size, noise_dim) tensor of random noise

        Returns:
            actions: (batch_size, horizon, action_dim) tensor of generated actions
        """
        # Encode inputs
        state_features = self.state_encoder(state)
        noise_features = self.noise_encoder(noise)


        # Combine features
        combined = torch.cat([state_features, noise_features], dim=-1)

        # Process combined features
        processed = self.combined_processor(combined)

        # Generate full trajectory
        trajectory_flat = self.trajectory_generator(processed)

        # Reshape to (batch_size, horizon, action_dim)
        trajectory = trajectory_flat.reshape(-1, self.horizon, self.action_dim)

        return trajectory
    



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


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (2, out_channels, 1))
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        
        #embed = embed.view(embed.shape[0], 2, self.out_channels, 1)

        #embed = embed.reshape(
            #embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out




# class GeneratorConditionalUnet1D(nn.Module):
#     def __init__(self,
#         global_cond_dim,
#         input_dim=12,
#         down_dims=[64,128,256],
#         kernel_size=5,
#         n_groups=8
#         ):
#         """
#         input_dim: Dim of actions.
#         global_cond_dim: Dim of global conditioning applied with FiLM
#           in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
#         diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
#         down_dims: Channel size for each UNet level.
#           The length of this array determines numebr of levels.
#         kernel_size: Conv kernel size
#         n_groups: Number of groups for GroupNorm
#         """

        

        
        
#         super().__init__()
        
#         self.global_cond_dim = global_cond_dim
        
#         cond_dim = down_dims[0]
#         self.sample_proj = nn.Conv1d(12, down_dims[0], kernel_size=1)
#         self.cond_proj = nn.Linear(global_cond_dim, cond_dim)
        
#         all_dims = down_dims
#         in_out = list(zip(all_dims[:-1], all_dims[1:]))
#         mid_dim = all_dims[-1]
        
        
#         self.mid_modules = nn.ModuleList([
#             ConditionalResidualBlock1D(
#                 mid_dim, mid_dim, cond_dim=cond_dim,
#                 kernel_size=kernel_size, n_groups=n_groups
#             ),
#             ConditionalResidualBlock1D(
#                 mid_dim, mid_dim, cond_dim=cond_dim,
#                 kernel_size=kernel_size, n_groups=n_groups
#             ),
#         ])

#         down_modules = nn.ModuleList([])
#         for ind, (dim_in, dim_out) in enumerate(in_out):
#             is_last = ind >= (len(in_out) - 1)
#             down_modules.append(nn.ModuleList([
#                 ConditionalResidualBlock1D(
#                     dim_in, dim_out, cond_dim=cond_dim,
#                     kernel_size=kernel_size, n_groups=n_groups),
#                 ConditionalResidualBlock1D(
#                     dim_out, dim_out, cond_dim=cond_dim,
#                     kernel_size=kernel_size, n_groups=n_groups),
#                 Downsample1d(dim_out) if not is_last else nn.Identity()
#             ]))

#         reversed_in_out = list(reversed(in_out))
#         up_modules = nn.ModuleList([])
#         x_channels = mid_dim  # mid output channels
#         for idx, (dim_in, dim_out) in enumerate(reversed_in_out):
#             skip_channels = dim_out
#             in_ch = x_channels + skip_channels
#             out_ch = dim_out  # not skip_channels
#             is_last = idx == len(reversed_in_out) - 1
#             up_modules.append(nn.ModuleList([
#                 ConditionalResidualBlock1D(in_ch, out_ch, cond_dim=cond_dim,
#                                         kernel_size=kernel_size, n_groups=n_groups),
#                 ConditionalResidualBlock1D(out_ch, out_ch, cond_dim=cond_dim,
#                                         kernel_size=kernel_size, n_groups=n_groups),
#                 Upsample1d(out_ch) if not is_last else nn.Identity()
#             ]))
#             x_channels = out_ch  # next iteration input
            
#         start_dim = up_modules[-1][1].out_channels
#         final_conv = nn.Sequential(
#             Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
#             nn.Conv1d(start_dim, 12, 1),
#         )

#         self.up_modules = up_modules
#         self.down_modules = down_modules
#         self.final_conv = final_conv

#         print("number of parameters: {:e}".format(
#             sum(p.numel() for p in self.parameters()))
#         )

#     def forward(self, global_cond, sample):
#         """
#         x: (B,T,input_dim)
#         global_cond: (B,global_cond_dim)
#         output: (B,T,input_dim)
#         """
#         # (B,T,C)
#         #sample = sample.moveaxis(-1,-2)
#         # (B,C,T)
#         #print("global_cond at forward:", global_cond.shape)

#         x = sample.permute(0, 2, 1).contiguous()
#         x = self.sample_proj(x)

#         global_feature = self.cond_proj(global_cond)
#         #global_feature = global_feature.flatten(1)

#         h = []
#         for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
#             x = resnet(x, global_feature)
#             x = resnet2(x, global_feature)
#             h.append(x)
#             x = downsample(x)

#         for mid_module in self.mid_modules:
#             x = mid_module(x, global_feature)

#         for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
#             skip = h.pop()
#             if skip.shape[2] != x.shape[2]:
#                 x = F.interpolate(x, size=skip.shape[2], mode='linear', align_corners=False)
#             x = torch.cat((x, skip), dim=1)
#             x = resnet(x, global_feature)
#             x = resnet2(x, global_feature)
#             x = upsample(x)

#         x = self.final_conv(x)

#         # (B,C,T)
#         x = x.permute(0,2,1).contiguous()

#         # (B,T,C)
#         return x

class GeneratorConditionalTransformer(nn.Module):
    def __init__(self,
                 global_cond_dim,
                 input_dim,
                 output_dim,
                 embed_dim=768,
                 num_heads=8,
                 num_layers=6,
                 ff_dim=2048,
                 dropout=0.1):
        
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 100, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.global_cond_proj = nn.Linear(global_cond_dim, embed_dim)
        self.output_layer = nn.Linear(embed_dim, output_dim)
        
    def forward(self, x, global_cond):
        batch_size = x.size(0)
        
        x = self.input_embedding(x)
        
        x = x.unsqueeze(1)
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        
        global_cond = global_cond.view(batch_size, -1)
        global_cond = self.global_cond_proj(global_cond)
        global_cond = global_cond.unsqueeze(1)
        x = x + global_cond
        
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        
        x = self.output_layer(x)
        
        x = x.squeeze(1)
        
        return x