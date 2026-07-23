import torch
from torch import nn
import torch.nn.functional as F
import math
import mlflow
import torchvision
from torchmetrics.image.fid import FrechetInceptionDistance
import os

import torch.nn.utils.parametrizations as P


class OpticalConv2d(nn.Conv2d):
    def __init__(self, *args, weight_noise_std=0.03, prev_noise_std=0.05, post_noise_std=0.05,
                 weight_clip=1.0, apply_sn=True, **kwargs):
        # 1. 初始化原生的 Conv2d 参数
        super().__init__(*args, **kwargs)
        self.weight_noise_std = weight_noise_std
        self.prev_noise_std = prev_noise_std
        self.post_noise_std = post_noise_std
        self.weight_clip = weight_clip

        # 2. 直接在内部给自己套上谱归一化
        # 这里的 name='weight' 会将原生的 self.weight 替换为谱归一化后的版本
        if apply_sn:
            P.spectral_norm(self, name='weight')

    def forward(self, input):

        if self.training:
            noisy_input = input + torch.randn_like(input) * self.prev_noise_std
        else:
            noisy_input = input

        if self.training:
            noisy_weight = self.weight + torch.randn_like(self.weight) * self.weight_noise_std
        else:
            noisy_weight = self.weight

        noisy_weight = torch.clamp(noisy_weight, -self.weight_clip, self.weight_clip)

        out = F.conv2d(noisy_input, noisy_weight, self.bias, self.stride, self.padding, self.dilation,
                       self.groups)

        if self.training:
            out = out + torch.randn_like(out) * self.post_noise_std

        return out

class OpticalConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, weight_noise_std=0.03, prev_noise_std=0.05, post_noise_std=0.05,weight_clip=1.0,apply_sn=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_noise_std = weight_noise_std
        self.prev_noise_std = prev_noise_std
        self.post_noise_std = post_noise_std
        self.weight_clip = weight_clip
        if apply_sn:
            P.spectral_norm(self, name='weight')
    def forward(self, input):
        # 1. 模拟权重偏差
        if self.training:
            noisy_weight = self.weight + torch.randn_like(self.weight) * self.weight_noise_std
            # 2. 模拟前端调制器噪声 (放在 training 控制下)
            noisy_input = input + torch.randn_like(input) * self.prev_noise_std
        else:
            noisy_weight = self.weight
            noisy_input = input

            # 权重截断
        noisy_weight = torch.clamp(noisy_weight, -self.weight_clip, self.weight_clip)

        # 3. 理想数字域卷积
        out = F.conv_transpose2d(
            noisy_input,
            noisy_weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
        )

        # 4. 后级读出噪声
        if self.training:
            out = out + torch.randn_like(out) * self.post_noise_std

        return out

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ContinuousTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        """
        time: 连续的噪声大小 t (例如 [0.002, 80])
        """
        device = time.device

        # 1. EDM / CM 核心预处理：取对数并缩放
        # 加 1e-8 是为了绝对的数值安全，防止意外输入 t=0 导致 log(0) = -inf
        c_noise = 0.25 * torch.log(time + 1e-8)

        # 2. 计算标准正余弦频率
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)

        # 3. 将预处理后的 c_noise 作为 "伪时间步" 与频率相乘
        embeddings = c_noise[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )

        # self.block2 = nn.Sequential(
        #     nn.GroupNorm(8, out_channels),
        #     nn.SiLU(),
        #     nn.Dropout(dropout),
        #     nn.Conv2d(out_channels, out_channels, 3, padding=1),
        # )
        #
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        # h = self.block2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        q = q.reshape(B, C, H * W).permute(0, 2, 1)
        k = k.reshape(B, C, H * W)
        v = v.reshape(B, C, H * W).permute(0, 2, 1)

        attn = torch.bmm(q, k) * (int(C) ** (-0.5))
        attn = F.softmax(attn, dim=2)

        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        h = self.proj_out(h)

        return x + h

class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, time_emb_dim=128,image_size=None):
        super().__init__()

        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim * 4),
        )
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.res1 = ResidualBlock(64, 64, time_emb_dim * 4)
        self.down1 = nn.Conv2d(64, 64, 3, stride=2, padding=1)  # 32->16

        self.res2 = ResidualBlock(64, 128, time_emb_dim * 4)
        self.down2 = nn.Conv2d(128, 128, 3, stride=2, padding=1)  # 16->8

        self.res3 = ResidualBlock(128, 256, time_emb_dim * 4)
        self.down3 = nn.Conv2d(256, 256, 3, stride=2, padding=1)  # 8->4

        # Middle
        self.mid1 = ResidualBlock(256, 512, time_emb_dim * 4)
        self.mid_attn = AttentionBlock(512)
        self.mid2 = ResidualBlock(512, 512, time_emb_dim * 4)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1)  # 4->8
        self.res_up3 = ResidualBlock(256 + 256, 256, time_emb_dim * 4)

        self.up2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)  # 8->16
        self.res_up2 = ResidualBlock(128 + 128, 128, time_emb_dim * 4)

        self.up1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)  # 16->32
        self.res_up1 = ResidualBlock(64 + 64, 64, time_emb_dim * 4)

        # Output
        self.output = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, 3, padding=1),
        )

    def forward(self, x, time):
        # Time embedding
        time_emb = self.time_embedding(time)
        time_emb = self.time_mlp(time_emb)

        # Encoder
        x1 = self.conv1(x)
        x1 = self.res1(x1, time_emb)

        x2 = self.down1(x1)
        x2 = self.res2(x2, time_emb)

        x3 = self.down2(x2)
        x3 = self.res3(x3, time_emb)

        x4 = self.down3(x3)

        # Middle
        x4 = self.mid1(x4, time_emb)
        x4 = self.mid_attn(x4)
        x4 = self.mid2(x4, time_emb)

        # Decoder
        x = self.up3(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.res_up3(x, time_emb)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.res_up2(x, time_emb)

        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.res_up1(x, time_emb)

        return self.output(x)

class FlexibleUNet(nn.Module):
    """Highly configurable UNet for diffusion models.

    Args:
        in_channels:            Input image channels (e.g. 3 for RGB).
        out_channels:           Output image channels (typically same as in_channels).
        base_channels:          Number of channels after the initial conv (default 64).
        channel_multipliers:    Multiplier per resolution level, from highest to lowest res.
                                e.g. [1, 2, 4, 8] means channels = base * [1, 2, 4, 8].
                                Must have exactly one more element than num_res_blocks (the extra
                                one is for the bottleneck / middle block).
        num_res_blocks:         How many ResidualBlocks per encoder/decoder level. Can be:
                                - int: same number for all levels.
                                - list of ints: one per level (not including middle).
        use_attention:          Whether to insert an AttentionBlock after each ResidualBlock. Can be:
                                - bool: apply to all or none.
                                - list of bools: one per level (not including middle).
        time_emb_dim:           Dimension of the sinusoidal time embedding.
        time_mlp_ratio:         Multiplier for the inner dimension of the time MLP.
        dropout:                Dropout rate inside ResidualBlocks.
        norm_groups:            Number of groups for GroupNorm.
        middle_attention:       Whether to use attention in the bottleneck (default True).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        channel_multipliers: list = None,
        num_res_blocks: int | list = 2,
        use_attention: bool | list = False,
        time_emb_dim: int = 128,
        time_mlp_ratio: int = 4,
        dropout: float = 0.1,
        norm_groups: int = 8,
        middle_attention: bool = False,
        device="cuda",
        continuous_time:bool=True,
    ):
        super().__init__()
        self.device=device
        # ── defaults ──────────────────────────────────────────────────
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8]  # 4 levels: 3 encoder/decoder + 1 bottleneck

        self.num_levels = len(channel_multipliers) - 1  # encoder/decoder levels
        time_mlp_dim = time_emb_dim * time_mlp_ratio

        # Normalise per-level configs to lists ---------------------------------
        if isinstance(num_res_blocks, int):
            num_res_blocks = [num_res_blocks] * self.num_levels
        if isinstance(use_attention, bool):
            use_attention = [use_attention] * self.num_levels

        assert len(num_res_blocks) == self.num_levels
        assert len(use_attention) == self.num_levels

        # ── time embedding ────────────────────────────────────────────
        self.time_embedding = ContinuousTimeEmbedding(time_emb_dim) \
            if continuous_time else TimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_mlp_dim),
            nn.SiLU(),
            nn.Linear(time_mlp_dim, time_mlp_dim),
        )

        # ── input projection ──────────────────────────────────────────
        self.input_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)

        # ── encoder ───────────────────────────────────────────────────
        self.encoder_blocks = nn.ModuleList()
        self.down_modules = nn.ModuleList()
        ch = base_channels

        for level in range(self.num_levels):
            in_ch = ch
            out_ch = base_channels * channel_multipliers[level]

            level_blocks = nn.ModuleList()
            for _ in range(num_res_blocks[level]):
                level_blocks.append(ResidualBlock(in_ch, out_ch, time_mlp_dim, dropout))
                if use_attention[level]:
                    level_blocks.append(AttentionBlock(out_ch))
                in_ch = out_ch

            self.encoder_blocks.append(level_blocks)
            self.down_modules.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1,groups=out_ch//2))
            ch = out_ch

        # ── bottleneck ───────────────────────────────────────────────
        bottleneck_ch = base_channels * channel_multipliers[-1]
        self.mid_block1 = ResidualBlock(ch, bottleneck_ch, time_mlp_dim, dropout)
        if middle_attention:
            self.mid_attn = AttentionBlock(bottleneck_ch)
        else:
            self.mid_attn = nn.Identity()
        self.mid_block2 = ResidualBlock(bottleneck_ch, bottleneck_ch, time_mlp_dim, dropout)

        # ── decoder ───────────────────────────────────────────────────
        self.decoder_blocks = nn.ModuleList()
        self.up_modules = nn.ModuleList()
        ch = bottleneck_ch

        for level in reversed(range(self.num_levels)):
            out_ch_dec = base_channels * channel_multipliers[level]

            self.up_modules.append(
                # nn.ConvTranspose2d(ch, out_ch_dec, 3, stride=2, padding=1, output_padding=1)
                nn.ConvTranspose2d(ch, out_ch_dec, 2, stride=2, padding=0, output_padding=0,groups=out_ch_dec//2))

            level_blocks = nn.ModuleList()
            in_ch = out_ch_dec * 2  # due to skip connection concatenation
            for _ in range(num_res_blocks[level]):
                level_blocks.append(ResidualBlock(in_ch, out_ch_dec, time_mlp_dim, dropout))
                if use_attention[level]:
                    level_blocks.append(AttentionBlock(out_ch_dec))
                in_ch = out_ch_dec

            self.decoder_blocks.append(level_blocks)
            ch = out_ch_dec

        # ── output ────────────────────────────────────────────────────
        self.output = nn.Sequential(
            nn.GroupNorm(norm_groups, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_embedding(time)
        time_emb = self.time_mlp(time_emb)

        # Input projection
        x = self.input_conv(x)

        # ── encoder ───────────────────────────────────────────────────
        skips = []
        for level in range(self.num_levels):
            for block in self.encoder_blocks[level]:
                if isinstance(block, ResidualBlock):
                    x = block(x, time_emb)
                else:
                    x = block(x)
            skips.append(x)
            x = self.down_modules[level](x)

        # ── bottleneck ───────────────────────────────────────────────
        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        # ── decoder ───────────────────────────────────────────────────
        for level in range(self.num_levels):
            x = self.up_modules[level](x)
            skip = skips[self.num_levels - 1 - level]
            # print(f"DEBUG: x shape: {x.shape}, skip shape: {skip.shape}")
            x = torch.cat([x, skip], dim=1)
            for block in self.decoder_blocks[level]:
                if isinstance(block, ResidualBlock):
                    x = block(x, time_emb)
                else:
                    x = block(x)

        return self.output(x)


class DDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda',schedule=None):
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0).to(device)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), self.alpha_cumprod[:-1]])

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alpha_cumprod_prev) / (
                    1.0 - self.alpha_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alpha_cumprod_prev) * torch.sqrt(self.alphas) / (
                    1.0 - self.alpha_cumprod)

    def add_noise(self, x_start, timesteps, noise=None):
        """Forward diffusion process - add noise to images"""
        if noise is None:
            noise = torch.randn_like(x_start)

        # Move timesteps to same device
        timesteps = timesteps.to(self.device)

        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[timesteps].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[timesteps].reshape(-1, 1, 1, 1)

        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def sample_prev_timestep(self, model_output, timestep, sample):
        """Reverse diffusion process - remove noise from images"""
        timestep = timestep.to(self.device)

        # Compute coefficients for predicted original sample (x_0) and current sample (x_t)
        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod_prev[timestep] if timestep > 0 else torch.tensor(1.0).to(
            self.device)
        beta_prod_t = 1 - alpha_prod_t

        # Compute predicted original sample from predicted noise
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

        # Compute coefficients for pred_original_sample and current sample x_t
        pred_original_sample_coeff = (alpha_prod_t_prev ** (0.5) * self.betas[timestep]) / beta_prod_t
        current_sample_coeff = self.alphas[timestep] ** (0.5) * (1 - alpha_prod_t_prev) / beta_prod_t

        # Compute predicted previous sample
        pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample

        return pred_prev_sample


class DDIMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda', schedule=None):
        self.num_timesteps = num_timesteps
        self.device = device

        # 1. 噪声调度 (和 DDPM 完全一样)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0).to(device)

        # 加噪过程的系数 (Forward process)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        mlflow.log_parameter
        # 注：DDIM 不需要 DDPM 中那些复杂的 posterior_variance，
        # 因为 DDIM 的推导是非马尔可夫的，直接计算 x_{t-1}。

    def add_noise(self, x_start, timesteps, noise=None):
        """Forward diffusion process - 与 DDPM 完全一致"""
        if noise is None:
            noise = torch.randn_like(x_start)

        timesteps = timesteps.to(self.device)
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[timesteps].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[timesteps].reshape(-1, 1, 1, 1)

        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def sample_prev_timestep(self, model_output, timestep, sample, prev_timestep=None, eta=0.0):
        """
        Reverse diffusion process - DDIM 核心采样逻辑
        :param model_output: 模型的预测输出 (预测的噪声 epsilon)
        :param timestep: 当前步数 t
        :param sample: 当前图像 x_t
        :param prev_timestep: 目标步数 t-delta (支持跳跃采样)
        :param eta: 随机性控制。eta=0 为纯确定性 DDIM，eta=1 为 DDPM
        """
        timestep = timestep.to(self.device)

        # 默认步长为 1，但你可以通过传参实现加速跳跃 (如 t-20)
        if prev_timestep is None:
            prev_timestep = timestep - 1

        # 提取当前步和上一步的 alpha_cumprod
        alpha_prod_t = self.alpha_cumprod[timestep]

        # 当跳跃到 t < 0 时（即最后一步输出），alpha_prod_t_prev 设为 1.0
        if prev_timestep >= 0:
            alpha_prod_t_prev = self.alpha_cumprod[prev_timestep]
        else:
            alpha_prod_t_prev = torch.tensor(1.0).to(self.device)

        beta_prod_t = 1.0 - alpha_prod_t
        beta_prod_t_prev = 1.0 - alpha_prod_t_prev

        # 1. 计算预测的原始样本 x_0 (predicted original sample)
        # 公式: x_0 = (x_t - sqrt(1 - alpha_t) * epsilon) / sqrt(alpha_t)
        pred_original_sample = (sample - torch.sqrt(beta_prod_t) * model_output) / torch.sqrt(alpha_prod_t)

        # 2. 计算方差 (由 eta 控制，用于决定退化回多少随机性)
        variance = (beta_prod_t_prev / beta_prod_t) * (1.0 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * torch.sqrt(variance)

        # 3. 计算指向 x_t 的方向 (direction pointing to x_t)
        # 公式: d_t = sqrt(1 - alpha_{t-1} - sigma_t^2) * epsilon
        pred_sample_direction = torch.sqrt(
            torch.clamp(beta_prod_t_prev - std_dev_t ** 2, min=0)) * model_output

        # 4. 计算前一个时刻的样本 x_{t-1}
        # 公式: x_{t-1} = sqrt(alpha_{t-1}) * x_0 + d_t + sigma_t * noise
        pred_prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

        # 如果 eta > 0，则注入高斯噪声（蒸馏时请务必保持 eta=0）
        if eta > 0:
            noise = torch.randn_like(sample)
            pred_prev_sample = pred_prev_sample + std_dev_t * noise

        return pred_prev_sample

class Generator:
    def __init__(self,init_size,latent_dim,output_size,model):
        """

        Args:
            init_size:
            latent_dim:
            output_size:
            model: 生成器网络，输入形状是(latent_dim,),输出形状是目标图像尺寸(batch_size,C,H,W)
        """
        super(Generator,self).__init__()
        self.init_size = init_size
        self.output_size = output_size
        self.model=model
        self.latent_dim = latent_dim
        self.optimizer=...
    def generate(self,pic_num):
        z=torch.randn((pic_num,self.latent_dim),device=self.model.device)
        fake=self.model.forward(z)
        return fake

class Discriminator:
    def __init__(self,input_size,model,):
        """

        Args:
            input_size:
            model: 判别器网络，输入形状是图片尺寸(batch_size,C,H,W),输出是打分结果，[batch_size,] 或者 [batch_size,0]
        """
        super(Discriminator,self).__init__()
        self.input_size = input_size
        self.model=model
        self.optimizer=...
    def forward(self,x):
        mean_score=self.model.forward(x)
        return mean_score
def compute_gradient_penalty():
    gp=...
    return gp
def evaluate_metrics():
    # 进行FID指标评估
    ...
def save_ckpt_hp():
    # 保存设置和训练权重等
    ...

def train_gan(generator,discriminator,max_epoch,dataloader):
    generator.model.train(),discriminator.model.train()
    for epoch in range(max_epoch):
        for real in dataloader:
            # train D
            discriminator.optimizer.zero_grad()
            batch_size=real.shape[0]
            true_score=discriminator.forward(real)
            fake=generator.generate(batch_size)
            fake_score=discriminator.forward(fake.detach())
            w_loss=-(true_score-fake_score)
            w_loss+=compute_gradient_penalty(fake,real)
            w_loss.backward()
            discriminator.optimizer.step()

            # train G
            generator.optimizer.zero_grad()
            fake=generator.generate(batch_size)
            fake_score=discriminator.forward(fake)
            w_loss=-fake_score
            w_loss.backward()
            generator.optimizer.step()
        # evaluate metrics
        evaluate_metrics()
        # save checkpoint and hyperparameters
        save_ckpt_hp()


import math
class Consistency:
    def __init__(self, model, min_discrete, max_discrete, sigma_data=0.5, sigma_min=0.002, sigma_max=80.0, rho=7.0,mu0=0.9,lr=1e-3,input_img_shape=(1,32,32)):
        self.model = model
        self.discrete_step = min_discrete  # 当前训练阶段的离散步数 N
        self.s0=min_discrete
        self.s1=max_discrete
        # 边界条件和 Karras 噪声调度的超参数
        self.sigma_data = sigma_data  # 数据的标准差，通常设为 0.5
        self.sigma_min = sigma_min  # 极小的噪声边界 epsilon
        self.sigma_max = sigma_max  # 最大噪声边界 T
        self.rho = rho  # Karras 调度的指数
        self.mu0=mu0 # ema 衰减速率
        self.lr = lr
        self.input_img_shape=input_img_shape
        self.config_optimizer()
    def config_optimizer(self,):
        self.optimizer=torch.optim.Adam(lr=self.lr,params=self.model.parameters())

    def update_discrete_step(self, curr_step, max_step):
        part2 = math.sqrt(curr_step / max_step * (self.s1 ** 2 - self.s0 ** 2) + self.s0 ** 2)
        # 修复：应该是限制在最大步数 s1 内，并且推荐用 ceil (向上取整)
        new_step = min(self.s1, math.ceil(part2))
        self.discrete_step = new_step
    @property
    def mu(self):
        part=self.s0*math.log(self.mu0)/self.discrete_step
        return math.exp(part)

    def _get_sigmas(self, step_indices):
        """将离散的索引映射为连续的噪声大小 (连续的 t)"""
        # 使用 Karras 公式: t_i = (sigma_min^(1/rho) + i/(N-1) * (sigma_max^(1/rho) - sigma_min^(1/rho)))^rho
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)

        # 将步数归一化到 [0, 1] 之间
        fraction = step_indices / (self.discrete_step - 1)
        sigmas = (min_inv_rho + fraction * (max_inv_rho - min_inv_rho)) ** self.rho
        return sigmas

    def get_neighbor_pair(self, x):
        batch_size = x.shape[0]

        # Q: t对于同一batch内的多个样本，是一样的吗？还是每个样本独立？
        # A: 必须是独立的！为了最大化训练效率，batch 中的每张图应该学习不同时间步的一致性。
        # Q: t是int，还是什么取值范围？
        # A: 这里的 step_indices 是 int，取值范围是 [0, self.total_step - 2]。
        step_indices = torch.randint(0, self.discrete_step - 1, (batch_size,), device=x.device)

        # 将离散索引映射为相邻的连续噪声大小 t_n 和 t_n+1 (这里我们用 t_n 和 t_n_next 表示)
        t_n = self._get_sigmas(step_indices)
        t_n_next = self._get_sigmas(step_indices + 1)

        # 为了能和图像 x 相乘，将 t 调整为 (batch_size, 1, 1, 1) 的形状
        t_n_view = t_n.view(-1, 1, 1, 1)
        t_n_next_view = t_n_next.view(-1, 1, 1, 1)

        # Q: noise 呢？
        # A: noise 是标准高斯噪声。每个样本独立采样一个 noise，但为了保证处于同一条 ODE 轨迹上，
        #    同一个样本的 x_t 和 x_t_next 必须加上 **完全相同的那个 noise**！
        noise = torch.randn_like(x)

        # 生成相邻时间步的加噪图像
        x_t = x + t_n_view * noise
        x_t_next = x + t_n_next_view * noise

        # 返回离散的 step_indices 用于某些可能的条件注入，以及连续的 t_n, t_n_next
        return x_t, x_t_next, t_n, t_n_next

    def get_c_skip(self, t):
        """计算跳跃连接的权重系数 c_skip"""
        return (self.sigma_data ** 2) / ((t - self.sigma_min) ** 2 + self.sigma_data ** 2)

    def get_c_out(self, t):
        """计算网络输出的权重系数 c_out"""
        return self.sigma_data * (t - self.sigma_min) / torch.sqrt(
            (t - self.sigma_min) ** 2 + self.sigma_data ** 2)

    def wrap_out(self, x_t, t):
        """
        满足边界条件的模型输出包装函数
        x_t: 加噪图像
        t: 连续的噪声大小 (1D tensor, shape: [batch_size])
        """
        # 将 t 调整为 (batch_size, 1, 1, 1)
        t_view = t.view(-1, 1, 1, 1)

        c_skip = self.get_c_skip(t_view)
        c_out = self.get_c_out(t_view)

        # 这里的 self.model(x_t, t) 需要你的模型支持连续的噪声 t 作为条件输入
        pred = c_skip * x_t + c_out * self.model(x_t, t)
        return pred

    def generate(self, n_sample):
        T = self._get_sigmas(torch.full((n_sample,), self.discrete_step - 1, device=self.model.device))
        z = torch.randn(n_sample, *self.input_img_shape, device=self.model.device)

        # 🚨 必须将初始噪声缩放到对应的 T 量级！
        T_view = T.view(-1, 1, 1, 1)
        z = z * T_view

        return self.wrap_out(z, T)

def update_ema(stu_model, teach_model, mu):
    with torch.no_grad():
        for param_stu, param_teach in zip(stu_model.parameters(), teach_model.parameters()):
            param_teach.copy_(mu * param_teach + (1.0 - mu) * param_stu)

def pseudo_huber_loss(input, target, c=0.00054):
    """
    计算 Pseudo-Huber 损失。
    c 值可以根据你数据的归一化范围进行微调。
    如果你的数据归一化在 [-1, 1]，0.00054 到 0.03 之间都是可以尝试的值。
    """
    return (torch.sqrt((input - target)**2 + c**2) - c).mean()
import lpips
lpips_loss_fn = lpips.LPIPS(net='vgg').eval().cuda().requires_grad_(False)

def make_loss_f(mse_w, huber_w, lpips_w, lpips_loss_fn=lpips_loss_fn, c=0.01, lambda_w=1.0, ):

    def compute_consistency_loss(y_stu, y_teach,):
        total_loss = 0.0
        mse_val, huber_val, lpips_val = 0.0, 0.0, 0.0

        # 1. 计算 MSE Loss
        if mse_w > 0.0:
            mse_loss = F.mse_loss(y_stu, y_teach)
            total_loss += mse_w * mse_loss
            mse_val = mse_loss.item()

        # 2. 计算 Pseudo-Huber Loss
        if huber_w > 0.0 and c is not None:
            huber_loss = pseudo_huber_loss(y_stu, y_teach, c)
            total_loss += huber_w * huber_loss
            huber_val = huber_loss.item()

        # 3. 计算 LPIPS Loss
        if lpips_w > 0.0 and lpips_loss_fn is not None:
            # lpips_loss_fn 从外部闭包捕获
            lpips_loss = lpips_loss_fn(y_stu, y_teach).mean()
            total_loss += lpips_w * lpips_loss
            lpips_val = lpips_loss.item()

        total_loss = lambda_w * total_loss
        return {"total_loss":total_loss, "mse":mse_val, "huber":huber_val, "lpips":lpips_val}

    return compute_consistency_loss
def evaluate_metrics_cm(cm, dataloader,  num_samples=5000, save_dir="samples",device=None):
    """
    评估 Consistency Model 的 FID 并保存采样图像。
    """

    print("Starting evaluation...")
    cm.model.eval()

    # 初始化 FID 计算器 (特征维度 2048 是标准 FID)
    fid = FrechetInceptionDistance(feature=2048).to(device)
    os.makedirs(save_dir, exist_ok=True)

    samples_generated = 0
    saved_images = False

    with torch.no_grad():
        # 1. 收集真实图像特征
        for real_batch in dataloader:
            # 假设 batch 是图像张量，且被归一化到 [-1, 1]
            real_images = real_batch[0].to(device)

            # torchmetrics 的 FID 期望输入范围是 [0, 255] 的 uint8
            # 或者归一化前先转回 [0, 1] 范围内的 float
            real_images_uint8 = ((real_images + 1.0) / 2.0 * 255).byte()
            # 如果是单通道，就复制成 3 通道
            if real_images_uint8.shape[1] == 1:
                real_images_uint8 = real_images_uint8.repeat(1, 3, 1, 1)


            fid.update(real_images_uint8, real=True)

            # 2. 生成对应数量的假图像
            batch_size = real_images.shape[0]

            # 使用模型进行一步生成
            fake_images = cm.generate(batch_size)
            fake_images = torch.clamp(fake_images, -1.0, 1.0)  # 截断到合理范围

            # 保存第一批生成的图片供肉眼观察
            if not saved_images:
                torchvision.utils.save_image(
                    fake_images,
                    os.path.join(save_dir, "eval_sample.png"),
                    nrow=8,
                    normalize=True,
                    value_range=(-1, 1)
                )
                saved_images = True

            fake_images_uint8 = ((fake_images + 1.0) / 2.0 * 255).byte()
            # 别忘了生成的假图像也需要做同样的转换
            if fake_images_uint8 .shape[1] == 1:
                fake_images_uint8  = fake_images_uint8.repeat(1, 3, 1, 1)
            fid.update(fake_images_uint8, real=False)

            samples_generated += batch_size
            if samples_generated >= num_samples:
                break

    # 3. 计算最终 FID
    fid_score = fid.compute().item()
    print(f"Evaluation finished! FID Score: {fid_score:.4f}")

    cm.model.train()  # 评估结束后恢复训练模式

    return {"fid": fid_score}

def train_cm(log_manager,max_epoch, dataloader, cm_teach, cm_stu,eval_num,device,loss_fn):
    cm_teach.model.eval(),cm_stu.model.train()
    curr_step,max_step=0,max_epoch * len(dataloader)
    for epoch in range(max_epoch):
        for batch in dataloader:
            img,label=batch
            img=img.to(device)
            cm_teach.model.zero_grad(),cm_stu.model.zero_grad()
            x_t,x_t_next,t_n,t_n_next=cm_teach.get_neighbor_pair(img)
            y_stu=cm_stu.wrap_out(x_t_next,t_n_next)
            with torch.no_grad():
                y_teach=cm_teach.wrap_out(x_t, t_n)
            loss_dict=loss_fn(y_stu,y_teach)
            loss_dict["total_loss"].backward()

            # 可选：梯度裁剪，防止训练初期 Loss 爆炸
            torch.nn.utils.clip_grad_norm_(cm_stu.model.parameters(), max_norm=1.0)
            cm_stu.optimizer.step()

            update_ema(cm_stu.model,cm_teach.model,cm_stu.mu)# 没写过这个代码
            cm_stu.update_discrete_step(curr_step,max_step)
            # print logs

            if curr_step % 100 == 0:
                # 使用列表推导式将键值对格式化为 "Key: Value" 的字符串列表
                loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in loss_dict.items()])
                print(f"Step {curr_step}/{max_step} | {loss_str}")

            curr_step += 1
        # evaluate
        # save ckpt hyperparameters
        if epoch%3==0:
            metrics=evaluate_metrics_cm(cm_teach,dataloader,num_samples=eval_num,device=device)
            log_manager.save_ckpt(cm_stu,cm_teach,cm_stu.optimizer,step=curr_step,metrics=metrics,)



class NoiseLayer(nn.Module):
    def __init__(self, noise_std):
        super(NoiseLayer, self).__init__()
        self.noise_std = noise_std
        
    def forward(self, x):
        if self.training and self.noise_std > 0:
            # 1. 判断是否为复数张量
            if x.is_complex():
                # 分别生成实部和虚部的独立噪声
                # 注意：为了保持总方差为 noise_std^2，实部和虚部的标准差需要除以 sqrt(2)
                std_complex = self.noise_std / (2 ** 0.5)
                
                noise_real = torch.randn_like(x.real) * std_complex
                noise_imag = torch.randn_like(x.imag) * std_complex
                
                # 组合成复数噪声并相加
                # torch.complex 在新版 PyTorch 中推荐使用，或者直接 x.real + noise_real ...
                return torch.complex(x.real + noise_real, x.imag + noise_imag)
            
            else:
                # 2. 实数张量处理逻辑（保持原样）
                noise = torch.randn_like(x) * self.noise_std
                return x + noise
                
        return x
class cTransposeConv(nn.Module):
    """
    Complex transpose convolution (learnable complex weights).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 2,
        stride: int = 2,
        padding: int = 0,
        bias: bool = False,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()
        assert in_channels % groups == 0
        assert out_channels % groups == 0

        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(
            torch.randn((in_channels, out_channels // groups, *kernel_size), dtype=torch.cfloat)
        )
        self.bias = nn.Parameter(torch.randn((out_channels,), dtype=torch.cfloat)) if bias else None

    def forward(self, x):
        if x.dtype != torch.cfloat:
            x = torch.complex(x, torch.zeros_like(x))
        return F.conv_transpose2d(
            x, self.weight, self.bias, self.stride, self.padding, 0, self.dilation, self.groups
        )
