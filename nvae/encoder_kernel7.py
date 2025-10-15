import torch
import torch.nn as nn

from nvae.common_kernel7 import EncoderResidualBlock, Swish


class ConvBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()

        self._seq = nn.Sequential(

            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=1),
            nn.BatchNorm2d(out_channel // 2), Swish(),
            nn.Conv2d(out_channel // 2, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel), Swish()
        )

    def forward(self, x):
        return self._seq(x)


class EncoderBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        modules = []
        for i in range(len(channels) - 1):
            modules.append(ConvBlock(channels[i], channels[i + 1]))

        self.modules_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.modules_list:
            x = module(x)
        return x


class Encoder(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        # 각 EncoderBlock이 하나의 group
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock([3, z_dim // 16, z_dim // 8]),  # (16, 16)
            EncoderBlock([z_dim // 8, z_dim // 4, z_dim // 2]),  # (4, 4)
            EncoderBlock([z_dim // 2, z_dim]),  # (2, 2)
        ])

        # Residual Block (논문에서의 r, 각 계층마다 feature 반환)
        self.encoder_residual_blocks = nn.ModuleList([
            EncoderResidualBlock(z_dim // 8),
            EncoderResidualBlock(z_dim // 2),
            EncoderResidualBlock(z_dim),
        ])

        # h (trainable parameter)
        self.condition_x = nn.Sequential(
            Swish(),
            nn.Conv2d(z_dim, z_dim * 2, kernel_size=1)
        )

    def forward(self, x):
        xs = [] # 예를 들어, 3계층이면 xs = [x1, x2, x3]
        # x3는 마지막 계층 (가장 깊은 계층, latent space에 가까움)
        
        last_x = x
        for e, r in zip(self.encoder_blocks, self.encoder_residual_blocks):
            x = r(e(x))
            last_x = x
            xs.append(x)

        mu, log_var = self.condition_x(last_x).chunk(2, dim=1)

        return mu, log_var, xs[:-1][::-1] 
    
        # 맨 마지막 계층의 피처 : latent 평균, 분산 계산에만 사용
        # 나머지 계층의 피처 : 디코더에 ㅈㄴ달
    
