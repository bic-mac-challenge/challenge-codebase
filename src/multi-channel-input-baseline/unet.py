"""
unet.py  –  3-D residual U-Net for pseudo-CT prediction.

The number of input channels is set at build time via build_model(in_channels=…)
so that the architecture automatically adapts when additional modalities are added
in config.yaml without any code changes.

Future model variants (e.g. transformer-based, multi-scale, attention U-Net)
should expose the same build_model(in_channels, out_channels) signature so that
train.py and predict.py remain unchanged.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.act   = nn.LeakyReLU(0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch)
        self.skip  = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x) if self.skip is not None else x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class EncoderBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch)
        self.pool  = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor):
        skip = self.block(x)
        return skip, self.pool(skip)


class DecoderBlock(nn.Module):

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up    = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
        self.block = ResidualBlock(in_ch, out_ch)   # in_ch = out_ch (up) + out_ch (skip)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return self.block(torch.cat([x, skip], dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# U-Net
# ─────────────────────────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    """
    3-D residual U-Net with four encoder / decoder levels.

    Parameters
    ----------
    in_channels:
        Number of input channels.  Pass the value returned by
        transforms.get_in_channels(modalities) so this stays in sync with the
        modality config automatically.
    out_channels:
        Number of output channels (1 for pseudo-CT regression).
    base_features:
        Width of the first encoder level.  All subsequent levels double this.
        Reducing this (e.g. to 16) lowers VRAM at the cost of capacity.
    """

    def __init__(
        self,
        in_channels:   int = 1,
        out_channels:  int = 1,
        base_features: int = 32,
    ):
        super().__init__()
        f = base_features
        # Encoder
        self.enc1 = EncoderBlock(in_channels, f)
        self.enc2 = EncoderBlock(f,      f * 2)
        self.enc3 = EncoderBlock(f * 2,  f * 4)
        self.enc4 = EncoderBlock(f * 4,  f * 8)
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(f * 8, f * 16),
            nn.Dropout3d(0.2),
        )
        # Decoder
        self.dec4 = DecoderBlock(f * 16, f * 8)
        self.dec3 = DecoderBlock(f * 8,  f * 4)
        self.dec2 = DecoderBlock(f * 4,  f * 2)
        self.dec1 = DecoderBlock(f * 2,  f)
        # Output
        self.out_conv = nn.Conv3d(f, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b  = self.bottleneck(p4)

        d4 = self.dec4(b,  s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        return self.out_conv(d1)


# ─────────────────────────────────────────────────────────────────────────────
# Builder  –  single entry point used by train.py and predict.py
# ─────────────────────────────────────────────────────────────────────────────

def build_model(in_channels: int = 1, out_channels: int = 1, base_features: int = 32) -> UNet3D:
    """
    Instantiate the model.

    in_channels should be derived from the active modality list:

        from transforms import get_in_channels
        model = build_model(in_channels=get_in_channels(cfg["input_modalities"]))

    To swap in a different architecture in the future, replace UNet3D here
    while keeping the same function signature.
    """
    return UNet3D(in_channels=in_channels, out_channels=out_channels, base_features=base_features)
