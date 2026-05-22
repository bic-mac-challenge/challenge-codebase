import torch
import torch.nn as nn


# -----------------------------
# Residual Conv Block
# -----------------------------

class ResidualBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_ch)
        self.relu = nn.LeakyReLU(0.01, inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(out_ch)

        self.skip = None

        if in_ch != out_ch:
            self.skip = nn.Conv3d(in_ch, out_ch, 1)

    def forward(self, x):

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.skip is not None:
            identity = self.skip(identity)

        out += identity
        out = self.relu(out)

        return out


# -----------------------------
# Encoder Block
# -----------------------------

class EncoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.block = ResidualBlock(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):

        x = self.block(x)
        p = self.pool(x)

        return x, p


# -----------------------------
# Decoder Block
# -----------------------------

class DecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):

        super().__init__()

        self.up = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)

        self.block = ResidualBlock(in_ch, out_ch)

    def forward(self, x, skip):

        x = self.up(x)

        x = torch.cat([x, skip], dim=1)

        x = self.block(x)

        return x


# -----------------------------
# UNet
# -----------------------------

class UNet3D(nn.Module):

    def __init__(self, in_channels=4, out_channels=1):

        super().__init__()

        # Encoder
        self.enc1 = EncoderBlock(in_channels, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock(256, 512),
            nn.Dropout3d(0.2)
        )

        # Decoder
        self.dec4 = DecoderBlock(512, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 32)

        # Output
        self.out_conv = nn.Conv3d(32, out_channels, 1)

    def forward(self, x):

        s1, p1 = self.enc1(x)
        s2, p2 = self.enc2(p1)
        s3, p3 = self.enc3(p2)
        s4, p4 = self.enc4(p3)

        b = self.bottleneck(p4)

        d4 = self.dec4(b, s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)

        out = self.out_conv(d1)

        return out


# -----------------------------
# Builder
# -----------------------------

def build_model():

    return UNet3D(
        in_channels=1,  # only NAC-PET as input
        out_channels=1
    )