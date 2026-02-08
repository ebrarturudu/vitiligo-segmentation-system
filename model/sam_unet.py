import torch
import torch.nn as nn
from transformers import SamModel

class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        return self.block(x)

class SAM_UNet(nn.Module):
    def __init__(self, num_classes=1):
        super(SAM_UNet, self).__init__()
        self.sam = SamModel.from_pretrained("facebook/sam-vit-base")

        # SAM'i dondur
        for param in self.sam.parameters():
            param.requires_grad = False

        self.encoder_channels = 256  # SAM output channels: [B, 256, 64, 64]

        self.decoder = nn.Sequential(
            UNetDecoderBlock(self.encoder_channels, 128),  # 64 → 128
            UNetDecoderBlock(128, 64),                     # 128 → 256
            UNetDecoderBlock(64, 32),                      # 256 → 512
            UNetDecoderBlock(32, 16),                      # 512 → 1024
            nn.Conv2d(16, num_classes, kernel_size=1)      # Final output
        )

        '''def forward(self, x):
        with torch.no_grad():
            sam_features = self.sam.get_image_embeddings(x)  # [B, 256, 64, 64]

        out = self.decoder(sam_features)  # [B, num_classes, 1024, 1024]
        return out'''
    def forward(self, x):
        with torch.no_grad():
            sam_features = self.sam.get_image_embeddings(x)
        out = self.decoder(sam_features)
        return out