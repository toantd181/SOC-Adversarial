import torch
import torch.nn as nn
from typing import Tuple

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool: bool = False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLu(inplace = True)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2) if pool else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(self.conv(x)))
        if self.pool:
            x = self.pool(x)
        return x
    
class TrafficSignNet(nn.Module):
    def __init__(self, num_classes: int = 43):
        super(TrafficSignNet, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32, pool = True),
            ConvBlock(32, 64, pool = True),
            ConvBlock(64, 128, pool = True),
            ConvBlock(128, 256, pool = True)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)

        return x
    
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Start testing model on device: {device}")
    
    model = TrafficSignNet(num_classes=43).to(device)
    
    dummy_input = torch.randn(4, 3, 64, 64).to(device)
    
    logits = model(dummy_input)
    
    print(f"[SUCCESS] Model structure works well!")
    print(f"Input size: {dummy_input.shape}")
    print(f"Output size: {logits.shape}")