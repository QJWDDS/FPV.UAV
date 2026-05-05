import torch
import torch.nn as nn
import torchvision.models as models

class ResNetPilotNet(nn.Module):
    def __init__(self, output_dim=4, use_pretrained=True):
        super(ResNetPilotNet, self).__init__()
        
        # ResNet18
        weights = models.ResNet18_Weights.DEFAULT if use_pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # 替换全连接层
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.backbone(x)

if __name__ == '__main__':
    model = ResNetPilotNet()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}, Output shape: {output.shape}")