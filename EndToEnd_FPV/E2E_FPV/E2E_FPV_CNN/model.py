import torch
import torch.nn as nn

class E2EPilotNet(nn.Module):
    def __init__(self, output_dim=4):
        super(E2EPilotNet, self).__init__()
        
        # Backbone
        self.features = nn.Sequential(
            # Input: 3 x 128 x 128
            nn.Conv2d(3, 24, kernel_size=5, stride=2),  # -> 24 x 62 x 62
            nn.BatchNorm2d(24),
            nn.ReLU(),
            
            nn.Conv2d(24, 36, kernel_size=5, stride=2), # -> 36 x 29 x 29
            nn.BatchNorm2d(36),
            nn.ReLU(),
            
            nn.Conv2d(36, 48, kernel_size=5, stride=2), # -> 48 x 13 x 13
            nn.BatchNorm2d(48),
            nn.ReLU(),
            
            nn.Conv2d(48, 64, kernel_size=3, stride=1), # -> 64 x 11 x 11
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1), # -> 64 x 9 x 9
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # 64 * 9 * 9 = 5184
            nn.Linear(5184, 100),
            nn.ReLU(),
            nn.Dropout(0.3), # 防止过拟合
            
            nn.Linear(100, 50),
            nn.ReLU(),
            
            nn.Linear(50, output_dim) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x