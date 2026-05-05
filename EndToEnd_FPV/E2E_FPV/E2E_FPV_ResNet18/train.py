import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import json
import matplotlib.pyplot as plt

from dataset import DroneDataset
from model import ResNetPilotNet

# --- 参数设置 ---
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 30
DATA_DIR = os.path.expanduser('~/sh_ws/document/baylands_data/e2evirtual')
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_VISION = "v1"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, f'resnet_e2e_model_{TRAIN_VISION}.pth')
STATS_SAVE_PATH = os.path.join(MODEL_DIR, f'label_stats_{TRAIN_VISION}.json')

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 初始化
    full_dataset = DroneDataset(root_dir=DATA_DIR, transform=data_transform)
    
    with open(STATS_SAVE_PATH, 'w') as f:
        json.dump(full_dataset.get_label_stats(), f)
    print(f"Saved label normalization stats to {STATS_SAVE_PATH}")

    # 划分训练/验证集
    val_size = int(0.1 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = ResNetPilotNet(output_dim=4, use_pretrained=True).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss(beta=1.0) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    print("Starting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
        
        avg_val_loss = val_running_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("  -> Saved Best Model")

    print("Training Complete!")
    
    # 画图并保存
    plt.figure()
    plt.plot(train_losses, label='Train Loss (Smooth L1)')
    plt.plot(val_losses, label='Val Loss (Smooth L1)')
    plt.legend()
    plt.title('Training Curve')
    plt.savefig(os.path.join(MODEL_DIR, f'resnet_loss_curve_{TRAIN_VISION}.png'))

if __name__ == '__main__':
    train()