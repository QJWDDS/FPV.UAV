import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import matplotlib.pyplot as plt

from dataset import DroneDataset
from model import E2EPilotNet

# --- 参数设置 ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 30
DATA_DIR = os.path.expanduser('~/sh_ws/document/baylands_data/e2evirtual') #
TRAIN_VERSION = "v1"

if not os.path.exists("models"): os.makedirs("models")

MODEL_SAVE_PATH = f'models/baylands_e2e_model_{TRAIN_VERSION}.pth'
LOSS_CURVE_PATH= f'models/baylands_loss_curve_{TRAIN_VERSION}'

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        # ImageNet 标准均值方差
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = DroneDataset(DATA_DIR, transform=data_transform)
    
    train_size = int(0.8 * len(full_dataset))  # 划分训练集
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Total samples: {len(full_dataset)} | Train: {train_size} | Val: {val_size}")

    model = E2EPilotNet().to(device)
    
    criterion = nn.MSELoss() # 均方误差
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    #
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

        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("  -> Saved Best Model")

    print("Training Complete!")
    
    # 绘制 Loss 曲线
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(LOSS_CURVE_PATH)
    plt.show()

if __name__ == '__main__':
    train()