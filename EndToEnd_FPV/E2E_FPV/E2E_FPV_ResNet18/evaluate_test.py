import torch
import pandas as pd
import numpy as np
import json
from PIL import Image
from torchvision import transforms
from pathlib import Path
import os

from model import ResNetPilotNet

# --- 配置路径 ---
TRAIN_VISION = "v1"
CHECKPOINT_PATH = f"models/resnet_e2e_model_{TRAIN_VISION}.pth" 
STATS_PATH = f"models/label_stats_{TRAIN_VISION}.json"
TEST_FOLDER = Path(os.path.expanduser('~/sh_ws/document/baylands_data/e2evirtual/20260115_170649'))

CSV_PATH = TEST_FOLDER / 'data.csv'
IMG_DIR = TEST_FOLDER / 'images'
TEST_IMG_NAME = "002230.jpg" ### Single frame ###

def test_resnet_single_frame():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(STATS_PATH, 'r') as f:
        stats = json.load(f)
    label_mean = np.array(stats['mean'], dtype=np.float32)
    label_std = np.array(stats['std'], dtype=np.float32)

    model = ResNetPilotNet(output_dim=4, use_pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    df = pd.read_csv(CSV_PATH)
    test_row = df[df['img_name'] == TEST_IMG_NAME]
    if test_row.empty:
        print(f"Error: {TEST_IMG_NAME} not found in {CSV_PATH}")
        return
        
    test_row = test_row.iloc[0]
    img_path = IMG_DIR / TEST_IMG_NAME
    
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0).to(device) 
    
    with torch.no_grad():
        pred_norm = model(img_tensor).cpu().numpy()[0]
    
    pred_cmd = (pred_norm * label_std) + label_mean
    
    real_cmd = np.array([
        test_row['v_body_x'], test_row['v_body_y'], test_row['v_body_z'], test_row['yaw_rate_cmd']
    ], dtype=np.float32)
    
    print("-" * 75)
    print(f"{'Dimension':<12} | {'vx':<12} | {'vy':<12} | {'vz':<12} | {'yaw_rate':<12}")
    print("-" * 75)
    print(f"{'Predicted':<12} | {pred_cmd[0]:12.4f} | {pred_cmd[1]:12.4f} | {pred_cmd[2]:12.4f} | {pred_cmd[3]:12.4f}")
    print(f"{'Real CMD':<12} | {real_cmd[0]:12.4f} | {real_cmd[1]:12.4f} | {real_cmd[2]:12.4f} | {real_cmd[3]:12.4f}")
    
    abs_error = np.abs(pred_cmd - real_cmd)
    print(f"{'Abs Error':<12} | {abs_error[0]:12.4f} | {abs_error[1]:12.4f} | {abs_error[2]:12.4f} | {abs_error[3]:12.4f}")
    print("-" * 75)

if __name__ == '__main__':
    test_resnet_single_frame()