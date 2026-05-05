import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
import os

from model import E2EPilotNet

# --- 配置路径 ---
TRAIN_VERSION = "v1"
CHECKPOINT_PATH = f"models/baylands_e2e_model_{TRAIN_VERSION}.pth" 
TEST_FOLDER = Path(os.path.expanduser('~/sh_ws/document/baylands_data/e2evirtual/20260115_170649'))


CSV_PATH = TEST_FOLDER / 'data.csv'
IMG_DIR = TEST_FOLDER / 'images'

# 指定测试图片
TEST_IMG_NAME = "009846.jpg" 

def calculate_safe_relative_error(pred, real, epsilon=1e-3):
    rel_errors = []
    for p, r in zip(pred, real):
        if abs(r) < epsilon:
            rel_errors.append(float('nan'))
        else:
            rel_errors.append(abs(p - r) / abs(r))
    return np.array(rel_errors)

def test_cnn_single_frame():
    print(f"Loading CNN model from {CHECKPOINT_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = E2EPilotNet(output_dim=4)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if not CSV_PATH.exists():
        print(f"Error: CSV not found at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    target_rows = df[df['img_name'] == TEST_IMG_NAME]
    if len(target_rows) == 0:
        print(f"Error: Image {TEST_IMG_NAME} not found in {CSV_PATH}")
        return
        
    test_row = target_rows.iloc[0]
    img_path = IMG_DIR / TEST_IMG_NAME
    
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return
        
    # (1, 3, 128, 128)
    img_tensor = transform(image).unsqueeze(0).to(device) 
    
    print(f"\nRunning CNN inference for {TEST_IMG_NAME}...")
    with torch.no_grad():
        predicted_action = model(img_tensor)
    
    pred_cmd = predicted_action.cpu().numpy()[0] 
    
    real_cmd = np.array([
        test_row['v_body_x'], test_row['v_body_y'], test_row['v_body_z'], test_row['yaw_rate_cmd']
    ], dtype=np.float32)
    
    # 计算误差
    abs_error = np.abs(pred_cmd - real_cmd)
    rel_error = calculate_safe_relative_error(pred_cmd, real_cmd)
    
    print("-" * 65)
    print(f"{'Dimension':<12} | {'vx':<10} | {'vy':<10} | {'vz':<10} | {'yaw_rate':<10}")
    print("-" * 65)
    print(f"{'Predicted':<12} | {pred_cmd[0]:10.4f} | {pred_cmd[1]:10.4f} | {pred_cmd[2]:10.4f} | {pred_cmd[3]:10.4f}")
    print(f"{'Ground Truth':<12} | {real_cmd[0]:10.4f} | {real_cmd[1]:10.4f} | {real_cmd[2]:10.4f} | {real_cmd[3]:10.4f}")
    print("-" * 65)
    print(f"{'Abs Error':<12} | {abs_error[0]:10.4f} | {abs_error[1]:10.4f} | {abs_error[2]:10.4f} | {abs_error[3]:10.4f}")
    
    rel_str = []
    for re in rel_error:
        if np.isnan(re):
            rel_str.append(f"{'N/A':>10}")
        else:
            rel_str.append(f"{re*100:9.2f}%")
            
    print(f"{'Rel Error(%)':<12} | {rel_str[0]} | {rel_str[1]} | {rel_str[2]} | {rel_str[3]}")
    print("-" * 65)

if __name__ == "__main__":
    test_cnn_single_frame()