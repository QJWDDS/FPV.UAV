import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import os

from model import E2EPilotNet

# --- 配置路径 ---
TRAIN_VERSION = "v1"
CHECKPOINT_PATH = f"models/baylands_e2e_model_{TRAIN_VERSION}.pth" 
ROOT_DIR = Path(os.path.expanduser('~/sh_ws/document/baylands_data/e2evirtual'))

if not os.path.exists("evaluate"): os.makedirs("evaluate")
EVALUATION_PATH = f'evaluate/cnn_random_evaluation_results_baylands_{TRAIN_VERSION}.png'

NUM_SAMPLES = 1000 # 随机抽样数量

def evaluate_cnn_random_samples():
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

    print(f"Scanning for data in {ROOT_DIR}...")
    all_data = []
    folders = [f for f in sorted(ROOT_DIR.iterdir()) if f.is_dir()]
    
    for folder in folders:
        csv_path = folder / "data.csv"
        img_dir = folder / "images"
        if not csv_path.exists():
            continue
            
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_path = img_dir / row['img_name']
            action = [row['v_body_x'], row['v_body_y'], row['v_body_z'], row['yaw_rate_cmd']]
            all_data.append({
                'img_path': img_path,
                'action': action
            })

    total_frames = len(all_data)
    print(f"Found {total_frames} valid frames in total.")
    
    # 随机抽样
    sample_size = min(NUM_SAMPLES, total_frames)
    sampled_data = random.sample(all_data, sample_size)
    print(f"Randomly selected {sample_size} frames for evaluation.")

    all_preds = []
    all_reals = []
    
    with torch.no_grad():
        for item in tqdm(sampled_data, desc="Evaluating Random Samples (CNN)"):
            img_path = item['img_path']
            real_cmd = np.array(item['action'], dtype=np.float32)
            
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue
                
            img_tensor = transform(image).unsqueeze(0).to(device) 
            
            predicted_action = model(img_tensor)
            pred_cmd = predicted_action.cpu().numpy()[0]
            
            all_preds.append(pred_cmd)
            all_reals.append(real_cmd)

    preds = np.array(all_preds)  # Shape: (100, 4)
    reals = np.array(all_reals)
    abs_errors = np.abs(preds - reals)
    
    mean_abs_errors = np.mean(abs_errors, axis=0)
    max_abs_errors = np.max(abs_errors, axis=0)
    
    action_names = ['vx', 'vy', 'vz', 'yaw_rate']
    
    print("\n" + "="*40)
    print(f"【{sample_size}帧随机抽样评估结果 (CNN)】 (Mean Absolute Error)")
    print("="*40)
    for i, name in enumerate(action_names):
        print(f"{name:<10}: 平均误差 = {mean_abs_errors[i]:.4f} | 最大误差 = {max_abs_errors[i]:.4f}")
    print("="*40)

    # 绘制图表
    print("\nGenerating evaluation plots...")
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'E2EPilotNet (CNN) - Random {sample_size} Samples Evaluation', fontsize=16, fontweight='bold')

    for i in range(4):
        ax = plt.subplot(4, 2, 2*i + 1)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.scatter(range(len(abs_errors)), abs_errors[:, i], color='tab:red', alpha=0.7, s=15)
        ax.set_title(f'Absolute Error Distribution: {action_names[i]}', fontsize=12)
        ax.set_ylabel('Absolute Error')
        ax.grid(True, linestyle=':', alpha=0.6)
        if i == 3:
            ax.set_xlabel('Sample Index')

    ax_bar = plt.subplot(1, 2, 2)
    bars = ax_bar.bar(action_names, mean_abs_errors, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:purple'], alpha=0.8)
    ax_bar.set_title('Mean Absolute Error (MAE) per Dimension', fontsize=14)
    ax_bar.set_ylabel('Mean Error Value')
    ax_bar.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax_bar.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), 
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    plot_filename = EVALUATION_PATH
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved successfully to {plot_filename}")

if __name__ == "__main__":
    evaluate_cnn_random_samples()