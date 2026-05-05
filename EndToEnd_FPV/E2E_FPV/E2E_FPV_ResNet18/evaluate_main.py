import torch
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import random

from model import ResNetPilotNet

# --- 配置路径 ---
TRAIN_VISION = "v1"
CHECKPOINT_PATH = f"models/resnet_e2e_model_{TRAIN_VISION}.pth" 
STATS_PATH = f"models/label_stats_{TRAIN_VISION}.json"
ROOT_DIR = Path(os.path.expanduser('~/sh_ws/document/baylands_data/e2evirtual'))
if not os.path.exists("evaluate"): os.makedirs("evaluate")
EVALUATION_PATH = f'evaluate/resnet_random_evaluation_results_{TRAIN_VISION}.png'
NUM_SAMPLES = 1000 ###

def evaluate_resnet_random_samples():
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

    all_data_paths = []
    for folder in sorted(os.listdir(ROOT_DIR)):
        folder_path = ROOT_DIR / folder
        csv_path = folder_path / 'data.csv'
        if folder_path.is_dir() and csv_path.exists():
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                img_path = folder_path / 'images' / row['img_name']
                all_data_paths.append({
                    'img_path': img_path,
                    'labels': [row['v_body_x'], row['v_body_y'], row['v_body_z'], row['yaw_rate_cmd']]
                })

    sampled_data = random.sample(all_data_paths, min(NUM_SAMPLES, len(all_data_paths)))
    
    abs_errors = []
    from PIL import Image

    print(f"Evaluating {len(sampled_data)} random samples...")
    for data in tqdm(sampled_data):
        img_path = data['img_path']
        real_cmd = np.array(data['labels'], dtype=np.float32)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            continue
            
        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_norm = model(img_tensor).cpu().numpy()[0]
        
        pred_cmd = (pred_norm * label_std) + label_mean
        abs_errors.append(np.abs(pred_cmd - real_cmd))

    abs_errors = np.array(abs_errors)
    mean_abs_errors = np.mean(abs_errors, axis=0)
    action_names = ['v_body_x', 'v_body_y', 'v_body_z', 'yaw_rate']

    # --- 绘图 ---
    plt.figure(figsize=(16, 10))
    plt.suptitle(f'ResNet PilotNet - Random {NUM_SAMPLES} Samples Evaluation', fontsize=16, fontweight='bold')

    for i in range(4):
        ax = plt.subplot(4, 2, 2*i + 1)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.scatter(range(len(abs_errors)), abs_errors[:, i], color='tab:red', alpha=0.7, s=15)
        ax.set_title(f'Absolute Error Distribution: {action_names[i]}', fontsize=12)
        ax.set_ylabel('Absolute Error')
        ax.grid(True, linestyle=':', alpha=0.6)

    ax_bar = plt.subplot(1, 2, 2)
    bars = ax_bar.bar(action_names, mean_abs_errors, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:purple'], alpha=0.8)
    ax_bar.set_title('Mean Absolute Error (MAE) per Dimension', fontsize=14)
    ax_bar.set_ylabel('Mean Error Value')
    ax_bar.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    for bar in bars:
        height = bar.get_height()
        ax_bar.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(EVALUATION_PATH), exist_ok=True)
    plt.savefig(EVALUATION_PATH, dpi=300)
    print(f"Evaluation chart saved to {EVALUATION_PATH}")

if __name__ == '__main__':
    evaluate_resnet_random_samples()