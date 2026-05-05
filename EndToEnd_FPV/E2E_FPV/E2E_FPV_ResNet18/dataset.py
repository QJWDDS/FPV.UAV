import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np

class DroneDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_stats=None):
        self.root_dir = root_dir
        self.transform = transform
        self.all_data = []

        print(f"Scanning data in: {root_dir} ...")
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        # 遍历读取
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            csv_path = os.path.join(folder_path, 'data.csv')
            if os.path.isdir(folder_path) and os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df['folder_path'] = folder_path
                    self.all_data.append(df)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")

        self.combined_frame = pd.concat(self.all_data, ignore_index=True)
        print(f"Total dataset size: {len(self.combined_frame)} samples.")

        # --- 标签标准化 (Z-Score Normalization) ---
        self.label_cols = ['v_body_x', 'v_body_y', 'v_body_z', 'yaw_rate_cmd']
        
        if label_stats is None:
            labels_np = self.combined_frame[self.label_cols].values
            self.mean = np.mean(labels_np, axis=0)
            self.std = np.std(labels_np, axis=0)
            self.std[self.std < 1e-6] = 1e-6 # 防止除以 0
            print(f"Calculated Label Mean: {self.mean}")
            print(f"Calculated Label Std:  {self.std}")
        else:
            self.mean = np.array(label_stats['mean'])
            self.std = np.array(label_stats['std'])

    def get_label_stats(self):
        """返回标准化参数"""
        return {
            'mean': self.mean.tolist(),
            'std': self.std.tolist()
        }

    def __len__(self):
        return len(self.combined_frame)

    def __getitem__(self, idx):
        row = self.combined_frame.iloc[idx]
        img_name = row['img_name']
        folder_path = row['folder_path']
        img_path = os.path.join(folder_path, 'images', img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__(np.random.randint(0, len(self)))
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        
        raw_label = row[self.label_cols].values.astype(np.float32)
        norm_label = (raw_label - self.mean) / self.std

        return image, torch.tensor(norm_label, dtype=torch.float32)