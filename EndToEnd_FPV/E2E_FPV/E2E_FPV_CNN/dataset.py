import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np
from torchvision import transforms

class DroneDataset(Dataset):
    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.transform = transform
        self.all_data = []

        print(f"Scanning data in: {root_dir} ...")
        valid_folders = 0
        
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")

        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            
            # data.csv
            csv_path = os.path.join(folder_path, 'data.csv')
            if os.path.isdir(folder_path) and os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df['folder_path'] = folder_path
                    
                    self.all_data.append(df)
                    valid_folders += 1
                    print(f"  -> Loaded {len(df)} samples from: {folder_name}")
                except Exception as e:
                    print(f"  [WARN] Failed to load {folder_name}: {e}")

        if valid_folders == 0:
            raise RuntimeError(f"No valid data folders found in {root_dir}")

        self.combined_frame = pd.concat(self.all_data, ignore_index=True)
        print(f"Total dataset size: {len(self.combined_frame)} samples from {valid_folders} folders.")

    def __len__(self):
        return len(self.combined_frame)

    def __getitem__(self, idx):
        row = self.combined_frame.iloc[idx]
        
        img_name = row['img_name']
        folder_path = row['folder_path']
        
        img_path = os.path.join(folder_path, 'images', img_name)
        
        # 读取图片
        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__(np.random.randint(0, len(self)))
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
        
        # 获取标签
        label_cols = ['v_body_x', 'v_body_y', 'v_body_z', 'yaw_rate_cmd']
        label_values = row[label_cols].values.astype(np.float32)
        
        label = torch.from_numpy(label_values)
        
        return image, label