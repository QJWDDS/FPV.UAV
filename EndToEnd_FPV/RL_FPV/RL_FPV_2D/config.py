import torch
import numpy as np

class Config:
    # --- 训练参数 ---
    MAX_EPISODES = 1000  # 总训练回合
    MAX_STEPS = 400  # 单回合最大步数
    BATCH_SIZE = 128  # 批次大小
    LR_ACTOR = 1e-4  # Actor 学习率
    LR_CRITIC = 1e-3  # Critic 学习率
    GAMMA = 0.99  # 折扣因子
    TAU = 0.005  # 软更新系数
    MEMORY_CAPACITY = 20000  # 经验回放池大小
    WARMUP_STEPS = 1000  # 预热步数（随机动作）

    # --- 无人机物理参数 ---
    DT = 0.1  # 仿真时间步长 (s)
    V_MAX = 5.0  # 最大线速度 (m/s)
    YAW_RATE_MAX = np.deg2rad(60)  # 最大偏航角速度 (rad/s)

    # --- 传感器(单目)参数 ---
    FOV = np.deg2rad(100)  # 视场角 (度)
    MAX_SENSE_DIST = 15.0  # 最大观测距离 m

    # --- 环境设定 ---
    FIELD_SIZE = 40.0  # 场地边长 (*m x *m)
    CAPTURE_DIST = 0.5  # 捕获判定距离
    TARGET_V_MAX = 4.8  # 目标最大移动速度

    # --- 系统配置 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    TRAIN_VERSION = "v1"
    MODEL_PATH = "models"
    MODEL_ACTOR_PATH = f"{MODEL_PATH}/uav_actor_{TRAIN_VERSION}.pth"
    MODEL_CRITIC_PATH = f"{MODEL_PATH}/uav_critic_{TRAIN_VERSION}.pth"


cfg = Config()