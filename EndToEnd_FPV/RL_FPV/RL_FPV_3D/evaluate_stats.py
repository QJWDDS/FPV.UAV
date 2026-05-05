import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from env import MonocularUAV3DEnv
from ddpg import DDPGAgent
from config import cfg

def main():
    if not os.path.exists(cfg.MODEL_ACTOR_PATH):
        print("Error: Model file not found. Please run train.py first.")
        return

    # 评估回合数
    num_episodes = 1000
    print(f"Starting Statistical Evaluation for {num_episodes} episodes (No Rendering)...")
    
    env = MonocularUAV3DEnv(render_mode=None)

    # 动作维度3: [v_cmd_x, v_cmd_z, yaw_rate_cmd]
    max_act = np.array([cfg.V_X_MAX, cfg.V_Z_MAX, cfg.YAW_RATE_MAX])
    agent = DDPGAgent(state_dim=5, action_dim=3, max_action=max_act)

    agent.actor.load_state_dict(torch.load(cfg.MODEL_ACTOR_PATH, map_location=cfg.DEVICE))
    agent.actor.eval()

    stats = {
        'success': 0,
        'crash': 0,
        'timeout': 0,
        'min_distances': [],
        'rewards': [],
        'steps': []
    }

    for i in range(num_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        ep_step = 0
        ep_min_dist = float('inf')

        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)

            uav_pos = env.uav_state[:3]
            target_pos = env.target_pos
            dist = np.linalg.norm(uav_pos - target_pos)

            if dist < ep_min_dist:
                ep_min_dist = dist
                
            ep_reward += reward
            ep_step += 1
            
            done = terminated or truncated

            if done:
                if terminated:
                    if dist <= cfg.CAPTURE_DIST:
                        stats['success'] += 1
                    else:
                        stats['crash'] += 1
                elif truncated:
                    stats['timeout'] += 1

        stats['min_distances'].append(ep_min_dist)
        stats['rewards'].append(ep_reward)
        stats['steps'].append(ep_step)
        
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i + 1}/{num_episodes} episodes...")

    print("Evaluation Complete. Generating charts...")
    env.close()
    
    # 绘制统计图表
    generate_charts(stats, num_episodes)


def generate_charts(stats, num_episodes):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'UAV Interception Evaluation Stats ({num_episodes} Episodes)', fontsize=16)

    # ================= 任务结果饼图 =================
    failed_count = stats['crash'] + stats['timeout']
    
    labels = ['Success (Intercepted)', 'Failed (Crashed/Timeout)']
    sizes = [stats['success'], failed_count]
    colors = ['#4CAF50', '#F44336']
    explode = (0.1, 0)

    labels_filtered = [l for s, l in zip(sizes, labels) if s > 0]
    sizes_filtered = [s for s in sizes if s > 0]
    colors_filtered = [c for s, c in zip(sizes, colors) if s > 0]
    explode_filtered = tuple([0.1 if 'Success' in l else 0 for l in labels_filtered])

    axs[0].pie(sizes_filtered, explode=explode_filtered, labels=labels_filtered, colors=colors_filtered,
               autopct='%1.1f%%', shadow=True, startangle=140)
    axs[0].set_title('Interception Result')

    # ================= 最小逼近距离分布直方图 =================
    axs[1].hist(stats['min_distances'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axs[1].axvline(x=cfg.CAPTURE_DIST, color='r', linestyle='dashed', linewidth=2, label=f'Capture Threshold ({cfg.CAPTURE_DIST}m)')
    axs[1].set_title('Minimum Approach Distance Distribution')
    axs[1].set_xlabel('Distance (m)')
    axs[1].set_ylabel('Frequency')
    axs[1].legend()

    # ================= 每回合步数散点图 =================
    episodes_x = np.arange(1, num_episodes + 1)

    success_mask = np.array(stats['min_distances']) <= cfg.CAPTURE_DIST
    
    axs[2].scatter(episodes_x[success_mask], np.array(stats['steps'])[success_mask], 
                   color='green', label='Success', alpha=0.6)
    axs[2].scatter(episodes_x[~success_mask], np.array(stats['steps'])[~success_mask], 
                   color='red', label='Failed', alpha=0.6)

    avg_steps = np.mean(stats['steps'])
    axs[2].axhline(y=avg_steps, color='blue', linestyle='-.', label=f'Avg Steps: {avg_steps:.1f}')
    
    axs[2].set_title('Steps per Episode')
    axs[2].set_xlabel('Episode')
    axs[2].set_ylabel('Total Steps')
    axs[2].legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    if not os.path.exists("evaluation"):
        os.makedirs("evaluation")
        
    save_path = f"evaluation/evaluation_stats_{cfg.TRAIN_VERSION}.png"
    plt.savefig(save_path, dpi=300)
    print(f"Stats charts saved successfully to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    main()