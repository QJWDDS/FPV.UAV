import torch
import numpy as np
from env import MonocularUAVEnv
from ddpg import DDPGAgent
from config import cfg
import os

def main():
    if not os.path.exists(cfg.MODEL_ACTOR_PATH):
        print("Error: Model file not found. Please run train.py first.")
        return

    print("Starting Evaluation with Visualization...")

    env = MonocularUAVEnv(render_mode='human')

    agent = DDPGAgent(
        state_dim=4,
        action_dim=2,
        max_action=np.array([cfg.V_MAX, cfg.YAW_RATE_MAX])
    )

    agent.actor.load_state_dict(torch.load(cfg.MODEL_ACTOR_PATH, map_location=cfg.DEVICE))
    agent.actor.eval()  

    episodes_to_watch = 20  ###
    for i in range(episodes_to_watch):
        state, _ = env.reset()
        print(f"Evaluation Episode {i + 1}/{episodes_to_watch}")

        done = False
        while not done:
            # 不加噪声
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            # 渲染画面
            env.render()
            done = terminated or truncated
            if terminated:
                print(" -> Intercepted or Crashed")

    print("Evaluation finished.")
    env.close()

if __name__ == "__main__":
    main()