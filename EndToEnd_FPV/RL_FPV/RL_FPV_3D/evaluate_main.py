import torch
import numpy as np
import os
from env import MonocularUAV3DEnv
from ddpg import DDPGAgent
from config import cfg


def main():
    if not os.path.exists(cfg.MODEL_ACTOR_PATH):
        print("Error: Model file not found. Please run train.py first.")
        return

    print("Starting 3D Evaluation with Pyramid FOV Visualization...")
    env = MonocularUAV3DEnv(render_mode='human')

    max_act = np.array([cfg.V_X_MAX, cfg.V_Z_MAX, cfg.YAW_RATE_MAX])
    agent = DDPGAgent(state_dim=5, action_dim=3, max_action=max_act)

    agent.actor.load_state_dict(torch.load(cfg.MODEL_ACTOR_PATH, map_location=cfg.DEVICE))
    agent.actor.eval()

    episodes_to_watch = 10
    for i in range(episodes_to_watch):
        state, _ = env.reset()
        print(f"Evaluation Episode {i + 1}/{episodes_to_watch}")

        done = False
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)
            env.render()

            done = terminated or truncated
            if terminated:
                print(" -> Intercepted or Crashed")
            elif truncated:
                print(" -> Time Limit Reached")

    print("Evaluation finished.")
    env.close()

if __name__ == "__main__":
    main()