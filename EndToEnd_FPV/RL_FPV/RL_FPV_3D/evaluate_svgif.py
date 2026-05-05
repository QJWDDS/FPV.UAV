import torch
import numpy as np
from env import MonocularUAV3DEnv
from ddpg import DDPGAgent
from config import cfg
import os
import imageio
from PIL import Image

def main():
    if not os.path.exists(cfg.MODEL_ACTOR_PATH):
        print("Error: Model file not found. Please run train.py first.")
        return

    gif_dir = "gifs"
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)
        print(f"Created directory '{gif_dir}' for saving GIFs.")

    print("Starting 3D Evaluation with Pyramid FOV Visualization and GIF saving...")

    env = MonocularUAV3DEnv(render_mode='human')

    # 动作维度3: [v_cmd_x, v_cmd_z, yaw_rate_cmd]
    max_act = np.array([cfg.V_X_MAX, cfg.V_Z_MAX, cfg.YAW_RATE_MAX])
    agent = DDPGAgent(state_dim=5, action_dim=3, max_action=max_act)

    agent.actor.load_state_dict(torch.load(cfg.MODEL_ACTOR_PATH, map_location=cfg.DEVICE))
    agent.actor.eval()

    episodes_to_watch = 5  # 评估的回合数
    for i in range(episodes_to_watch):
        state, _ = env.reset()
        print(f"Evaluation Episode {i + 1}/{episodes_to_watch}")

        expected_shape = None
        frames = []
        done = False
        
        while not done:
            action = agent.select_action(state)
            state, reward, terminated, truncated, _ = env.step(action)

            env.render()

            if env.fig:
                try:
                    env.fig.canvas.draw()
                    
                    image = np.array(env.fig.canvas.buffer_rgba())                 
                    frame_rgb = image[:, :, :3]
                    if expected_shape is None:
                        expected_shape = frame_rgb.shape[:2]
                    if frame_rgb.shape[:2] != expected_shape:
                        img_pil = Image.fromarray(frame_rgb)
                        img_pil = img_pil.resize((expected_shape[1], expected_shape[0]), Image.Resampling.LANCZOS)
                        frame_rgb = np.array(img_pil)

                    frames.append(frame_rgb)

                except Exception as e:
                    print(f"Warning: Frame capture issue - {e}")

            done = terminated or truncated
            if terminated:
                print(" -> Intercepted or Crashed")
            elif truncated:
                print(" -> Time Limit Reached")

        if len(frames) > 0:
            gif_path = os.path.join(gif_dir, f'eval_episode_{cfg.TRAIN_VERSION}_{i + 1}.gif')
            try:
                # fps=15
                imageio.mimsave(gif_path, frames, fps=15, loop=0)
                print(f" -> Saved GIF successfully to {gif_path}")
            except ValueError as ve:
                print(f" -> Failed to save GIF due to ValueError: {ve}")
            except Exception as e:
                print(f" -> Failed to save GIF: {e}")

    print("Evaluation finished.")
    env.close()

if __name__ == "__main__":
    main()