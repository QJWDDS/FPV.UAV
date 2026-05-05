import torch
import numpy as np
from env import MonocularUAVEnv
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

    print("Starting Evaluation with Visualization...")
    env = MonocularUAVEnv(render_mode='human')

    agent = DDPGAgent(
        state_dim=4,
        action_dim=2,
        max_action=np.array([cfg.V_MAX, cfg.YAW_RATE_MAX])
    )

    agent.actor.load_state_dict(torch.load(cfg.MODEL_ACTOR_PATH, map_location=cfg.DEVICE))
    agent.actor.eval()  

    episodes_to_watch = 5  ###
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

        if len(frames) > 0:
            gif_path = os.path.join(gif_dir, f'eval_episode_{cfg.TRAIN_VERSION}_{i + 1}.gif')
            # fps
            imageio.mimsave(gif_path, frames, fps=15, loop=0)
            print(f" -> Saved GIF to {gif_path}")

    print("Evaluation finished.")
    env.close()


if __name__ == "__main__":
    main()