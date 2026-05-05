import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from env import MonocularUAV3DEnv
from ddpg import DDPGAgent, OUActionNoise
from config import cfg

def save_reward_curve(rewards_history, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    rewards = np.array(rewards_history, dtype=np.float64)
    episodes = np.arange(1, len(rewards) + 1)
    smooth_window = min(20, len(rewards))
    moving_avg = np.convolve(rewards, np.ones(smooth_window) / smooth_window, mode="valid")

    np.save(os.path.join(save_dir, f"episode_rewards_{cfg.TRAIN_VERSION}.npy"), rewards)
    np.savetxt(
        os.path.join(save_dir, f"episode_rewards_{cfg.TRAIN_VERSION}.csv"),
        np.column_stack((episodes, rewards)),
        delimiter=",",
        header="episode,reward",
        comments=""
    )

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    ax.scatter(
        episodes,
        rewards,
        s=8,
        alpha=0.35,
        color="#4C78A8",
        edgecolors="none",
        label="Episode reward",
        rasterized=True,
        zorder=2,
    )

    ax.plot(
        np.arange(smooth_window, len(rewards) + 1),
        moving_avg,
        linewidth=2.2,
        color="#D62728",
        label=f"Moving average ({smooth_window})",
        zorder=3,
    )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=4, width=1.0)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35, zorder=1)
    ax.legend(frameon=False, loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"reward_scatter_paper_{cfg.TRAIN_VERSION}.png"), dpi=600, bbox_inches="tight")
    plt.close(fig)



def main():
    if not os.path.exists('models'):
        os.makedirs('models')

    print(f"Initializing 3D Training (Reduced Action Space) on {cfg.DEVICE}...")
    env = MonocularUAV3DEnv()

    # 状态维度5: [v_bx, v_bz, 偏航角偏, 俯仰角偏, 是否可见]
    # 动作维度3: [v_cmd_x, v_cmd_z, yaw_rate_cmd]
    max_act = np.array([cfg.V_X_MAX, cfg.V_Z_MAX, cfg.YAW_RATE_MAX])
    agent = DDPGAgent(state_dim=5, action_dim=3, max_action=max_act)

    noise = OUActionNoise(mean=np.zeros(3), std_deviation=float(0.2) * np.ones(3))
    rewards_history = []

    for episode in range(cfg.MAX_EPISODES):
        state, _ = env.reset()
        noise.reset()
        episode_reward = 0

        for step in range(cfg.MAX_STEPS):
            if len(agent.memory) < cfg.WARMUP_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, noise())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.memory.push(state, action, reward, next_state, done)

            if len(agent.memory) > cfg.WARMUP_STEPS:
                agent.update()

            state = next_state
            episode_reward += reward

            if done:
                break

        rewards_history.append(episode_reward)
        avg_rew = np.mean(rewards_history[-20:]) if rewards_history else 0
        print(f"Ep {episode + 1}/{cfg.MAX_EPISODES} | Steps: {step + 1} | Reward: {episode_reward:.2f} | Avg(20): {avg_rew:.2f}")

    torch.save(agent.actor.state_dict(), cfg.MODEL_ACTOR_PATH)
    torch.save(agent.critic.state_dict(), cfg.MODEL_CRITIC_PATH)
    print(f"Models saved to {cfg.MODEL_ACTOR_PATH} & {cfg.MODEL_CRITIC_PATH}")

    save_reward_curve(rewards_history, save_dir="results")
    print("Reward history saved")

    env.close()


if __name__ == "__main__":
    main()
