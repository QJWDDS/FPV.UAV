import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config import cfg

class MonocularUAV3DEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, render_mode=None):
        super(MonocularUAV3DEnv, self).__init__()

        # 动作空间: 机体系 [v_x, v_z, yaw_rate]
        self.max_action = np.array([cfg.V_X_MAX, cfg.V_Z_MAX, cfg.YAW_RATE_MAX])
        self.action_space = spaces.Box(low=-self.max_action, high=self.max_action, dtype=np.float64)

        # 观测空间: [v_x, v_z, 方位角偏差(水平), 俯仰角偏差(垂直), 是否可见]
        high_obs = np.array([cfg.V_X_MAX, cfg.V_Z_MAX, np.pi, np.pi, 1.0])
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float64)

        self.render_mode = render_mode
        self.fig, self.ax = None, None

    def reset(self, seed=None):
        super().reset(seed=seed)

        # UAV [x, y, z, yaw]
        self.uav_state = np.array([0.0, 0.0, cfg.FIELD_SIZE_Z / 2, 0.0])
        self.uav_vel_body = np.zeros(2)

        # 目标
        valid_spawn = False
        while not valid_spawn:
            dist = np.random.uniform(3.0, cfg.MAX_SENSE_DIST * 0.8)
            azimuth = np.random.uniform(-cfg.FOV_H / 2.5, cfg.FOV_H / 2.5)
            elevation = np.random.uniform(-cfg.FOV_V / 2.5, cfg.FOV_V / 2.5)

            dx_b = dist * np.cos(elevation) * np.cos(azimuth)
            dy_b = dist * np.cos(elevation) * np.sin(azimuth)
            dz_b = dist * np.sin(elevation)

            yaw = self.uav_state[3]
            t_x = self.uav_state[0] + dx_b * np.cos(yaw) - dy_b * np.sin(yaw)
            t_y = self.uav_state[1] + dx_b * np.sin(yaw) + dy_b * np.cos(yaw)
            t_z = self.uav_state[2] + dz_b

            if (abs(t_x) < cfg.FIELD_SIZE_XY / 2 and
                    abs(t_y) < cfg.FIELD_SIZE_XY / 2 and
                    0 < t_z < cfg.FIELD_SIZE_Z):
                valid_spawn = True
                self.target_pos = np.array([t_x, t_y, t_z])

                vel_theta = np.random.uniform(0, 2 * np.pi)
                vel_phi = np.random.uniform(0, np.pi)
                v_mag = np.random.uniform(1.0, cfg.TARGET_V_MAX)
                self.target_vel = np.array([
                    v_mag * np.sin(vel_phi) * np.cos(vel_theta),
                    v_mag * np.sin(vel_phi) * np.sin(vel_theta),
                    v_mag * np.cos(vel_phi) * 0.5
                ])

        self.uav_traj = [self.uav_state[:3].copy()]
        self.target_traj = [self.target_pos.copy()]
        self.steps = 0

        obs, _, _ = self._get_obs()
        return obs, {}

    def step(self, action):
        cmd_vx = np.clip(action[0], -cfg.V_X_MAX, cfg.V_X_MAX)
        cmd_vz = np.clip(action[1], -cfg.V_Z_MAX, cfg.V_Z_MAX)
        cmd_omega = np.clip(action[2], -cfg.YAW_RATE_MAX, cfg.YAW_RATE_MAX)

        # 低通滤波
        self.uav_vel_body[0] += (cmd_vx - self.uav_vel_body[0]) * 0.2
        self.uav_vel_body[1] += (cmd_vz - self.uav_vel_body[1]) * 0.2

        # 运动学更新
        yaw = self.uav_state[3]
        v_wx = self.uav_vel_body[0] * np.cos(yaw)
        v_wy = self.uav_vel_body[0] * np.sin(yaw)
        v_wz = self.uav_vel_body[1]

        self.uav_state[3] += cmd_omega * cfg.DT
        self.uav_state[3] = self._wrap_angle(self.uav_state[3])
        self.uav_state[0] += v_wx * cfg.DT
        self.uav_state[1] += v_wy * cfg.DT
        self.uav_state[2] += v_wz * cfg.DT

        self.target_pos += self.target_vel * cfg.DT
        limit_xy = cfg.FIELD_SIZE_XY / 2
        for i in range(2):
            if abs(self.target_pos[i]) > limit_xy:
                self.target_vel[i] *= -1
                self.target_pos[i] = np.clip(self.target_pos[i], -limit_xy, limit_xy)
        if self.target_pos[2] > cfg.FIELD_SIZE_Z or self.target_pos[2] < 0:
            self.target_vel[2] *= -1
            self.target_pos[2] = np.clip(self.target_pos[2], 0, cfg.FIELD_SIZE_Z)

        self.uav_traj.append(self.uav_state[:3].copy())
        self.target_traj.append(self.target_pos.copy())

        obs, angle_err, dist = self._get_obs()
        is_visible = obs[4] > 0.5

        # 奖励
        reward = -0.05
        terminated = False
        truncated = False

        if is_visible:
            azimuth_err, elevation_err = angle_err
            align_reward = 0.1 * ((1.0 - abs(azimuth_err) / (cfg.FOV_H / 2)) +
                                  (1.0 - abs(elevation_err) / (cfg.FOV_V / 2)))
            reward += align_reward
            reward += (cfg.MAX_SENSE_DIST - dist) * 0.01

            if dist < cfg.CAPTURE_DIST:
                reward += 2000.0
                terminated = True
        else:
            reward -= 0.5

        if (abs(self.uav_state[0]) > limit_xy or
                abs(self.uav_state[1]) > limit_xy or
                self.uav_state[2] < 0 or self.uav_state[2] > cfg.FIELD_SIZE_Z):
            reward -= 10
            terminated = True

        self.steps += 1
        if self.steps >= cfg.MAX_STEPS:
            truncated = True

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        dx_w = self.target_pos[0] - self.uav_state[0]
        dy_w = self.target_pos[1] - self.uav_state[1]
        dz_w = self.target_pos[2] - self.uav_state[2]
        dist = np.linalg.norm([dx_w, dy_w, dz_w])

        yaw = self.uav_state[3]
        dx_b = dx_w * np.cos(-yaw) - dy_w * np.sin(-yaw)
        dy_b = dx_w * np.sin(-yaw) + dy_w * np.cos(-yaw)
        dz_b = dz_w

        azimuth_err = np.arctan2(dy_b, dx_b)
        elevation_err = np.arctan2(dz_b, np.sqrt(dx_b ** 2 + dy_b ** 2))

        is_visible = (dist < cfg.MAX_SENSE_DIST) and \
                     (abs(azimuth_err) < cfg.FOV_H / 2) and \
                     (abs(elevation_err) < cfg.FOV_V / 2) and (dx_b > 0)

        obs_azimuth = azimuth_err if is_visible else 0.0
        obs_elevation = elevation_err if is_visible else 0.0

        obs = np.array([
            self.uav_vel_body[0], self.uav_vel_body[1],
            obs_azimuth, obs_elevation,
            1.0 if is_visible else 0.0
        ], dtype=np.float64)

        return obs, (azimuth_err, elevation_err), dist

    def _wrap_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def render(self):
        if self.render_mode != 'human': return

        if self.fig is None:
            plt.ion()
            self.fig = plt.figure(figsize=(10, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')

        self.ax.cla()
        limit = cfg.FIELD_SIZE_XY / 2
        self.ax.set_xlim([-limit, limit])
        self.ax.set_ylim([-limit, limit])
        self.ax.set_zlim([0, cfg.FIELD_SIZE_Z])
        self.ax.set_box_aspect((1, 1, cfg.FIELD_SIZE_Z / cfg.FIELD_SIZE_XY))
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Height (m)')
        self.ax.set_title("3D Monocular UAV Interception (Heave/Surge/Yaw)")

        traj_u = np.array(self.uav_traj)
        traj_t = np.array(self.target_traj)

        self.ax.plot(traj_u[:, 0], traj_u[:, 1], traj_u[:, 2], 'b-', alpha=0.5, label='UAV Traj')
        self.ax.plot(traj_t[:, 0], traj_t[:, 1], traj_t[:, 2], 'r--', alpha=0.5, label='Target Traj')
        self.ax.scatter(*self.target_pos, color='red', s=50, label='Target')

        ux, uy, uz, uyaw = self.uav_state
        arm_len = 1.0
        rotors = [
            (arm_len * np.cos(uyaw + np.pi / 4), arm_len * np.sin(uyaw + np.pi / 4)),
            (arm_len * np.cos(uyaw + 3 * np.pi / 4), arm_len * np.sin(uyaw + 3 * np.pi / 4)),
            (arm_len * np.cos(uyaw + 5 * np.pi / 4), arm_len * np.sin(uyaw + 5 * np.pi / 4)),
            (arm_len * np.cos(uyaw + 7 * np.pi / 4), arm_len * np.sin(uyaw + 7 * np.pi / 4))
        ]

        self.ax.plot([ux + rotors[0][0], ux + rotors[2][0]], [uy + rotors[0][1], uy + rotors[2][1]], [uz, uz], 'k-', lw=2)
        self.ax.plot([ux + rotors[1][0], ux + rotors[3][0]], [uy + rotors[1][1], uy + rotors[3][1]], [uz, uz], 'k-', lw=2)

        for rx, ry in rotors:
            self.ax.scatter(ux + rx, uy + ry, uz, color='blue', s=30)
        self.ax.scatter(ux, uy, uz, color='black', s=50)

        # 绘制 FOV
        D = cfg.MAX_SENSE_DIST
        angles = [
            (cfg.FOV_H / 2, cfg.FOV_V / 2),    # 右上
            (-cfg.FOV_H / 2, cfg.FOV_V / 2),   # 左上
            (-cfg.FOV_H / 2, -cfg.FOV_V / 2),  # 左下
            (cfg.FOV_H / 2, -cfg.FOV_V / 2)    # 右下
        ]

        obs, _, _ = self._get_obs()
        cone_color = 'green' if obs[4] > 0.5 else 'orange'

        corners = []
        for h_ang, v_ang in angles:
            dx_b = D * np.cos(v_ang) * np.cos(h_ang)
            dy_b = D * np.cos(v_ang) * np.sin(h_ang)
            dz_b = D * np.sin(v_ang)

            cx = ux + dx_b * np.cos(uyaw) - dy_b * np.sin(uyaw)
            cy = uy + dx_b * np.sin(uyaw) + dy_b * np.cos(uyaw)
            cz = uz + dz_b
            corners.append((cx, cy, cz))

        for corner in corners:
            self.ax.plot([ux, corner[0]], [uy, corner[1]], [uz, corner[2]], color=cone_color, linestyle='-', alpha=0.4)

        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color=cone_color, linestyle='-', alpha=0.4)

        self.ax.legend(loc='upper right')
        plt.pause(0.001)

    def close(self):
        if self.fig: plt.close(self.fig)