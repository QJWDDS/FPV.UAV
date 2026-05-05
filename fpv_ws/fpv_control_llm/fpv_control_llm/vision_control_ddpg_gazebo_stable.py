#!/usr/bin/env python3
import math
import threading
from enum import Enum
from typing import List, Tuple
import os
import numpy as np
import rclpy
import torch
import torch.nn as nn
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data

from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleAttitude, VehicleCommand, VehicleOdometry
from fpv_control.msg import ImageDeviation


class Actor(nn.Module):
    """5 -> 256 -> 128 -> 3, tanh * max_action"""

    def __init__(self, state_dim: int, action_dim: int, max_action: np.ndarray, device: torch.device):
        super().__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, action_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.max_action = torch.tensor(max_action, dtype=torch.float32, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        return self.max_action * self.tanh(self.l3(x))


class State(Enum):
    TAKEOFF = 0
    HOVER = 1
    GUIDANCE = 2


class VisionControlDDPG(Node):
    def __init__(self) -> None:
        super().__init__("vision_control_ddpg")

        home_dir = os.path.expanduser('~')
        modelpath = os.path.join(home_dir, 'EndToEnd_FPV/RL_FPV/RL_FPV_3D/models/uav_actor_3d_v1.pth')
        # ===== 参数 =====
        self.declare_parameter("model_path", modelpath)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("control_period", 0.02)       # 50 Hz 发布 setpoint
        self.declare_parameter("policy_period", 0.10)        # 10 Hz 刷新策略
        self.declare_parameter("dt_model", 0.10)             # 训练环境 DT
        self.declare_parameter("velocity_lpf_alpha", 0.2)    # 训练环境速度低通 alpha
        self.declare_parameter("use_time_scaled_lpf", True)

        # ===== 角度定义 =====
        self.declare_parameter("angle_unit_deg", True)
        self.declare_parameter("invert_angle_x_for_policy", False)
        self.declare_parameter("invert_yaw_rate_output", False)

        # ===== 日志 =====
        self.declare_parameter("log_guidance_debug", True)
        self.declare_parameter("log_every_n", 1)

        # ===== 起飞/悬停/丢失 =====
        self.declare_parameter("takeoff_relative_altitude", 10.0)
        self.declare_parameter("takeoff_timeout", 30.0)
        self.declare_parameter("min_relative_altitude", 0.5)
        self.declare_parameter("max_relative_altitude", 50.0)
        self.declare_parameter("target_loss_timeout", 0.5)
        self.declare_parameter("target_loss_max_count", 25)

        # ===== 训练动作上限 =====
        self.declare_parameter("v_x_max", 5.0)
        self.declare_parameter("v_z_max", 2.0)
        self.declare_parameter("yaw_rate_max", math.radians(45.0))

        # ===== 部署动作整形 =====
        self.declare_parameter("deploy_scale_vx", 0.30)
        self.declare_parameter("deploy_scale_vz", 0.35)
        self.declare_parameter("deploy_scale_yaw", 0.60)
        self.declare_parameter("deploy_vx_limit", 1.5)
        self.declare_parameter("deploy_vz_limit", 0.7)
        self.declare_parameter("deploy_yaw_rate_limit", math.radians(22.0))

        # ===== 混合安全覆盖=====
        self.declare_parameter("use_safe_yaw_override", True)
        self.declare_parameter("use_safe_vz_override", True)
        self.declare_parameter("override_only_if_conflict", True)
        self.declare_parameter("safe_yaw_gain", 1.8)
        self.declare_parameter("safe_vertical_gain", 1.2)

        # ===== 平滑 / 防抖 =====
        self.declare_parameter("angle_filter_alpha", 0.25)
        self.declare_parameter("measurement_hold_timeout", 0.20)
        self.declare_parameter("command_filter_alpha", 0.15)
        self.declare_parameter("yaw_filter_alpha", 0.20)
        self.declare_parameter("enable_angle_speed_gating", True)
        self.declare_parameter("angle_gate_stop_deg", 30.0)
        self.declare_parameter("angle_gate_min_scale", 0.15)
        self.declare_parameter("stale_vx_min_scale", 0.50)
        self.declare_parameter("stale_yaw_min_scale", 0.70)

        # ===== 话题 =====
        self.declare_parameter("deviation_topic", "/camera/image_deviation")
        self.declare_parameter("attitude_topic", "/fmu/out/vehicle_attitude")
        self.declare_parameter("odometry_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("offboard_control_mode_topic", "/fmu/in/offboard_control_mode")
        self.declare_parameter("trajectory_setpoint_topic", "/fmu/in/trajectory_setpoint")
        self.declare_parameter("vehicle_command_topic", "/fmu/in/vehicle_command")

        # ===== 读取参数 =====
        self.model_path = str(self.get_parameter("model_path").value)
        device_name = str(self.get_parameter("device").value)
        self.device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else device_name)

        self.control_period = float(self.get_parameter("control_period").value)
        self.policy_period = float(self.get_parameter("policy_period").value)
        self.dt_model = float(self.get_parameter("dt_model").value)
        self.velocity_lpf_alpha = float(self.get_parameter("velocity_lpf_alpha").value)
        self.use_time_scaled_lpf = bool(self.get_parameter("use_time_scaled_lpf").value)

        self.angle_unit_deg = bool(self.get_parameter("angle_unit_deg").value)
        self.invert_angle_x_for_policy = bool(self.get_parameter("invert_angle_x_for_policy").value)
        self.invert_yaw_rate_output = bool(self.get_parameter("invert_yaw_rate_output").value)

        self.log_guidance_debug = bool(self.get_parameter("log_guidance_debug").value)
        self.log_every_n = max(1, int(self.get_parameter("log_every_n").value))

        self.takeoff_relative_altitude = float(self.get_parameter("takeoff_relative_altitude").value)
        self.takeoff_timeout = float(self.get_parameter("takeoff_timeout").value)
        self.min_relative_altitude = float(self.get_parameter("min_relative_altitude").value)
        self.max_relative_altitude = float(self.get_parameter("max_relative_altitude").value)
        self.target_loss_timeout = float(self.get_parameter("target_loss_timeout").value)
        self.target_loss_max_count = int(self.get_parameter("target_loss_max_count").value)

        self.v_x_max = float(self.get_parameter("v_x_max").value)
        self.v_z_max = float(self.get_parameter("v_z_max").value)
        self.yaw_rate_max = float(self.get_parameter("yaw_rate_max").value)

        self.deploy_scale_vx = float(self.get_parameter("deploy_scale_vx").value)
        self.deploy_scale_vz = float(self.get_parameter("deploy_scale_vz").value)
        self.deploy_scale_yaw = float(self.get_parameter("deploy_scale_yaw").value)
        self.deploy_vx_limit = float(self.get_parameter("deploy_vx_limit").value)
        self.deploy_vz_limit = float(self.get_parameter("deploy_vz_limit").value)
        self.deploy_yaw_rate_limit = float(self.get_parameter("deploy_yaw_rate_limit").value)

        self.use_safe_yaw_override = bool(self.get_parameter("use_safe_yaw_override").value)
        self.use_safe_vz_override = bool(self.get_parameter("use_safe_vz_override").value)
        self.override_only_if_conflict = bool(self.get_parameter("override_only_if_conflict").value)
        self.safe_yaw_gain = float(self.get_parameter("safe_yaw_gain").value)
        self.safe_vertical_gain = float(self.get_parameter("safe_vertical_gain").value)

        self.angle_filter_alpha = float(self.get_parameter("angle_filter_alpha").value)
        self.measurement_hold_timeout = float(self.get_parameter("measurement_hold_timeout").value)
        self.command_filter_alpha = float(self.get_parameter("command_filter_alpha").value)
        self.yaw_filter_alpha = float(self.get_parameter("yaw_filter_alpha").value)
        self.enable_angle_speed_gating = bool(self.get_parameter("enable_angle_speed_gating").value)
        self.angle_gate_stop_deg = float(self.get_parameter("angle_gate_stop_deg").value)
        self.angle_gate_min_scale = float(self.get_parameter("angle_gate_min_scale").value)
        self.stale_vx_min_scale = float(self.get_parameter("stale_vx_min_scale").value)
        self.stale_yaw_min_scale = float(self.get_parameter("stale_yaw_min_scale").value)

        self.deviation_topic = str(self.get_parameter("deviation_topic").value)
        self.attitude_topic = str(self.get_parameter("attitude_topic").value)
        self.odometry_topic = str(self.get_parameter("odometry_topic").value)
        self.offboard_control_mode_topic = str(self.get_parameter("offboard_control_mode_topic").value)
        self.trajectory_setpoint_topic = str(self.get_parameter("trajectory_setpoint_topic").value)
        self.vehicle_command_topic = str(self.get_parameter("vehicle_command_topic").value)

        self.train_max_action = np.array([self.v_x_max, self.v_z_max, self.yaw_rate_max], dtype=np.float32)
        self.policy_update_steps = max(1, int(round(self.policy_period / max(self.control_period, 1e-6))))
        if self.use_time_scaled_lpf and 0.0 < self.velocity_lpf_alpha < 1.0:
            self.velocity_lpf_alpha_eff = 1.0 - (1.0 - self.velocity_lpf_alpha) ** (self.control_period / max(self.dt_model, 1e-6))
        else:
            self.velocity_lpf_alpha_eff = self.velocity_lpf_alpha

        # ===== 模型 =====
        self.actor = Actor(5, 3, self.train_max_action, self.device).to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.actor.load_state_dict(state_dict)
        self.actor.eval()

        # ===== 锁 =====
        self.deviation_lock = threading.Lock()
        self.attitude_lock = threading.Lock()
        self.odom_lock = threading.Lock()

        # ===== 状态 =====
        self.offboard_setpoint_counter = 0
        self.current_state = State.TAKEOFF
        self.takeoff_start_time = self.get_clock().now()
        self.last_valid_target_time = self.get_clock().now()
        self.last_deviation_time = self.get_clock().now()

        self.has_new_deviation = False
        self.last_msg_valid = False
        self.has_attitude_data = False
        self.has_odometry = False
        self.has_arm_position = False
        self.has_valid_target = False

        self.current_attitude_q: List[float] = [1.0, 0.0, 0.0, 0.0]
        self.current_yaw = 0.0
        self.takeoff_yaw = 0.0

        self.current_position_x = 0.0
        self.current_position_y = 0.0
        self.current_position_z = 0.0

        self.last_angle_x = float("nan")
        self.last_angle_y = float("nan")
        self.last_valid_angle_x = 0.0
        self.last_valid_angle_y = 0.0
        self.filtered_angle_x = 0.0
        self.filtered_angle_y = 0.0
        self.target_loss_count = 0

        self.arm_position_x = 0.0
        self.arm_position_y = 0.0
        self.base_altitude = 0.0

        self.hover_position_x = 0.0
        self.hover_position_y = 0.0
        self.hover_position_z = 0.0

        self.target_velocity_x = 0.0
        self.target_velocity_y = 0.0
        self.target_velocity_z = 0.0
        self.des_yaw_rate = 0.0

        self.est_v_bx = 0.0
        self.est_v_bz_up = 0.0

        # ===== 策略/动作缓存 =====
        self.policy_tick = 0
        self.guidance_debug_counter = 0
        self.last_policy_state = np.zeros(5, dtype=np.float32)
        self.last_policy_angle_x_raw = 0.0
        self.last_policy_angle_y_raw = 0.0
        self.last_action_raw = np.zeros(3, dtype=np.float32)
        self.last_action_shaped = np.zeros(3, dtype=np.float32)
        self.last_action_filtered = np.zeros(3, dtype=np.float32)
        self.last_policy_refresh = False
        self.last_mode = "idle"
        self.last_new_dev = False
        self.last_fresh_valid = False
        self.last_msg_age = 1e9
        self.last_vx_gate = 1.0
        self.last_stale_scale = 1.0
        self.last_yaw_src = "ddpg"
        self.last_vz_src = "ddpg"

        # ===== 发布器/订阅器 =====
        pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.offboard_control_mode_publisher = self.create_publisher(OffboardControlMode, self.offboard_control_mode_topic, pub_qos)
        self.trajectory_setpoint_publisher = self.create_publisher(TrajectorySetpoint, self.trajectory_setpoint_topic, pub_qos)
        self.vehicle_command_publisher = self.create_publisher(VehicleCommand, self.vehicle_command_topic, pub_qos)

        self.image_deviation_sub = self.create_subscription(ImageDeviation, self.deviation_topic, self.image_deviation_callback, qos_profile_sensor_data)
        self.vehicle_attitude_sub = self.create_subscription(VehicleAttitude, self.attitude_topic, self.vehicle_attitude_callback, qos_profile_sensor_data)
        self.vehicle_odometry_sub = self.create_subscription(VehicleOdometry, self.odometry_topic, self.vehicle_odometry_callback, qos_profile_sensor_data)

        self.timer = self.create_timer(self.control_period, self.timer_callback)

        self.get_logger().info("Starting vision_control_ddpg")
        self.get_logger().info(f"Model path: {self.model_path}")
        self.get_logger().info(f"Device: {self.device}")
        self.get_logger().info(
            f"Train limits: vx={self.v_x_max:.2f} m/s, vz={self.v_z_max:.2f} m/s, yaw_rate={math.degrees(self.yaw_rate_max):.1f} deg/s"
        )
        self.get_logger().info(
            f"Deploy limits: vx={self.deploy_vx_limit:.2f} m/s, vz={self.deploy_vz_limit:.2f} m/s, yaw_rate={math.degrees(self.deploy_yaw_rate_limit):.1f} deg/s"
        )
        self.get_logger().info(
            f"policy_period={self.policy_period:.3f}s, control_period={self.control_period:.3f}s, alpha_eff={self.velocity_lpf_alpha_eff:.4f}"
        )

    # ======================= 回调 =======================
    def image_deviation_callback(self, msg: ImageDeviation) -> None:
        now = self.get_clock().now()
        ax = float(msg.angle_x)
        ay = float(msg.angle_y)
        valid = not (math.isnan(ax) or math.isnan(ay))
        with self.deviation_lock:
            self.last_angle_x = ax
            self.last_angle_y = ay
            self.last_msg_valid = valid
            self.last_deviation_time = now
            self.has_new_deviation = True
            if valid:
                self.last_valid_angle_x = ax
                self.last_valid_angle_y = ay
                self.last_valid_target_time = now
                self.target_loss_count = 0
                self.has_valid_target = True

    def vehicle_attitude_callback(self, msg: VehicleAttitude) -> None:
        with self.attitude_lock:
            self.current_attitude_q = [float(msg.q[0]), float(msg.q[1]), float(msg.q[2]), float(msg.q[3])]
            self.current_yaw = self.get_yaw_from_quaternion(self.current_attitude_q)
            self.has_attitude_data = True

    def vehicle_odometry_callback(self, msg: VehicleOdometry) -> None:
        with self.odom_lock:
            self.current_position_x = float(msg.position[0])
            self.current_position_y = float(msg.position[1])
            self.current_position_z = float(msg.position[2])
            self.has_odometry = True

    # ======================= 工具函数 =======================
    @staticmethod
    def normalize_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    @staticmethod
    def get_yaw_from_quaternion(q: List[float]) -> float:
        w, x, y, z = q
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def body_to_ground(q: List[float], body_vec: np.ndarray) -> np.ndarray:
        """FRD 机体系 -> NED"""
        w, x, y, z = q
        r = np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
            ],
            dtype=np.float64,
        )
        return r @ body_vec

    def convert_angles_for_policy(self, angle_x_raw: float, angle_y_raw: float) -> Tuple[float, float]:
        angle_x_used = -angle_x_raw if self.invert_angle_x_for_policy else angle_x_raw
        if self.angle_unit_deg:
            azimuth = math.radians(angle_x_used)
            elevation = -math.radians(angle_y_raw)  # 话题 angle_y 下正上负 -> 训练里上正下负
        else:
            azimuth = angle_x_used
            elevation = -angle_y_raw
        return azimuth, elevation

    def update_filtered_angles(self, angle_x_raw: float, angle_y_raw: float, reset: bool = False) -> None:
        if reset:
            self.filtered_angle_x = angle_x_raw
            self.filtered_angle_y = angle_y_raw
            return
        a = float(np.clip(self.angle_filter_alpha, 0.0, 1.0))
        self.filtered_angle_x += (angle_x_raw - self.filtered_angle_x) * a
        self.filtered_angle_y += (angle_y_raw - self.filtered_angle_y) * a

    def build_policy_state(self, angle_x_raw: float, angle_y_raw: float, visible: bool) -> np.ndarray:
        azimuth, elevation = self.convert_angles_for_policy(angle_x_raw, angle_y_raw)
        state = np.array(
            [
                self.est_v_bx,
                self.est_v_bz_up,
                azimuth if visible else 0.0,
                elevation if visible else 0.0,
                1.0 if visible else 0.0,
            ],
            dtype=np.float32,
        )
        self.last_policy_state = state.copy()
        self.last_policy_angle_x_raw = angle_x_raw
        self.last_policy_angle_y_raw = angle_y_raw
        return state

    def infer_action(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = torch.tensor(state.reshape(1, -1), dtype=torch.float32, device=self.device)
            action = self.actor(x).cpu().numpy().reshape(-1)
        return np.clip(action, -self.train_max_action, self.train_max_action)

    def compute_vx_gate(self, angle_x_raw: float) -> float:
        if not self.enable_angle_speed_gating:
            return 1.0
        abs_deg = abs(angle_x_raw) if self.angle_unit_deg else math.degrees(abs(angle_x_raw))
        stop_deg = max(1e-6, self.angle_gate_stop_deg)
        scale = 1.0 - abs_deg / stop_deg
        return float(np.clip(scale, self.angle_gate_min_scale, 1.0))

    def compute_stale_scale(self, time_since_valid: float) -> Tuple[float, float]:
        if time_since_valid <= self.measurement_hold_timeout:
            return 1.0, 1.0
        if time_since_valid >= self.target_loss_timeout:
            return self.stale_vx_min_scale, self.stale_yaw_min_scale
        span = max(1e-6, self.target_loss_timeout - self.measurement_hold_timeout)
        ratio = (time_since_valid - self.measurement_hold_timeout) / span
        vx_scale = 1.0 - (1.0 - self.stale_vx_min_scale) * ratio
        yaw_scale = 1.0 - (1.0 - self.stale_yaw_min_scale) * ratio
        return float(vx_scale), float(yaw_scale)

    def compute_safe_yaw_rate(self, angle_x_raw: float) -> float:
        angle_x_rad = math.radians(angle_x_raw) if self.angle_unit_deg else angle_x_raw
        yaw_rate = self.safe_yaw_gain * angle_x_rad
        return float(np.clip(yaw_rate, -self.deploy_yaw_rate_limit, self.deploy_yaw_rate_limit))

    def compute_safe_vz_up(self, angle_y_raw: float) -> float:
        angle_y_rad = math.radians(angle_y_raw) if self.angle_unit_deg else angle_y_raw
        vz_up = -self.safe_vertical_gain * angle_y_rad
        return float(np.clip(vz_up, -self.deploy_vz_limit, self.deploy_vz_limit))

    def apply_safe_overrides(self, action_shaped: np.ndarray, angle_x_raw: float, angle_y_raw: float, visible: bool) -> np.ndarray:
        action = action_shaped.copy()
        self.last_yaw_src = "ddpg"
        self.last_vz_src = "ddpg"
        if not visible:
            return action

        if self.use_safe_yaw_override:
            safe_yaw = self.compute_safe_yaw_rate(angle_x_raw)
            conflict = (safe_yaw != 0.0 and action[2] != 0.0 and safe_yaw * float(action[2]) < 0.0)
            if (not self.override_only_if_conflict) or conflict:
                action[2] = safe_yaw
                self.last_yaw_src = "safe"

        if self.use_safe_vz_override:
            safe_vz = self.compute_safe_vz_up(angle_y_raw)
            conflict = (safe_vz != 0.0 and action[1] != 0.0 and safe_vz * float(action[1]) < 0.0)
            if (not self.override_only_if_conflict) or conflict:
                action[1] = safe_vz
                self.last_vz_src = "safe"

        return action

    def shape_action_for_deployment(self, action_raw: np.ndarray, angle_x_raw: float, time_since_valid: float) -> np.ndarray:
        vx = float(action_raw[0]) * self.deploy_scale_vx
        vz_up = float(action_raw[1]) * self.deploy_scale_vz
        yaw_rate = float(action_raw[2]) * self.deploy_scale_yaw

        vx_gate = self.compute_vx_gate(angle_x_raw)
        stale_vx_scale, stale_yaw_scale = self.compute_stale_scale(time_since_valid)
        self.last_vx_gate = vx_gate
        self.last_stale_scale = stale_vx_scale
        vx *= vx_gate * stale_vx_scale
        yaw_rate *= stale_yaw_scale

        vx = float(np.clip(vx, -self.deploy_vx_limit, self.deploy_vx_limit))
        vz_up = float(np.clip(vz_up, -self.deploy_vz_limit, self.deploy_vz_limit))
        yaw_rate = float(np.clip(yaw_rate, -self.deploy_yaw_rate_limit, self.deploy_yaw_rate_limit))
        return np.array([vx, vz_up, yaw_rate], dtype=np.float32)

    def smooth_command(self, desired_action: np.ndarray) -> np.ndarray:
        a_v = float(np.clip(self.command_filter_alpha, 0.0, 1.0))
        a_y = float(np.clip(self.yaw_filter_alpha, 0.0, 1.0))
        self.last_action_filtered[0] += (float(desired_action[0]) - float(self.last_action_filtered[0])) * a_v
        self.last_action_filtered[1] += (float(desired_action[1]) - float(self.last_action_filtered[1])) * a_v
        self.last_action_filtered[2] += (float(desired_action[2]) - float(self.last_action_filtered[2])) * a_y
        return self.last_action_filtered.copy()

    def action_to_ned(self, action_used: np.ndarray) -> Tuple[float, float, float, float]:
        cmd_vx = float(np.clip(action_used[0], -self.deploy_vx_limit, self.deploy_vx_limit))
        cmd_vz_up = float(np.clip(action_used[1], -self.deploy_vz_limit, self.deploy_vz_limit))
        cmd_yaw_rate = float(np.clip(action_used[2], -self.deploy_yaw_rate_limit, self.deploy_yaw_rate_limit))
        if self.invert_yaw_rate_output:
            cmd_yaw_rate = -cmd_yaw_rate

        self.est_v_bx += (cmd_vx - self.est_v_bx) * self.velocity_lpf_alpha_eff
        self.est_v_bz_up += (cmd_vz_up - self.est_v_bz_up) * self.velocity_lpf_alpha_eff

        body_vel_frd = np.array([self.est_v_bx, 0.0, -self.est_v_bz_up], dtype=np.float64)
        with self.attitude_lock:
            q = list(self.current_attitude_q)
        ned_vel = self.body_to_ground(q, body_vel_frd)
        return float(ned_vel[0]), float(ned_vel[1]), float(ned_vel[2]), cmd_yaw_rate

    def enforce_altitude_constraints(self) -> None:
        current_relative_alt = self.base_altitude - self.current_position_z
        if current_relative_alt < self.min_relative_altitude and self.target_velocity_z > 0.0:
            self.target_velocity_z = 0.0
            self.est_v_bz_up = max(0.0, self.est_v_bz_up)
        elif current_relative_alt > self.max_relative_altitude and self.target_velocity_z < 0.0:
            self.target_velocity_z = 0.0
            self.est_v_bz_up = min(0.0, self.est_v_bz_up)

    def reset_guidance_states(self) -> None:
        self.est_v_bx = 0.0
        self.est_v_bz_up = 0.0
        self.policy_tick = 0
        self.last_action_raw[:] = 0.0
        self.last_action_shaped[:] = 0.0
        self.last_action_filtered[:] = 0.0
        self.filtered_angle_x = self.last_valid_angle_x
        self.filtered_angle_y = self.last_valid_angle_y
        self.last_policy_refresh = False

    # ======================= 发布 =======================
    def publish_offboard_control_mode(self) -> None:
        msg = OffboardControlMode()
        if self.current_state in (State.TAKEOFF, State.HOVER):
            msg.position = True
            msg.velocity = False
        else:
            msg.position = False
            msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def publish_trajectory_setpoint(self) -> None:
        msg = TrajectorySetpoint()
        if self.current_state == State.TAKEOFF:
            msg.position = [float(self.arm_position_x), float(self.arm_position_y), float(self.base_altitude - self.takeoff_relative_altitude)]
            msg.velocity = [float("nan"), float("nan"), float("nan")]
            msg.yaw = float(self.takeoff_yaw)
        elif self.current_state == State.HOVER:
            msg.position = [float(self.hover_position_x), float(self.hover_position_y), float(self.hover_position_z)]
            msg.velocity = [float("nan"), float("nan"), float("nan")]
            msg.yaw = float(self.current_yaw)
        else:
            msg.position = [float("nan"), float("nan"), float("nan")]
            msg.velocity = [float(self.target_velocity_x), float(self.target_velocity_y), float(self.target_velocity_z)]
            msg.yaw = float("nan")
            msg.yawspeed = float(self.des_yaw_rate)
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def publish_vehicle_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        msg = VehicleCommand()
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = int(command)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def arm(self) -> None:
        with self.odom_lock:
            if not self.has_odometry:
                self.get_logger().error("No odometry data for arming")
                return
            self.arm_position_x = self.current_position_x
            self.arm_position_y = self.current_position_y
            self.base_altitude = self.current_position_z
            self.has_arm_position = True
        with self.attitude_lock:
            if self.has_attitude_data:
                self.takeoff_yaw = self.current_yaw
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0, 0.0)
        self.get_logger().info(
            f"Arm command sent, arm pos=({self.arm_position_x:.2f}, {self.arm_position_y:.2f}), base_z={self.base_altitude:.2f}"
        )

    def log_guidance_snapshot(self, angle_sub_x: float, angle_sub_y: float) -> None:
        if not self.log_guidance_debug:
            return
        self.guidance_debug_counter += 1
        if self.guidance_debug_counter % self.log_every_n != 0:
            return
        yaw_sign_relation = (
            "same-sign" if (self.last_policy_angle_x_raw == 0.0 or self.des_yaw_rate == 0.0 or self.last_policy_angle_x_raw * self.des_yaw_rate > 0.0)
            else "opposite-sign"
        )
        az_deg = math.degrees(float(self.last_policy_state[2]))
        el_deg = math.degrees(float(self.last_policy_state[3]))
        self.get_logger().info(
            (
                "[GUIDANCE] new_dev=%d | fresh_valid=%d | visible=%d | mode=%s | vx_gate=%.2f | stale=%.2f | "
                "msg_age=%.3f s | angle_sub=(%+.2f, %+.2f) deg | angle_policy=(%+.2f, %+.2f) deg | "
                "state=[vbx=%+.2f, vbz=%+.2f, az=%+.2f, el=%+.2f, vis=%.0f] | "
                "ddpg_raw=[vx=%+.2f, vz_up=%+.2f, yaw_rate=%+.3f rad/s (%+.1f deg/s)] | "
                "ddpg_shaped=[vx=%+.2f, vz_up=%+.2f, yaw_rate=%+.3f rad/s (%+.1f deg/s)] | "
                "cmd_used=[vx=%+.2f, vz_up=%+.2f, yaw_rate=%+.3f rad/s (%+.1f deg/s)] | src[yaw=%s,vz=%s] | "
                "px4_ned=[vn=%+.2f, ve=%+.2f, vd=%+.2f, yawspeed=%+.3f] | yaw_now=%+.1f deg | "
                "policy_refresh=%d | hold_steps=%d | alpha_eff=%.4f | yaw_vs_angle_x=%s"
            )
            % (
                1 if self.last_new_dev else 0,
                1 if self.last_fresh_valid else 0,
                int(self.last_policy_state[4] > 0.5),
                self.last_mode,
                self.last_vx_gate,
                self.last_stale_scale,
                self.last_msg_age,
                angle_sub_x,
                angle_sub_y,
                self.last_policy_angle_x_raw,
                self.last_policy_angle_y_raw,
                float(self.last_policy_state[0]),
                float(self.last_policy_state[1]),
                az_deg,
                el_deg,
                float(self.last_policy_state[4]),
                float(self.last_action_raw[0]),
                float(self.last_action_raw[1]),
                float(self.last_action_raw[2]),
                math.degrees(float(self.last_action_raw[2])),
                float(self.last_action_shaped[0]),
                float(self.last_action_shaped[1]),
                float(self.last_action_shaped[2]),
                math.degrees(float(self.last_action_shaped[2])),
                float(self.last_action_filtered[0]),
                float(self.last_action_filtered[1]),
                float(self.last_action_filtered[2]),
                math.degrees(float(self.last_action_filtered[2])),
                self.last_yaw_src,
                self.last_vz_src,
                self.target_velocity_x,
                self.target_velocity_y,
                self.target_velocity_z,
                self.des_yaw_rate,
                math.degrees(self.current_yaw),
                1 if self.last_policy_refresh else 0,
                self.policy_update_steps,
                self.velocity_lpf_alpha_eff,
                yaw_sign_relation,
            )
        )

    # ======================= 主控制 =======================
    def timer_callback(self) -> None:
        if self.offboard_setpoint_counter < 100:
            self.publish_offboard_control_mode()
            self.publish_trajectory_setpoint()
            self.offboard_setpoint_counter += 1
            if self.offboard_setpoint_counter == 50:
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                self.get_logger().info("Switching to Offboard mode")
            if self.offboard_setpoint_counter == 100:
                self.arm()
                self.takeoff_start_time = self.get_clock().now()
            return

        if not (self.has_arm_position and self.has_attitude_data and self.has_odometry):
            return

        now = self.get_clock().now()
        with self.deviation_lock:
            angle_x = self.last_angle_x
            angle_y = self.last_angle_y
            new_dev = self.has_new_deviation
            msg_valid = self.last_msg_valid
            msg_age = (now - self.last_deviation_time).nanoseconds * 1e-9
            self.has_new_deviation = False
        self.last_new_dev = new_dev
        self.last_msg_age = msg_age

        with self.odom_lock:
            current_relative_alt = self.base_altitude - self.current_position_z

        takeoff_elapsed = (now - self.takeoff_start_time).nanoseconds * 1e-9
        time_since_valid = (now - self.last_valid_target_time).nanoseconds * 1e-9

        if self.current_state == State.TAKEOFF:
            if current_relative_alt >= self.takeoff_relative_altitude * 0.95:
                self.get_logger().info(f"Takeoff complete ({current_relative_alt:.2f} m)")
                self.current_state = State.HOVER
                with self.odom_lock:
                    self.hover_position_x = self.current_position_x
                    self.hover_position_y = self.current_position_y
                    self.hover_position_z = self.current_position_z
            elif takeoff_elapsed > self.takeoff_timeout:
                self.get_logger().warn(f"Takeoff timeout after {takeoff_elapsed:.2f} s")
                self.current_state = State.HOVER

        elif self.current_state == State.HOVER:
            if new_dev and msg_valid:
                self.current_state = State.GUIDANCE
                self.last_valid_target_time = now
                self.target_loss_count = 0
                self.has_valid_target = True
                self.filtered_angle_x = angle_x
                self.filtered_angle_y = angle_y
                self.reset_guidance_states()
                self.get_logger().info("Target detected, entering GUIDANCE")
            else:
                with self.odom_lock:
                    self.hover_position_x = self.current_position_x
                    self.hover_position_y = self.current_position_y
                    self.hover_position_z = self.current_position_z

        elif self.current_state == State.GUIDANCE:
            if not self.has_valid_target:
                self.get_logger().warn("No valid target history, returning to HOVER")
                self.current_state = State.HOVER
            elif time_since_valid > self.target_loss_timeout or self.target_loss_count > self.target_loss_max_count:
                self.get_logger().warn(
                    f"Target lost (timeout={time_since_valid:.2f}s, count={self.target_loss_count}), returning to HOVER"
                )
                self.current_state = State.HOVER
                with self.odom_lock:
                    self.hover_position_x = self.current_position_x
                    self.hover_position_y = self.current_position_y
                    self.hover_position_z = self.current_position_z
                self.reset_guidance_states()
                self.target_velocity_x = 0.0
                self.target_velocity_y = 0.0
                self.target_velocity_z = 0.0
                self.des_yaw_rate = 0.0
            else:
                fresh_valid = bool(new_dev and msg_valid)
                if fresh_valid:
                    self.update_filtered_angles(angle_x, angle_y, reset=False)
                    self.last_valid_target_time = now
                    self.target_loss_count = 0
                    mode = "ddpg"
                else:
                    self.target_loss_count += 1
                    mode = "hold"

                # 短时无效/NaN
                use_angle_x = self.filtered_angle_x
                use_angle_y = self.filtered_angle_y
                self.last_fresh_valid = fresh_valid
                self.last_mode = mode

                self.policy_tick += 1
                refresh = fresh_valid and ((self.policy_tick == 1) or ((self.policy_tick - 1) % self.policy_update_steps == 0))
                state = self.build_policy_state(use_angle_x, use_angle_y, True)
                if refresh:
                    self.last_action_raw = self.infer_action(state)
                shaped = self.shape_action_for_deployment(self.last_action_raw, use_angle_x, time_since_valid)
                shaped = self.apply_safe_overrides(shaped, use_angle_x, use_angle_y, True)
                self.last_action_shaped = shaped
                self.last_policy_refresh = refresh
                cmd_body = self.smooth_command(self.last_action_shaped)

                vx_n, vy_e, vz_d, yaw_rate = self.action_to_ned(cmd_body)
                self.target_velocity_x = vx_n
                self.target_velocity_y = vy_e
                self.target_velocity_z = vz_d
                self.des_yaw_rate = yaw_rate
                self.enforce_altitude_constraints()
                angle_sub_x = angle_x if new_dev else float("nan")
                angle_sub_y = angle_y if new_dev else float("nan")
                self.log_guidance_snapshot(angle_sub_x, angle_sub_y)

        self.publish_offboard_control_mode()
        self.publish_trajectory_setpoint()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionControlDDPG()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()