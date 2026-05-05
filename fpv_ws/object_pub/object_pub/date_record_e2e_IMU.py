#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from px4_msgs.msg import TrajectorySetpoint, VehicleAttitude, VehicleOdometry # 新增 Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
import csv
import time
import math

class DataRecorder(Node):
    def __init__(self):
        super().__init__('e2e_data_recorder')

        self.declare_parameter('save_dir', os.path.expanduser('~/sh_ws/document/baylands_data/IndependentE2E_IMU'))
        self.save_dir = self.get_parameter('save_dir').value

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.save_dir, timestamp)
        self.img_dir = os.path.join(self.session_dir, 'images')
        os.makedirs(self.img_dir, exist_ok=True)
        
        # State: [roll, pitch, yaw_rate, state_vx, state_vy, state_vz]
        # Action: [cmd_vx, cmd_vy, cmd_vz, cmd_yaw_rate]
        self.csv_path = os.path.join(self.session_dir, 'data.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'img_name', 
            'state_roll', 'state_pitch', 'state_yaw_rate', 'state_vx', 'state_vy', 'state_vz', # State (Input)
            'cmd_vx', 'cmd_vy', 'cmd_vz', 'cmd_yaw_rate' # Action (Label)
        ])

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.bridge = CvBridge()
        self.sub_img = self.create_subscription(Image, '/world/baylands/model/x500_mono_cam_0/link/camera_link/sensor/camera/image', self.img_callback, qos_profile)
        self.sub_traj = self.create_subscription(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', self.traj_callback, qos_profile)
        self.sub_att = self.create_subscription(VehicleAttitude, '/fmu/out/vehicle_attitude', self.att_callback, qos_profile)
        self.sub_odom = self.create_subscription(VehicleOdometry, '/fmu/out/vehicle_odometry', self.odom_callback, qos_profile)

        self.latest_traj = None
        self.latest_att = None
        self.latest_odom = None
        self.count = 0
        self.get_logger().info(f"Recording to {self.session_dir}")

    def traj_callback(self, msg):
        self.latest_traj = msg

    def att_callback(self, msg):
        self.latest_att = msg

    def odom_callback(self, msg):
        self.latest_odom = msg

    def q_mult(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return np.array([w, x, y, z])

    def ned_to_body(self, q, v_ned):
        # v_body = q_inv * v_ned * q
        v_ned_q = np.array([0.0, v_ned[0], v_ned[1], v_ned[2]])
        q_inv = np.array([q[0], -q[1], -q[2], -q[3]])
        temp = self.q_mult(q_inv, v_ned_q)
        v_body = self.q_mult(temp, q)
        return v_body[1:]

    def get_euler_from_q(self, q):
        # return roll, pitch, yaw
        w, x, y, z = q
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    def img_callback(self, msg):
        if self.latest_traj is None or self.latest_att is None or self.latest_odom is None:
            return

        cmd_vx_ned = self.latest_traj.velocity[0]
        cmd_vy_ned = self.latest_traj.velocity[1]
        cmd_vz_ned = self.latest_traj.velocity[2]
        cmd_yaw_rate = self.latest_traj.yawspeed

        if np.isnan(cmd_vx_ned): return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            q_current = np.array(self.latest_att.q)

            roll, pitch, _ = self.get_euler_from_q(q_current)
            
            real_yaw_rate = self.latest_odom.angular_velocity[2]

            #  odom.velocity NED 转 Body
            real_v_ned = np.array(self.latest_odom.velocity) 
            state_v_body = self.ned_to_body(q_current, real_v_ned)

            cmd_v_ned = np.array([cmd_vx_ned, cmd_vy_ned, cmd_vz_ned])
            cmd_v_body = self.ned_to_body(q_current, cmd_v_ned)

            filename = f"{self.count:06d}.jpg"
            img_save_path = os.path.join(self.img_dir, filename)
            cv2.imwrite(img_save_path, cv_image)
            

            self.csv_writer.writerow([
                filename, 
                f"{roll:.4f}", f"{pitch:.4f}", f"{real_yaw_rate:.4f}", 
                f"{state_v_body[0]:.4f}", f"{state_v_body[1]:.4f}", f"{state_v_body[2]:.4f}",
                f"{cmd_v_body[0]:.4f}", f"{cmd_v_body[1]:.4f}", f"{cmd_v_body[2]:.4f}", f"{cmd_yaw_rate:.4f}"
            ])
            
            self.count += 1
            if self.count % 100 == 0:
                self.get_logger().info(f"Recorded {self.count} frames. State Vz: {state_v_body[2]:.2f}")

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DataRecorder()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()