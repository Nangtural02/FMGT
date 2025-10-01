# 파일명: follower_estimator_node.py
import rclpy
import numpy as np
import math
import threading
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
import tf2_ros

from scipy.spatial.transform import Rotation

# --- 상수 및 유틸리티 함수 ---
GRAVITY = np.array([0, 0, -9.81])

def normalize_angle(angle): return (angle + np.pi) % (2 * np.pi) - np.pi
def yaw_to_quaternion(yaw):
    q = Rotation.from_euler('z', yaw).as_quat(); return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

class FollowerEstimatorNode(Node):
    def __init__(self):
        super().__init__('follower_estimator_node')
        
        self.follower_pose_pub = self.create_publisher(PoseStamped, '/follower/estimated_pose', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.lock = threading.Lock()

        self.x_f = np.zeros(9)
        self.P_f = np.eye(9) * 0.1
        self.Q_f = np.diag([1e-8, 1e-8, 1e-6, 0.05**2, 0.05**2, 0.05**2, 0.01**2, 0.01**2, (math.radians(0.1))**2])
        odom_pos_var, odom_yaw_var = 0.05**2, (math.radians(0.05))**2
        self.R_odom = np.diag([odom_pos_var, odom_pos_var, odom_yaw_var])
        
        self.is_initialized = False
        self.last_imu_timestamp = None
        self.initial_odom_pos = None
        self.initial_odom_yaw = 0.0
        self.last_odom_pose = None

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_predict_callback, sensor_qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_update_callback, sensor_qos)
        
        self.get_logger().info("Follower Estimator Node 시작됨.")
        
    def imu_predict_callback(self, imu_msg):
        with self.lock:
            if not self.is_initialized: return
            
            current_timestamp = rclpy.time.Time.from_msg(imu_msg.header.stamp).nanoseconds / 1e9
            if self.last_imu_timestamp is None: self.last_imu_timestamp = current_timestamp; return
            dt = current_timestamp - self.last_imu_timestamp
            self.last_imu_timestamp = current_timestamp
            if not (0 < dt < 0.5): return
            
            q_orientation = np.array([imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w])
            accel_original = np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z])
            omega_original = np.array([imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z])
            
            imu_rotation = Rotation.from_quat(q_orientation)
            pure_accel = accel_original - imu_rotation.inv().apply(GRAVITY)
            
            b_ax, b_ay, b_wz = self.x_f[6], self.x_f[7], self.x_f[8]
            ax_local, ay_local, omega_z_local = pure_accel[0] - b_ax, pure_accel[1] - b_ay, omega_original[2] - b_wz
            
            theta_f = self.x_f[2]
            cos_th, sin_th = math.cos(theta_f), math.sin(theta_f)
            ax_world, ay_world = ax_local * cos_th - ay_local * sin_th, ax_local * sin_th + ay_local * cos_th
            
            self.x_f[0] += self.x_f[3] * dt; self.x_f[1] += self.x_f[4] * dt
            self.x_f[2] = normalize_angle(self.x_f[2] + self.x_f[5] * dt)
            self.x_f[3] += ax_world * dt; self.x_f[4] += ay_world * dt; self.x_f[5] = omega_z_local
            
            F_f = np.eye(9)
            F_f[0, 3]=dt; F_f[1, 4]=dt; F_f[2, 5]=dt
            F_f[3, 2]=(-ax_local*sin_th-ay_local*cos_th)*dt; F_f[4, 2]=(ax_local*cos_th-ay_local*sin_th)*dt
            F_f[3, 6]=-cos_th*dt; F_f[3, 7]=sin_th*dt; F_f[4, 6]=-sin_th*dt; F_f[4, 7]=-cos_th*dt; F_f[5, 8]=-dt
            
            self.P_f = F_f @ self.P_f @ F_f.T + self.Q_f * dt

            header = imu_msg.header; header.frame_id = 'world'
            f_pose = PoseStamped(header=header, pose={'position': {'x': self.x_f[0], 'y': self.x_f[1]}, 'orientation': yaw_to_quaternion(self.x_f[2])})
            self.follower_pose_pub.publish(f_pose)

            if self.last_odom_pose:
                T_world_base = np.eye(4); T_world_base[:3, :3] = Rotation.from_euler('z', self.x_f[2]).as_matrix(); T_world_base[:2, 3] = self.x_f[:2]
                T_odom_base = np.eye(4)
                q_o = self.last_odom_pose.orientation
                T_odom_base[:3, :3] = Rotation.from_quat([q_o.x, q_o.y, q_o.z, q_o.w]).as_matrix()
                T_odom_base[0:2, 3] = [self.last_odom_pose.position.x, self.last_odom_pose.position.y]
                
                T_world_odom = T_world_base @ np.linalg.inv(T_odom_base)
                
                t = TransformStamped(); t.header.stamp = imu_msg.header.stamp
                t.header.frame_id = 'world'; t.child_frame_id = 'odom'
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = T_world_odom[0:3, 3]
                q_tf = Rotation.from_matrix(T_world_odom[:3, :3]).as_quat()
                t.transform.rotation = Quaternion(x=q_tf[0], y=q_tf[1], z=q_tf[2], w=q_tf[3])
                self.tf_broadcaster.sendTransform(t)

    def odom_update_callback(self, odom_msg):
        with self.lock:
            self.last_odom_pose = odom_msg.pose.pose
            if not self.is_initialized:
                q_odom = odom_msg.pose.pose.orientation
                self.initial_odom_yaw = Rotation.from_quat([q_odom.x, q_odom.y, q_odom.z, q_odom.w]).as_euler('zyx')[0]
                self.initial_odom_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
                self.is_initialized = True
                self.get_logger().info("Follower EKF 초기화 성공!")
                return
            
            q_odom = odom_msg.pose.pose.orientation
            odom_yaw = Rotation.from_quat([q_odom.x, q_odom.y, q_odom.z, q_odom.w]).as_euler('zyx')[0]
            odom_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
            
            relative_pos = odom_pos - self.initial_odom_pos
            relative_yaw = normalize_angle(odom_yaw - self.initial_odom_yaw)
            
            z = np.array([relative_pos[0], relative_pos[1], relative_yaw])
            h_x = self.x_f[[0, 1, 2]]
            H_f = np.zeros((3, 9)); H_f[0, 0]=1; H_f[1, 1]=1; H_f[2, 2]=1
            
            y_err = z - h_x; y_err[2] = normalize_angle(y_err[2])
            S = H_f @ self.P_f @ H_f.T + self.R_odom
            K = self.P_f @ H_f.T @ np.linalg.inv(S)
            
            self.x_f += K @ y_err; self.x_f[2] = normalize_angle(self.x_f[2])
            self.P_f = (np.eye(9) - K @ H_f) @ self.P_f

def main(args=None):
    rclpy.init(args=args)
    node = FollowerEstimatorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()