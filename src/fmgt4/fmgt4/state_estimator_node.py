
import rclpy, math
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point, PointStamped, TwistStamped, Twist, Quaternion, Vector3
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

GRAVITY = np.array([0, 0, -9.81])
# ★★★ 변경점: 실제 "헤드라이트" 앵커 위치 모델 적용 ★★★
# 이 값들은 실제 로봇에 맞게 측정/수정해야 합니다.
ANCHOR_FORWARD_OFFSET = 0.25  # 로봇 중심에서 앵커까지의 전방(x) 거리 (예: 20cm)
ANCHOR_WIDTH = 0.4           # 두 앵커 사이의 전체 폭 (y)
ANCHOR_A_LOCAL_POS = np.array([ANCHOR_FORWARD_OFFSET, ANCHOR_WIDTH / 2])  # Front-Left
ANCHOR_B_LOCAL_POS = np.array([ANCHOR_FORWARD_OFFSET, -ANCHOR_WIDTH / 2]) # Front-Right

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class StateEstimatorOdomNode(Node):
    def __init__(self):
        super().__init__('state_estimator_odom_node')
        self.follower_pos_pub = self.create_publisher(PointStamped, '/follower/estimated_position', 10)
        self.leader_pos_pub = self.create_publisher(PointStamped, '/leader/estimated_position', 10)
        self.follower_vel_pub = self.create_publisher(TwistStamped, '/follower/estimated_velocity', 10)

        self.x = np.zeros(10)
        self.P = np.eye(10) * 0.1
        self.P[6:8, 6:8] *= 1000
        self.P[8:10, 8:10] = 0.01**2

        leader_pos_process_noise = 1e-6
        bias_process_noise = 0.01**2
        self.Q = np.diag([
            1e-8, 1e-8, 0.05**2, 0.05**2, 1e-6, 0.05**2,
            leader_pos_process_noise, leader_pos_process_noise,
            bias_process_noise, bias_process_noise
        ])
        
        uwb_noise_variance = 0.3**2
        odom_pos_variance = 0.1**2
        odom_yaw_variance = (math.radians(2.0))**2
        self.R_uwb = np.diag([uwb_noise_variance, uwb_noise_variance])
        self.R_odom = np.diag([odom_pos_variance, odom_pos_variance, odom_yaw_variance])

        self.last_timestamp = None
        self.is_initialized = False
        self.initial_odom_pos = None
        self.initial_odom_yaw = 0.0
        
        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.uwb_sub = self.create_subscription(
            PointStamped, 'raw_uwb_distances', self.uwb_update_callback, sensor_qos_profile)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_predict_callback, sensor_qos_profile)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_update_callback, sensor_qos_profile)
        
        self.get_logger().info('State Estimator with Odom (Headlight Anchors) 시작됨.')

    def imu_predict_callback(self, imu_msg):
        if not self.is_initialized: return
        current_timestamp = imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec * 1e-9
        dt = current_timestamp - self.last_timestamp
        self.last_timestamp = current_timestamp
        if not (0 < dt < 0.5): return
        
        q_orientation = np.array([imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w])
        accel_original = np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z])
        omega_original = np.array([imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z])

        imu_rotation = Rotation.from_quat(q_orientation)
        gravity_in_imu_frame = imu_rotation.inv().apply(GRAVITY)
        pure_accel = accel_original - gravity_in_imu_frame
        
        b_ax_body, b_ay_body = self.x[8], self.x[9]
        ax_body = pure_accel[0] - b_ax_body
        ay_body = pure_accel[1] - b_ay_body
        
        theta_f = self.x[2]
        cos_th, sin_th = math.cos(theta_f), math.sin(theta_f)
        ax_world = ax_body * cos_th - ay_body * sin_th
        ay_world = ax_body * sin_th + ay_body * cos_th
        
        self.x[0] += self.x[3] * dt; self.x[1] += self.x[4] * dt
        self.x[2] = normalize_angle(self.x[2] + self.x[5] * dt)
        self.x[3] += ax_world * dt; self.x[4] += ay_world * dt
        self.x[5] = omega_original[2]
        
        F = np.eye(10)
        F[0,3]=dt; F[1,4]=dt; F[2,5]=dt
        F[3,2] = (-ax_body * sin_th - ay_body * cos_th) * dt
        F[4,2] = ( ax_body * cos_th - ay_body * sin_th) * dt
        F[3,8] = -cos_th*dt; F[3,9] = sin_th*dt
        F[4,8] = -sin_th*dt; F[4,9] = -cos_th*dt
        self.P = F @ self.P @ F.T + self.Q

        header = Header(stamp=imu_msg.header.stamp, frame_id='world')
        f_pos = PointStamped(header=header, point=Point(x=self.x[0], y=self.x[1], z=0.0))
        l_pos = PointStamped(header=header, point=Point(x=self.x[6], y=self.x[7], z=0.0))
        f_vel = TwistStamped(header=header, twist=Twist(linear=Vector3(x=self.x[3], y=self.x[4], z=0.0), angular=Vector3(z=self.x[5])))
        self.follower_pos_pub.publish(f_pos); self.leader_pos_pub.publish(l_pos); self.follower_vel_pub.publish(f_vel)

    def odom_update_callback(self, odom_msg):
        q_odom = odom_msg.pose.pose.orientation
        odom_yaw = Rotation.from_quat([q_odom.x, q_odom.y, q_odom.z, q_odom.w]).as_euler('zyx', degrees=False)[0]
        odom_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
        if not self.is_initialized:
            self.initial_odom_pos = odom_pos
            self.initial_odom_yaw = odom_yaw
            self.last_timestamp = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9
            self.x[0:3] = [0.0, 0.0, 0.0]; self.is_initialized = True
            self.get_logger().info(f"EKF 초기화 성공! (Odometry 기준)")
            return
        relative_pos = odom_pos - self.initial_odom_pos
        relative_yaw = normalize_angle(odom_yaw - self.initial_odom_yaw)
        z = np.array([relative_pos[0], relative_pos[1], relative_yaw])
        h_x = np.array([self.x[0], self.x[1], self.x[2]])
        H = np.zeros((3, 10)); H[0,0] = 1.0; H[1,1] = 1.0; H[2,2] = 1.0
        y_err = z - h_x; y_err[2] = normalize_angle(y_err[2])
        S = H @ self.P @ H.T + self.R_odom
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y_err; self.x[2] = normalize_angle(self.x[2])
            self.P = (np.eye(10) - K @ H) @ self.P
        except np.linalg.LinAlgError: pass

    def uwb_update_callback(self, uwb_msg):
        if not self.is_initialized: return
        if np.all(self.x[6:8] == 0):
            try:
                d_a, d_b = uwb_msg.point.x, uwb_msg.point.y
                if not (d_a > 0 and d_b > 0): return

                # ★★★ 변경점: 새로운 "헤드라이트" 앵커 위치에 맞는 초기 위치 계산식 ★★★
                X_off = ANCHOR_A_LOCAL_POS[0]
                Y_off = ANCHOR_A_LOCAL_POS[1]
                py_local = (d_b**2 - d_a**2) / (4 * Y_off)
                px_sq = d_a**2 - (py_local - Y_off)**2
                if px_sq < 0: return
                px_local = X_off + math.sqrt(px_sq) # Assume leader is in front
                
                initial_yaw = self.x[2]
                cos_th, sin_th = math.cos(initial_yaw), math.sin(initial_yaw)
                self.x[6] = self.x[0] + (px_local * cos_th - py_local * sin_th)
                self.x[7] = self.x[1] + (px_local * sin_th + py_local * cos_th)
                self.get_logger().info(f"리더 초기 위치 설정 완료: ({self.x[6]:.2f}, {self.x[7]:.2f})")
            except Exception: return

        z = np.array([uwb_msg.point.x, uwb_msg.point.y])
        pf_x, pf_y, theta_f, _, _, _, pl_x, pl_y, _, _ = self.x
        cos_th, sin_th = math.cos(theta_f), math.sin(theta_f)
        rot_matrix = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
        anchor_A_world = self.x[0:2] + rot_matrix @ ANCHOR_A_LOCAL_POS
        anchor_B_world = self.x[0:2] + rot_matrix @ ANCHOR_B_LOCAL_POS
        d_a_pred = np.linalg.norm(self.x[6:8] - anchor_A_world)
        d_b_pred = np.linalg.norm(self.x[6:8] - anchor_B_world)

        if d_a_pred > 1e-6 and d_b_pred > 1e-6:
            h_x = np.array([d_a_pred, d_b_pred])
            H = np.zeros((2, 10))
            dx_a = pl_x - anchor_A_world[0]; dy_a = pl_y - anchor_A_world[1]
            dx_b = pl_x - anchor_B_world[0]; dy_b = pl_y - anchor_B_world[1]
            H[0,0]=-dx_a/d_a_pred; H[0,1]=-dy_a/d_a_pred; H[0,6]=dx_a/d_a_pred; H[0,7]=dy_a/d_a_pred
            H[1,0]=-dx_b/d_b_pred; H[1,1]=-dy_b/d_b_pred; H[1,6]=dx_b/d_b_pred; H[1,7]=dy_b/d_b_pred

            # ★★★ 변경점: 새로운 앵커 위치에 맞는 자코비안 H 행렬 ★★★
            d_anchor_A_d_th_x = -ANCHOR_A_LOCAL_POS[0]*sin_th - ANCHOR_A_LOCAL_POS[1]*cos_th
            d_anchor_A_d_th_y =  ANCHOR_A_LOCAL_POS[0]*cos_th - ANCHOR_A_LOCAL_POS[1]*sin_th
            H[0,2] = (1/d_a_pred) * (dx_a * d_anchor_A_d_th_x + dy_a * d_anchor_A_d_th_y)
            d_anchor_B_d_th_x = -ANCHOR_B_LOCAL_POS[0]*sin_th - ANCHOR_B_LOCAL_POS[1]*cos_th
            d_anchor_B_d_th_y =  ANCHOR_B_LOCAL_POS[0]*cos_th - ANCHOR_B_LOCAL_POS[1]*sin_th
            H[1,2] = (1/d_b_pred) * (dx_b * d_anchor_B_d_th_x + dy_b * d_anchor_B_d_th_y)
            
            y_err = z - h_x
            S = H @ self.P @ H.T + self.R_uwb
            try:
                K = self.P @ H.T @ np.linalg.inv(S)
                self.x = self.x + K @ y_err; self.x[2] = normalize_angle(self.x[2])
                self.P = (np.eye(10) - K @ H) @ self.P
            except np.linalg.LinAlgError: pass
def main(args=None):
    rclpy.init(args=args)
    node = StateEstimatorOdomNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()