import rclpy, math, time
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped, TwistStamped, Twist, Quaternion, Vector3
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
import message_filters

# 로봇 로컬 좌표계 (ROS 표준: +x 전방, +y 왼쪽)
ANCHOR_DISTANCE = 0.4
ANCHOR_A_LOCAL_POS = np.array([0.0, ANCHOR_DISTANCE / 2])
ANCHOR_B_LOCAL_POS = np.array([0.0, -ANCHOR_DISTANCE / 2])
GRAVITY = np.array([0, 0, 9.81])

def get_yaw_from_quaternion(q_ros):
    q_scipy = np.array([q_ros.x, q_ros.y, q_ros.z, q_ros.w])
    if np.linalg.norm(q_scipy) < 1e-6: return 0.0
    q_scipy /= np.linalg.norm(q_scipy)
    return Rotation.from_quat(q_scipy).as_euler('zyx', degrees=False)[0]

class StateEstimatorNode(Node):
    def __init__(self):
        super().__init__('state_estimator_node')
        self.follower_pos_pub = self.create_publisher(PointStamped, '/follower/estimated_position', 10)
        self.leader_pos_pub = self.create_publisher(PointStamped, '/leader/estimated_position', 10)
        self.follower_vel_pub = self.create_publisher(TwistStamped, '/follower/estimated_velocity', 10)
        self.leader_vel_pub = self.create_publisher(TwistStamped, '/leader/estimated_velocity', 10)

        self.x = np.zeros(10); self.P = np.eye(10) * 0.1; self.P[6:, 6:] *= 1000

        self.Q = np.diag([0.01**2,0.01**2,0.01**2, 0.1**2,0.1**2,0.1**2, 0.1**2,0.1**2,0.5**2,0.5**2])
        self.R = np.diag([0.1**2, 0.1**2, 0.05**2, 0.1**2])
        self.last_timestamp = None; self.is_initialized = False
        
        # ★★★ 당신이 말한 정확한 변환: 센서 좌표계를 Z축 +90도 회전하여 로봇 좌표계로 ★★★
        self.imu_correction_rotation = Rotation.from_euler('z', 90, degrees=True)
        
        uwb_sub = message_filters.Subscriber(self, PointStamped, 'raw_uwb_distances')
        imu_sub = message_filters.Subscriber(self, Imu, '/imu/data')
        self.ts = message_filters.ApproximateTimeSynchronizer([uwb_sub, imu_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.ekf_callback)
        self.get_logger().info('State Estimator (Z+90 Correction) 시작됨.')

    def ekf_callback(self, uwb_msg, imu_msg):
        # --- 1. IMU 데이터 수신 및 좌표계 변환 ---
        q_original = np.array([imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w])
        if np.linalg.norm(q_original) < 1e-6: return
        
        corrected_rotation = Rotation.from_quat(q_original) * self.imu_correction_rotation
        corrected_yaw = corrected_rotation.as_euler('zyx', degrees=False)[0]

        accel_original = np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z])
        omega_original = np.array([imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z])
        acc_corrected = self.imu_correction_rotation.apply(accel_original)
        omega_corrected = self.imu_correction_rotation.apply(omega_original)
        
        current_timestamp = time.time()
        
        if not self.is_initialized:
            try:
                d_a, d_b = uwb_msg.point.x, uwb_msg.point.y; L = ANCHOR_DISTANCE
                if not (d_a > 0 and d_b > 0 and d_a + d_b >= L and abs(d_a - d_b) <= L): return
                py_local = (d_a**2 - d_b**2) / (2 * L); px_sq = d_a**2 - (py_local - ANCHOR_A_LOCAL_POS[1])**2
                if px_sq < 0: return
                px_local = math.sqrt(px_sq)
                cos_th, sin_th = math.cos(corrected_yaw), math.sin(corrected_yaw)
                self.x[6] = 0 + (px_local * cos_th - py_local * sin_th); self.x[7] = 0 + (px_local * sin_th + py_local * cos_th)
                self.x[2] = corrected_yaw
                self.is_initialized = True
                self.last_timestamp = current_timestamp
                self.get_logger().info(f'EKF 초기화 성공! 리더 위치: ({self.x[6]:.2f}, {self.x[7]:.2f}), 팔로워 Yaw: {math.degrees(self.x[2]):.1f}도')
            except: return
            return

        dt = current_timestamp - self.last_timestamp; self.last_timestamp = current_timestamp
        if not (0 < dt < 0.5): return

        # --- 2. 예측 ---
        gravity_imu = corrected_rotation.inv().apply(GRAVITY.reshape(1, -1))[0]
        accel_compensated = acc_corrected - gravity_imu
        ax_local, ay_local = accel_compensated[0], accel_compensated[1]
        
        theta_f, vf_x_old, vf_y_old = self.x[2], self.x[3], self.x[4]
        cos_th, sin_th = math.cos(theta_f), math.sin(theta_f)
        vf_x_new = vf_x_old + (ax_local * cos_th - ay_local * sin_th) * dt
        vf_y_new = vf_y_old + (ax_local * sin_th + ay_local * cos_th) * dt
        omega_f_new = omega_corrected[2]

        # ★★★ 드리프트 억제를 위한 인공 감쇠 ★★★
        DRAG = 0.1
        vf_x_new *= (1.0 - DRAG * dt)
        vf_y_new *= (1.0 - DRAG * dt)
        
        self.x[0] += (vf_x_old + vf_x_new)/2*dt; self.x[1] += (vf_y_old + vf_y_new)/2*dt
        self.x[2] += (self.x[5] + omega_f_new)/2*dt; self.x[3], self.x[4] = vf_x_new, vf_y_new
        self.x[5] = omega_f_new; self.x[6] += self.x[8]*dt; self.x[7] += self.x[9]*dt
        
        F = np.eye(10); F[0,3]=dt; F[1,4]=dt; F[2,5]=dt; F[6,8]=dt; F[7,9]=dt
        F[3,2]=(-ax_local*sin_th-ay_local*cos_th)*dt; F[4,2]=(ax_local*cos_th-ay_local*sin_th)*dt
        self.P = F @ self.P @ F.T + self.Q * dt
        
        # --- 3. 업데이트 ---
        z = np.array([uwb_msg.point.x, uwb_msg.point.y, corrected_yaw, omega_f_new])
        pf_x, pf_y, theta_f, _, _, omega_f, pl_x, pl_y, _, _ = self.x
        cos_th, sin_th = math.cos(theta_f), math.sin(theta_f)
        rot_matrix = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
        anchor_A_world = self.x[0:2] + rot_matrix @ ANCHOR_A_LOCAL_POS
        anchor_B_world = self.x[0:2] + rot_matrix @ ANCHOR_B_LOCAL_POS
        d_a_pred = np.linalg.norm(self.x[6:8] - anchor_A_world); d_b_pred = np.linalg.norm(self.x[6:8] - anchor_B_world)
        if d_a_pred < 1e-6 or d_b_pred < 1e-6: return
        h_x = np.array([d_a_pred, d_b_pred, theta_f, omega_f])

        H = np.zeros((4, 10))
        dx_a = pl_x - anchor_A_world[0]; dy_a = pl_y - anchor_A_world[1]
        dx_b = pl_x - anchor_B_world[0]; dy_b = pl_y - anchor_B_world[1]
        H[0,0]=-dx_a/d_a_pred; H[0,1]=-dy_a/d_a_pred; H[0,6]=dx_a/d_a_pred; H[0,7]=dy_a/d_a_pred
        H[0,2]=(dx_a*(ANCHOR_A_LOCAL_POS[1]*cos_th)+dy_a*(-ANCHOR_A_LOCAL_POS[1]*-sin_th))/d_a_pred # Simplified
        H[1,0]=-dx_b/d_b_pred; H[1,1]=-dy_b/d_b_pred; H[1,6]=dx_b/d_b_pred; H[1,7]=dy_b/d_b_pred
        H[1,2]=(dx_b*(ANCHOR_B_LOCAL_POS[1]*cos_th)+dy_b*(-ANCHOR_B_LOCAL_POS[1]*-sin_th))/d_b_pred # Simplified
        H[2, 2] = 1.0; H[3, 5] = 1.0

        y_err = z - h_x; y_err[2] = (y_err[2] + np.pi) % (2 * np.pi) - np.pi
        S = H @ self.P @ H.T + self.R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y_err
            self.P = (np.eye(10) - K @ H) @ self.P
        except np.linalg.LinAlgError: pass
        
        # --- 4. 발행 ---
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
        f_pos = PointStamped(header=header, point=Point(x=self.x[0], y=self.x[1], z=0.0))
        l_pos = PointStamped(header=header, point=Point(x=self.x[6], y=self.x[7], z=0.0))
        f_vel = TwistStamped(header=header, twist=Twist(linear=Vector3(x=self.x[3], y=self.x[4], z=0.0), angular=Vector3(z=self.x[5])))
        l_vel = TwistStamped(header=header, twist=Twist(linear=Vector3(x=self.x[8], y=self.x[9], z=0.0)))
        self.follower_pos_pub.publish(f_pos); self.leader_pos_pub.publish(l_pos)
        self.follower_vel_pub.publish(f_vel); self.leader_vel_pub.publish(l_vel)

def main(args=None):
    rclpy.init(args=args)
    node = StateEstimatorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()