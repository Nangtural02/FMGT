# fmgt4/state_estimator_node.py
import rclpy, math, time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped, TwistStamped, Twist, Quaternion, Vector3
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
import message_filters
from scipy.spatial.transform import Rotation

# 로봇 로컬 좌표계 기준 앵커 위치
ANCHOR_DISTANCE = 0.4
ANCHOR_A_LOCAL_POS = np.array([-ANCHOR_DISTANCE / 2, 0.0])
ANCHOR_B_LOCAL_POS = np.array([ ANCHOR_DISTANCE / 2, 0.0])
GRAVITY = 9.81


def quaternion_to_yaw(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

class StateEstimatorNode(Node):
    def __init__(self):
        super().__init__('state_estimator_node')
        self.follower_pos_pub = self.create_publisher(PointStamped, '/follower/estimated_position', 10)
        self.leader_pos_pub = self.create_publisher(PointStamped, '/leader/estimated_position', 10)
        self.follower_vel_pub = self.create_publisher(TwistStamped, '/follower/estimated_velocity', 10)
        self.leader_vel_pub = self.create_publisher(TwistStamped, '/leader/estimated_velocity', 10)

        # 10차원 상태 벡터 [pf_x, pf_y, θ_f, vf_x, vf_y, ω_f, pl_x, pl_y, vl_x, vl_y]
        self.x = np.zeros(10)
        self.P = np.eye(10) * 0.1
        self.P[6:, 6:] *= 1000

        # 프로세스 노이즈 Q (가속도의 불확실성)
        q_accel = 0.5**2
        q_omega_accel = 0.1**2
        q_leader_accel = 1.0**2
        self.Q_base = np.diag([0, 0, 0, q_accel, q_accel, q_omega_accel, 0, 0, q_leader_accel, q_leader_accel])

        # 측정 노이즈 R: [d_a, d_b, θ_imu, ω_imu]
        uwb_std = 0.1**2; imu_theta_std = 0.05**2; imu_omega_std = 0.1**2
        self.R = np.diag([uwb_std, uwb_std, imu_theta_std, imu_omega_std])
        
        self.last_timestamp = None
        
        uwb_sub = message_filters.Subscriber(self, PointStamped, 'raw_uwb_distances')
        imu_sub = message_filters.Subscriber(self, Imu, '/imu/data')
        self.ts = message_filters.ApproximateTimeSynchronizer([uwb_sub, imu_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.ekf_callback)
        self.get_logger().info('10-D EKF State Estimator Node (Gravity Compensated) 시작됨.')

    def ekf_callback(self, uwb_msg, imu_msg):
        current_timestamp = time.time()
        if self.last_timestamp is None: self.last_timestamp = current_timestamp; return
        dt = current_timestamp - self.last_timestamp
        self.last_timestamp = current_timestamp
        if dt <= 0 or dt > 0.5: return # 비정상적인 dt 필터링

        # --- 1. 예측 (Prediction) ---
        # IMU 데이터 추출
        q_orientation = np.array([
            imu_msg.orientation.x, imu_msg.orientation.y,
            imu_msg.orientation.z, imu_msg.orientation.w
        ])
        accel_imu = np.array([
            imu_msg.linear_acceleration.x,
            imu_msg.linear_acceleration.y,
            imu_msg.linear_acceleration.z
        ])
        
        # ★★★ 중력 보상 ★★★
        # 1. 쿼터니언으로 회전 객체 생성
        imu_rotation = Rotation.from_quat(q_orientation)
        # 2. 월드 좌표계의 중력 벡터 [0, 0, g]
        gravity_vector_world = np.array([0, 0, GRAVITY])
        # 3. 월드 기준 중력을 IMU 좌표계로 변환
        gravity_vector_imu = imu_rotation.inv().apply(gravity_vector_world)
        # 4. 측정된 가속도에서 중력 성분 제거
        accel_compensated = accel_imu - gravity_vector_imu
        ax_local, ay_local = accel_compensated[0], accel_compensated[1]

        # 이전 상태
        theta_f = self.x[2]
        # 로컬 가속도를 월드 가속도로 변환
        cos_th, sin_th = math.cos(theta_f), math.sin(theta_f)
        ax_world = ax_local * cos_th - ay_local * sin_th
        ay_world = ax_local * sin_th + ay_local * cos_th
        
        # 상태 전이 행렬 F
        F = np.eye(10); F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt; F[6, 8] = dt; F[7, 9] = dt
        # 제어 입력 행렬 B와 벡터 u
        B = np.zeros((10, 2)); B[0, 0]=0.5*dt**2; B[1, 1]=0.5*dt**2; B[3, 0]=dt; B[4, 1]=dt
        u = np.array([ax_world, ay_world])

        self.x = F @ self.x + B @ u
        self.P = F @ self.P @ F.T + self.Q_base * dt # 프로세스 노이즈
        
        # --- 2. 업데이트 (Update) ---
        d_a_meas, d_b_meas = uwb_msg.point.x, uwb_msg.point.y
        theta_meas = quaternion_to_yaw(imu_msg.orientation)
        omega_meas = imu_msg.angular_velocity.z
        z = np.array([d_a_meas, d_b_meas, theta_meas, omega_meas])

        pf_x, pf_y, theta_f, _, _, _, pl_x, pl_y, _, _ = self.x
        cos_th, sin_th = math.cos(theta_f), math.sin(theta_f)
        rot_matrix = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
        anchor_A_world = self.x[0:2] + rot_matrix @ ANCHOR_A_LOCAL_POS
        anchor_B_world = self.x[0:2] + rot_matrix @ ANCHOR_B_LOCAL_POS

        d_a_pred = np.linalg.norm(self.x[6:8] - anchor_A_world)
        d_b_pred = np.linalg.norm(self.x[6:8] - anchor_B_world)
        h_x = np.array([d_a_pred, d_b_pred, theta_f, self.x[5]])

        # 측정 야코비안 H
        H = np.zeros((4, 10))
        # 미분 계수들 (복잡하지만 수식대로 구현)
        dx_a = pl_x - anchor_A_world[0]; dy_a = pl_y - anchor_A_world[1]
        dx_b = pl_x - anchor_B_world[0]; dy_b = pl_y - anchor_B_world[1]
        # d(dist_a)/d(pf_x, pf_y, theta_f, pl_x, pl_y)
        H[0, 0] = -dx_a / d_a_pred
        H[0, 1] = -dy_a / d_a_pred
        H[0, 2] = (dx_a * (ANCHOR_A_LOCAL_POS[0]*sin_th + ANCHOR_A_LOCAL_POS[1]*cos_th) + dy_a * (-ANCHOR_A_LOCAL_POS[0]*cos_th + ANCHOR_A_LOCAL_POS[1]*sin_th)) / d_a_pred
        H[0, 6] = dx_a / d_a_pred
        H[0, 7] = dy_a / d_a_pred
        # d(dist_b)/d(pf_x, pf_y, theta_f, pl_x, pl_y)
        H[1, 0] = -dx_b / d_b_pred
        H[1, 1] = -dy_b / d_b_pred
        H[1, 2] = (dx_b * (ANCHOR_B_LOCAL_POS[0]*sin_th + ANCHOR_B_LOCAL_POS[1]*cos_th) + dy_b * (-ANCHOR_B_LOCAL_POS[0]*cos_th + ANCHOR_B_LOCAL_POS[1]*sin_th)) / d_b_pred
        H[1, 6] = dx_b / d_b_pred
        H[1, 7] = dy_b / d_b_pred
        # d(theta)/d(theta_f)
        H[2, 2] = 1.0
        # d(omega)/d(omega_f)
        H[3, 5] = 1.0

        # EKF 업데이트
        y_err = z - h_x
        y_err[2] = (y_err[2] + np.pi) % (2 * np.pi) - np.pi # 각도 오차 정규화
        S = H @ self.P @ H.T + self.R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y_err
            self.P = (np.eye(10) - K @ H) @ self.P
        except np.linalg.LinAlgError: pass
        
        # --- 3. 발행 ---
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
        
        # 팔로워 위치/속도 발행
        f_pos_msg = PointStamped(header=header, point=Point(x=self.x[0], y=self.x[1], z=0.0))
        self.follower_pos_pub.publish(f_pos_msg)
        f_linear_vel = Vector3(x=self.x[3], y=self.x[4], z=0.0)
        f_angular_vel = Vector3(x=0.0, y=0.0, z=self.x[5])
        f_vel_msg = TwistStamped(header=header, twist=Twist(linear=f_linear_vel, angular=f_angular_vel))
        self.follower_vel_pub.publish(f_vel_msg)
        
        # 리더 위치/속도 발행
        l_pos_msg = PointStamped(header=header, point=Point(x=self.x[6], y=self.x[7], z=0.0))
        self.leader_pos_pub.publish(l_pos_msg)
        l_linear_vel = Vector3(x=self.x[8], y=self.x[9], z=0.0)
        l_angular_vel = Vector3(x=0.0, y=0.0, z=0.0) # 리더의 각속도는 추정 안함
        l_vel_msg = TwistStamped(header=header, twist=Twist(linear=l_linear_vel, angular=l_angular_vel))
        self.leader_vel_pub.publish(l_vel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = StateEstimatorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__': main()