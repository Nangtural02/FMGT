# 파일명: fmgt4/state_estimator_node.py
import rclpy, math
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Point, PointStamped, TwistStamped, Twist, Quaternion, Vector3, TransformStamped, PoseStamped
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import tf2_ros
import threading

# --- 상수 정의 ---
GRAVITY = np.array([0, 0, -9.81])
ANCHOR_FORWARD_OFFSET = 0.25
ANCHOR_WIDTH = 0.4
ANCHOR_A_LOCAL_POS = np.array([ANCHOR_FORWARD_OFFSET, ANCHOR_WIDTH / 2])
ANCHOR_B_LOCAL_POS = np.array([ANCHOR_FORWARD_OFFSET, -ANCHOR_WIDTH / 2])

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class DualEkfStateEstimatorNode(Node):
    def __init__(self):
        super().__init__('state_estimator_node')
        # --- 발행자 ---
        self.follower_pos_pub = self.create_publisher(PointStamped, '/follower/estimated_position', 10)
        self.leader_pos_pub = self.create_publisher(PointStamped, '/leader/estimated_position', 10)
        self.follower_vel_pub = self.create_publisher(TwistStamped, '/follower/estimated_velocity', 10)
        self.follower_pose_pub = self.create_publisher(PoseStamped, '/follower/estimated_pose', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.lock = threading.Lock()

        # --- 필터 1: 팔로워 EKF 변수 (변경 없음) ---
        self.x_f = np.zeros(9); self.P_f = np.eye(9) * 0.1; self.P_f[6:9, 6:9] = 0.01**2
        self.Q_f = np.diag([1e-8, 1e-8, 1e-6, 0.05**2, 0.05**2, 0.05**2, 0.01**2, 0.01**2, (math.radians(0.1))**2])
        odom_pos_var = 0.1**2; odom_yaw_var = (math.radians(2.0))**2; odom_omega_var = 0.1**2
        self.R_odom = np.diag([odom_pos_var, odom_pos_var, odom_yaw_var, odom_omega_var])

        # ★★★ 변경점: 리더 KF (상태에서 속도 제거) ★★★
        # 상태: [x, y] (2차원)
        self.x_l = np.zeros(2)
        self.P_l = np.eye(2) * 1000
        # Q_l: 리더 정지 모델의 불확실성 (리더가 얼마나 빨리 움직일 수 있는가)
        leader_process_noise = 0.2**2
        self.Q_l = np.diag([leader_process_noise, leader_process_noise])
        uwb_var = 0.3**2
        self.R_uwb = np.diag([uwb_var, uwb_var])

        # --- 시스템 변수 ---
        self.last_imu_timestamp = None; self.is_initialized = False
        self.initial_odom_pos = None; self.initial_odom_yaw = 0.0
        self.last_odom_pose = None; self.last_odom_timestamp = None; self.last_odom_yaw = 0.0
        
        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        
        self.uwb_sub = self.create_subscription(
            PointStamped, 'raw_uwb_distances', self.uwb_update_callback, sensor_qos_profile)
        self.imu_sub = self.create_subscription(
            Imu, '/imu/data', self.imu_predict_callback, sensor_qos_profile)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_update_callback, sensor_qos_profile)
        
        self.get_logger().info('Dual EKF State Estimator (Stable Relative Motion) 시작됨.')

    def imu_predict_callback(self, imu_msg):
        with self.lock:
            if not self.is_initialized: return
            
            current_timestamp = imu_msg.header.stamp.sec + imu_msg.header.stamp.nanosec * 1e-9
            dt = current_timestamp - self.last_imu_timestamp
            self.last_imu_timestamp = current_timestamp
            if not (0 < dt < 0.5): return
            
            # --- 1. 팔로워 EKF 예측 ---
            x_f_before = self.x_f.copy() # 예측 전 팔로워 상태 저장

            q_orientation = np.array([imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w])
            accel_original = np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z])
            omega_original = np.array([imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z])
            imu_rotation = Rotation.from_quat(q_orientation); gravity_in_imu_frame = imu_rotation.inv().apply(GRAVITY); pure_accel = accel_original - gravity_in_imu_frame
            
            # <<< 기존 코드 (STELLA N1 기준: Y-forward, X-right)
            # ax_body_ros = pure_accel[1]; ay_body_ros = -pure_accel[0]
            # >>>
            
            # <<< 수정된 코드 (STELLA N5 기준: X-forward, Y-left)
            ax_body_ros = pure_accel[0]
            ay_body_ros = pure_accel[1]
            # >>>

            b_ax, b_ay, b_wz = self.x_f[6], self.x_f[7], self.x_f[8]
            ax_local = ax_body_ros - b_ax; ay_local = ay_body_ros - b_ay; omega_z_local = omega_original[2] - b_wz
            theta_f = self.x_f[2]; cos_th, sin_th = math.cos(theta_f), math.sin(theta_f)
            ax_world = ax_local * cos_th - ay_local * sin_th; ay_world = ax_local * sin_th + ay_local * cos_th
            self.x_f[0]+=self.x_f[3]*dt; self.x_f[1]+=self.x_f[4]*dt; self.x_f[2]=normalize_angle(self.x_f[2]+self.x_f[5]*dt)
            self.x_f[3]+=ax_world*dt; self.x_f[4]+=ay_world*dt; self.x_f[5]=omega_z_local
            F_f=np.eye(9); F_f[0,3]=dt; F_f[1,4]=dt; F_f[2,5]=dt
            F_f[3,2]=(-ax_local*sin_th-ay_local*cos_th)*dt; F_f[4,2]=(ax_local*cos_th-ay_local*sin_th)*dt
            F_f[3,6]=-cos_th*dt; F_f[3,7]=sin_th*dt; F_f[4,6]=-sin_th*dt; F_f[4,7]=-cos_th*dt; F_f[5,8]=-dt
            self.P_f = F_f @ self.P_f @ F_f.T + self.Q_f * dt

            # ★★★ 변경점: 리더 KF 예측 (상대 운동을 올바르게 반영) ★★★
            if np.any(self.x_l != 0): # 리더가 초기화된 후에만 예측 수행
                # 1. 팔로워의 움직임 계산
                delta_theta_f = normalize_angle(self.x_f[2] - x_f_before[2])
                
                # 2. 리더의 이전 위치를 '이전' 팔로워 위치 기준으로 변환
                p_l_relative_before = self.x_l - x_f_before[0:2]
                
                # 3. 팔로워의 회전 적용 (올바른 회전 중심 사용)
                rot_mat = np.array([[math.cos(delta_theta_f), -math.sin(delta_theta_f)],
                                    [math.sin(delta_theta_f),  math.cos(delta_theta_f)]])
                p_l_relative_after = rot_mat @ p_l_relative_before
                
                # 4. 팔로워의 '이전' 위치에 다시 더하여 새로운 월드 좌표 계산
                self.x_l = p_l_relative_after + x_f_before[0:2]
            
            # 불확실성만 증가 (Random Walk)
            self.P_l = self.P_l + self.Q_l * dt

            # --- 3. 발행 ---
            header = Header(stamp=imu_msg.header.stamp, frame_id='world')
            f_pos = PointStamped(header=header, point=Point(x=self.x_f[0], y=self.x_f[1], z=0.0))
            l_pos = PointStamped(header=header, point=Point(x=self.x_l[0], y=self.x_l[1], z=0.0))
            f_vel = TwistStamped(header=header, twist=Twist(linear=Vector3(x=self.x_f[3], y=self.x_f[4], z=0.0), angular=Vector3(z=self.x_f[5])))
            self.follower_pos_pub.publish(f_pos); self.leader_pos_pub.publish(l_pos); self.follower_vel_pub.publish(f_vel)
            f_pose = PoseStamped(header=header)
            f_pose.pose.position.x=self.x_f[0]; f_pose.pose.position.y=self.x_f[1]
            q = Rotation.from_euler('z', self.x_f[2]).as_quat()
            f_pose.pose.orientation.x=q[0]; f_pose.pose.orientation.y=q[1]; f_pose.pose.orientation.z=q[2]; f_pose.pose.orientation.w=q[3]
            self.follower_pose_pub.publish(f_pose)

            if self.last_odom_pose:
                T_world_base=np.eye(4); T_world_base[:2,3]=self.x_f[:2]; R_world_base=Rotation.from_euler('z',self.x_f[2]).as_matrix(); T_world_base[:3,:3]=R_world_base
                T_odom_base=np.eye(4); T_odom_base[0,3]=self.last_odom_pose.position.x; T_odom_base[1,3]=self.last_odom_pose.position.y
                q_o=self.last_odom_pose.orientation; R_odom_base=Rotation.from_quat([q_o.x,q_o.y,q_o.z,q_o.w]).as_matrix(); T_odom_base[:3,:3]=R_odom_base
                T_world_odom = T_world_base @ np.linalg.inv(T_odom_base)
                t = TransformStamped(); t.header.stamp=imu_msg.header.stamp; t.header.frame_id='world'; t.child_frame_id='odom'
                t.transform.translation.x=T_world_odom[0,3]; t.transform.translation.y=T_world_odom[1,3]; t.transform.translation.z=T_world_odom[2,3]
                q_tf=Rotation.from_matrix(T_world_odom[:3,:3]).as_quat()
                t.transform.rotation.x=q_tf[0]; t.transform.rotation.y=q_tf[1]; t.transform.rotation.z=q_tf[2]; t.transform.rotation.w=q_tf[3]
                self.tf_broadcaster.sendTransform(t)

            log_msg = (
                f"Follower(x,y,th): {self.x_f[0]:.2f}, {self.x_f[1]:.2f}, {math.degrees(self.x_f[2]):.1f} | "
                f"Leader(x,y): {self.x_l[0]:.2f}, {self.x_l[1]:.2f}"
            )
            self.get_logger().info(log_msg, throttle_duration_sec=0.1)

    def odom_update_callback(self, odom_msg):
        with self.lock:
            current_odom_timestamp = odom_msg.header.stamp.sec + odom_msg.header.stamp.nanosec * 1e-9
            self.last_odom_pose = odom_msg.pose.pose
            q_odom = odom_msg.pose.pose.orientation
            odom_yaw = Rotation.from_quat([q_odom.x, q_odom.y, q_odom.z, q_odom.w]).as_euler('zyx', degrees=False)[0]
            odom_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])
            if not self.is_initialized:
                self.initial_odom_pos = odom_pos; self.initial_odom_yaw = odom_yaw
                self.last_imu_timestamp = current_odom_timestamp
                self.last_odom_timestamp = current_odom_timestamp
                self.last_odom_yaw = odom_yaw; self.x_f[0:3] = [0.0, 0.0, 0.0]; self.is_initialized = True
                self.get_logger().info(f"Follower EKF 초기화 성공! (Odometry 기준)")
                return
            
            dt_odom = current_odom_timestamp - self.last_odom_timestamp
            if dt_odom < 1e-6: return
            measured_omega = normalize_angle(odom_yaw - self.last_odom_yaw) / dt_odom
            self.last_odom_timestamp = current_odom_timestamp; self.last_odom_yaw = odom_yaw
            relative_pos = odom_pos - self.initial_odom_pos; relative_yaw = normalize_angle(odom_yaw - self.initial_odom_yaw)
            z = np.array([relative_pos[0], relative_pos[1], relative_yaw, measured_omega])
            h_x = self.x_f[[0, 1, 2, 5]]
            H_f = np.zeros((4, 9)); H_f[0,0]=1.0; H_f[1,1]=1.0; H_f[2,2]=1.0; H_f[3,5]=1.0
            y_err = z - h_x; y_err[2] = normalize_angle(y_err[2])
            S = H_f @ self.P_f @ H_f.T + self.R_odom
            try:
                K = self.P_f @ H_f.T @ np.linalg.inv(S)
                self.x_f = self.x_f + K @ y_err; self.x_f[2] = normalize_angle(self.x_f[2])
                self.P_f = (np.eye(9) - K @ H_f) @ self.P_f
            except np.linalg.LinAlgError: pass

    def uwb_update_callback(self, uwb_msg):
        with self.lock:
            if not self.is_initialized: return
            
            try:
                d_a, d_b = uwb_msg.point.x, uwb_msg.point.y
                if not (d_a > 0 and d_b > 0): return
                pf_x, pf_y, theta_f = self.x_f[0], self.x_f[1], self.x_f[2]
                X_off = ANCHOR_A_LOCAL_POS[0]; Y_off = ANCHOR_A_LOCAL_POS[1]
                py_local = (d_b**2 - d_a**2) / (4 * Y_off)
                px_sq = d_a**2 - (py_local - Y_off)**2
                if px_sq < 0: return
                px_local = X_off + math.sqrt(px_sq)
                cos_th, sin_th = math.cos(theta_f), math.sin(theta_f)
                z_pos_x = pf_x + (px_local * cos_th - py_local * sin_th)
                z_pos_y = pf_y + (px_local * sin_th + py_local * cos_th)
                z_pos = np.array([z_pos_x, z_pos_y])
            except Exception: return
            
            if np.all(self.x_l == 0):
                self.x_l = z_pos
                self.get_logger().info(f"리더 초기 위치 설정 완료: ({self.x_l[0]:.2f}, {self.x_l[1]:.2f})")

            z = z_pos
            h_x = self.x_l
            H_l = np.eye(2)
            y_err = z - h_x
            S = H_l @ self.P_l @ H_l.T + self.R_uwb
            try:
                K = self.P_l @ H_l.T @ np.linalg.inv(S)
                self.x_l = self.x_l + K @ y_err
                self.P_l = (np.eye(2) - K @ H_l) @ self.P_l
            except np.linalg.LinAlgError: pass

def main(args=None):
    rclpy.init(args=args)
    node = DualEkfStateEstimatorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()