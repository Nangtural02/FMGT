# 파일명: follower_estimator_node.py
# 경로: fmgt/estimation/follower_estimator_node.py
"""
Follower Estimator Node (수정 버전)

이 노드는 로봇(팔로워)의 위치와 자세를 추정하는 역할을 합니다.
IMU와 Odometry 센서 데이터를 확장 칼만 필터(EKF)에 융합하여,
월드 좌표계(`world`)에서의 로봇 위치를 안정적으로 계산합니다.

- **핵심 수정 사항**:
  1. EKF 초기화: 첫 Odometry 메시지(IMU의 절대 각도 포함)를 기준으로 EKF 상태 변수 [x, y, theta]를 즉시 초기화합니다. (시작 각도 부정확성 문제 해결)
  2. Odometry 측정값: Odometry의 상대 변위가 아닌, **절대 위치/자세**를 EKF 보정 단계의 측정값으로 사용합니다.
  3. IMU 예측: Angular Velocity (v_theta)는 더 이상 IMU 측정값으로 직접 덮어쓰지 않고, 일정한 속도를 유지한다고 예측하도록 수정하여 필터의 견고성을 높였습니다.

- 구독 (Subscriptions):
  - /imu/data (sensor_msgs/Imu): EKF의 예측(predict) 단계에 사용됩니다. 가속도 및 각속도 데이터.
  - /odom (nav_msgs/Odometry): EKF의 보정(update) 단계에 사용됩니다. Odometry의 절대 위치/자세.

- 발행 (Publications):
  - /follower/estimated_pose (geometry_msgs/PoseStamped): 최종적으로 추정된 로봇의 위치와 자세.
  - /tf (tf2_msgs/TFMessage): 'world' 프레임과 'odom' 프레임 간의 변환(transform) 정보.
"""
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
GRAVITY = np.array([0, 0, -9.81]) # 중력 가속도 벡터

def normalize_angle(angle):
    """각도를 -pi 에서 +pi 사이로 정규화합니다."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def yaw_to_quaternion(yaw):
    """Yaw 각도(z축 회전)를 Quaternion 메시지 형태로 변환합니다."""
    q = Rotation.from_euler('z', yaw).as_quat()
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


class FollowerEstimatorNode(Node):
    def __init__(self):
        super().__init__('follower_estimator_node')
        
        # --- 발행자 및 TF 브로드캐스터 초기화 ---
        self.follower_pose_pub = self.create_publisher(PoseStamped, '/follower/estimated_pose', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.lock = threading.Lock() # 멀티스레드 환경에서의 데이터 동기화를 위한 Lock

        # --- EKF 상태 변수 및 공분산 행렬 초기화 ---
        # 상태 변수 (9x1): [x, y, theta, vx, vy, v_theta, b_ax, b_ay, b_wz]
        # 위치(x,y), 자세(theta), 속도(vx,vy), 각속도(v_theta), IMU 바이어스(b_ax,b_ay,b_wz)
        self.x_f = np.zeros(9)
        self.P_f = np.eye(9) * 0.1 # 상태 불확실성 공분산 행렬

        # 프로세스 노이즈 공분산 행렬 (Q_f): 모델의 불확실성을 나타냄
        self.Q_f = np.diag([
            1e-8, 1e-8, 1e-6,      # 위치/자세 노이즈
            0.05**2, 0.05**2, 0.05**2, # 속도 노이즈
            0.01**2, 0.01**2, (math.radians(0.1))**2 # 바이어스 노이즈
        ])
        
        # 측정 노이즈 공분산 행렬 (R_odom): Odometry 센서의 노이즈 (절대 위치/자세)
        odom_pos_var = 0.05**2
        odom_yaw_var = (math.radians(0.05))**2
        self.R_odom = np.diag([odom_pos_var, odom_pos_var, odom_yaw_var])
        
        # --- 초기화 관련 변수 ---
        self.is_initialized = False
        self.last_imu_timestamp = None
        # [삭제] initial_odom_pos, initial_odom_yaw는 EKF 상태를 직접 초기화하도록 변경하여 필요 없어짐.
        self.last_odom_pose = None # world->odom TF 계산에 사용

        # --- 구독자 초기화 ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_predict_callback, sensor_qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_update_callback, sensor_qos)
        
        self.get_logger().info("Follower Estimator Node 시작됨.")
        
    def imu_predict_callback(self, imu_msg):
        """IMU 데이터를 사용하여 EKF의 예측(predict) 단계를 수행합니다."""
        with self.lock:
            if not self.is_initialized:
                return
            
            # 시간 간격(dt) 계산
            current_timestamp = rclpy.time.Time.from_msg(imu_msg.header.stamp).nanoseconds / 1e9
            if self.last_imu_timestamp is None:
                self.last_imu_timestamp = current_timestamp
                return
            dt = current_timestamp - self.last_imu_timestamp
            self.last_imu_timestamp = current_timestamp
            if not (0 < dt < 0.5): # 비정상적인 dt 값은 무시
                return
            
            # IMU 데이터 추출 및 중력 보상
            q_orientation = np.array([imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w])
            accel_original = np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z])
            omega_original = np.array([imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z])
            
            imu_rotation = Rotation.from_quat(q_orientation)
            pure_accel = accel_original - imu_rotation.inv().apply(GRAVITY)
            
            # 추정된 바이어스를 사용하여 IMU 측정값 보정
            b_ax, b_ay, b_wz = self.x_f[6], self.x_f[7], self.x_f[8]
            ax_local = pure_accel[0] - b_ax
            ay_local = pure_accel[1] - b_ay
            omega_z_local = omega_original[2] - b_wz
            
            # 로봇 좌표계의 가속도를 월드 좌표계로 변환
            theta_f = self.x_f[2]
            cos_th, sin_th = math.cos(theta_f), math.sin(theta_f)
            ax_world = ax_local * cos_th - ay_local * sin_th
            ay_world = ax_local * sin_th + ay_local * cos_th
            
            # --- 상태 예측 (State Prediction) ---
            # 운동학 모델을 기반으로 다음 상태를 예측
            self.x_f[0] += self.x_f[3] * dt  # x += vx*dt
            self.x_f[1] += self.x_f[4] * dt  # y += vy*dt
            self.x_f[2] = normalize_angle(self.x_f[2] + self.x_f[5] * dt) # theta += v_theta*dt
            self.x_f[3] += ax_world * dt     # vx += ax*dt
            self.x_f[4] += ay_world * dt     # vy += ay*dt
            # [수정] v_theta 예측: Constant Velocity Model을 따름 (IMU 측정값으로 직접 덮어쓰지 않음)
            # self.x_f[5] = omega_z_local # <-- 기존의 문제적인 코드 제거
            # self.x_f[5]는 예측 단계에서 변하지 않고, Odometry update 시 보정되거나 
            # 다음 텀에서 IMU 바이어스를 통해 간접적으로 영향을 받음.
            
            # 상태 전이 행렬(F_f, Jacobian) 계산
            F_f = np.eye(9)
            F_f[0, 3] = dt; F_f[1, 4] = dt; F_f[2, 5] = dt
            F_f[3, 2] = (-ax_local * sin_th - ay_local * cos_th) * dt
            F_f[4, 2] = ( ax_local * cos_th - ay_local * sin_th) * dt
            # [수정] v_theta가 omega_z_local로 덮어쓰이지 않으므로, 아래 바이어스 영향은
            # v_theta에 대한 변화가 아닌 바이어스 자체의 예측에 영향을 미치도록 수정될 수 있으나,
            # 현재 코드의 F_f 구조를 유지하며, F_f[5, 8] 항을 제거하거나 0으로 둠
            # (바이어스 예측은 EKF의 고급 주제이므로, 여기서는 v_theta에 대한 직접적인 영향 항만 제거)
            # F_f[5, 8] = -dt # <-- 원래 코드에 있던 항을 제거하고 0으로 둠 (F_f는 np.eye(9)로 시작하므로 F_f[5, 8]=0)
            
            F_f[3, 6] = -cos_th * dt; F_f[3, 7] = sin_th * dt
            F_f[4, 6] = -sin_th * dt; F_f[4, 7] = -cos_th * dt
            
            # 공분산 예측: P = F * P * F^T + Q
            self.P_f = F_f @ self.P_f @ F_f.T + self.Q_f * dt

            # --- 예측된 위치 발행 ---
            f_pose = PoseStamped()
            f_pose.header = imu_msg.header
            f_pose.header.frame_id = 'world'
            f_pose.pose.position.x = self.x_f[0]
            f_pose.pose.position.y = self.x_f[1]
            f_pose.pose.position.z = 0.0 # 2D 환경이므로 z는 0으로 고정
            f_pose.pose.orientation = yaw_to_quaternion(self.x_f[2])
            self.follower_pose_pub.publish(f_pose)

            # --- world -> odom TF 발행 ---
            if self.last_odom_pose:
                T_world_base = np.eye(4)
                T_world_base[:3, :3] = Rotation.from_euler('z', self.x_f[2]).as_matrix()
                T_world_base[:2, 3] = self.x_f[:2]
                
                # odom_msg.pose.pose가 이미 last_odom_pose에 저장됨
                T_odom_base = np.eye(4)
                q_o = self.last_odom_pose.orientation
                T_odom_base[:3, :3] = Rotation.from_quat([q_o.x, q_o.y, q_o.z, q_o.w]).as_matrix()
                T_odom_base[0:2, 3] = [self.last_odom_pose.position.x, self.last_odom_pose.position.y]
                
                T_world_odom = T_world_base @ np.linalg.inv(T_odom_base)
                
                t = TransformStamped()
                t.header.stamp = imu_msg.header.stamp
                t.header.frame_id = 'world'
                t.child_frame_id = 'odom'
                t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = T_world_odom[0:3, 3]
                q_tf = Rotation.from_matrix(T_world_odom[:3, :3]).as_quat()
                t.transform.rotation = Quaternion(x=q_tf[0], y=q_tf[1], z=q_tf[2], w=q_tf[3])
                self.tf_broadcaster.sendTransform(t)

    def odom_update_callback(self, odom_msg):
        """Odometry 데이터를 사용하여 EKF의 보정(update) 단계를 수행합니다."""
        with self.lock:
            # TF 발행을 위해 최신 Odometry 포즈 저장
            self.last_odom_pose = odom_msg.pose.pose 
            
            # 현재 Odometry에서 절대 위치/방향 추출
            q_odom = odom_msg.pose.pose.orientation
            odom_yaw = Rotation.from_quat([q_odom.x, q_odom.y, q_odom.z, q_odom.w]).as_euler('zyx')[0]
            odom_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])

            if not self.is_initialized:
                # [수정] EKF 상태 변수(x_f)를 첫 측정값(Odometry의 절대 포즈)으로 즉시 초기화
                self.x_f[0] = odom_pos[0]
                self.x_f[1] = odom_pos[1]
                self.x_f[2] = odom_yaw
                
                # [수정] 공분산 P_f도 초기화 불확실성을 반영하도록 조정 (선택적)
                self.P_f[[0, 1, 2], [0, 1, 2]] = self.R_odom.diagonal() * 1.0 # 위치/각도 불확실성 설정

                self.is_initialized = True
                self.get_logger().info(f"Follower EKF 초기화 성공! Initial Pose: x={self.x_f[0]:.2f}, y={self.x_f[1]:.2f}, yaw={math.degrees(self.x_f[2]):.2f} deg")
                return
            
            # [수정] Odometry의 절대 위치와 각도를 측정값으로 사용 (z)
            z = np.array([odom_pos[0], odom_pos[1], odom_yaw])
            
            # 예측된 측정값 (h(x)) - EKF 상태의 위치/방향
            h_x = self.x_f[[0, 1, 2]]
            
            # 측정 모델의 Jacobian (H_f)
            H_f = np.zeros((3, 9)); H_f[0, 0]=1; H_f[1, 1]=1; H_f[2, 2]=1
            
            # 측정 오차 (Innovation)
            y_err = z - h_x
            y_err[2] = normalize_angle(y_err[2]) # 각도 오차는 정규화 (필수)
            
            # 칼만 게인(K) 계산
            S = H_f @ self.P_f @ H_f.T + self.R_odom
            K = self.P_f @ H_f.T @ np.linalg.inv(S)
            
            # 상태 및 공분산 보정
            self.x_f += K @ y_err
            self.x_f[2] = normalize_angle(self.x_f[2]) # 보정된 각도도 정규화
            self.P_f = (np.eye(9) - K @ H_f) @ self.P_f

def main(args=None):
    """노드를 초기화하고 실행합니다."""
    rclpy.init(args=args)
    node = FollowerEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()