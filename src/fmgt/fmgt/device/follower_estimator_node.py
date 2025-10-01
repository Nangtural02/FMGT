# 파일명: follower_estimator_node.py
# 경로: fmgt/estimation/follower_estimator_node.py
"""
Follower Estimator Node (최종 안정화 버전)

이 노드는 로봇(팔로워)의 위치와 자세를 추정하는 역할을 합니다.
IMU와 Odometry 센서 데이터를 확장 칼만 필터(EKF)에 융합하여,
월드 좌표계(`world`)에서의 로봇 위치를 안정적으로 계산합니다.

- 주요 기능:
  1. IMU(가속도, 각속도)를 사용한 EKF 예측(Predict) 단계 수행.
  2. Odometry(위치, 방향)를 사용한 EKF 보정(Update) 단계 수행.
  3. EKF의 'world' 좌표계를 항상 (0,0,0)에서 시작하도록 하여 디버깅 편의성 확보.
  4. RDP 알고리즘으로 최적화된 로봇의 이동 경로를 주기적으로 발행하여 시각적 디버깅 지원.

- 안정성 강화 조치:
  1. [성능 최적화]: IMU 콜백에서 경로 기록 로직을 분리. 별도의 저주기 타이머(0.2초)를
     사용하여 경로를 기록함으로써, 고주파 IMU 콜백의 성능 저하를 원천적으로 방지.
  2. [Joseph Form Covariance Update]: 수치적으로 더 안정적인 Joseph 형태의 공분산 업데이트 공식을 적용.
  3. [Symmetry Enforcement]: 매 업데이트 후 공분산 행렬의 대칭성을 강제로 유지하여 오차 누적을 방지.
  4. [Validity Check]: 상태 벡터에 NaN/Inf 값이 발생하는지 검사하여 필터의 발산을 조기에 감지.
  5. [ZeroDivisionError Guard]: RDP 알고리즘에서 경로의 시작점과 끝점이 같을 때 발생하는 오류를 방지.

- 구독 (Subscriptions):
  - /imu/data (sensor_msgs/Imu): EKF의 예측 단계에 사용됩니다.
  - /odom (nav_msgs/Odometry): EKF의 보정 단계에 사용됩니다.

- 발행 (Publications):
  - /follower/estimated_pose (geometry_msgs/PoseStamped): 최종적으로 추정된 로봇의 위치와 자세.
  - /tf (tf2_msgs/TFMessage): 'world' 프레임과 'odom' 프레임 간의 변환 정보.
  - /debug/follower_estimated_path (nav_msgs/Path): 디버깅을 위한 로봇의 추정 이동 경로.

- 파라미터 (Parameters):
  - path_publish_period_sec (double): 디버깅 경로를 발행하는 주기 (초).
  - path_logging_period_sec (double): 디버깅 경로를 히스토리에 기록하는 주기 (초).
  - path_history_max_length (int): 디버깅 경로 히스토리의 최대 길이.
  - rdp_epsilon (double): 경로 단순화(RDP)를 위한 임계값 (미터).
"""
import rclpy
import numpy as np
import math
import threading
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from collections import deque

from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
import tf2_ros

from scipy.spatial.transform import Rotation

# --- 상수 및 유틸리티 함수 ---
GRAVITY = np.array([0, 0, -9.81])

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

        # 파라미터 선언
        self.declare_parameter('path_publish_period_sec', 1.0)
        self.declare_parameter('path_logging_period_sec', 0.2)
        self.declare_parameter('path_history_max_length', 1000)
        self.declare_parameter('rdp_epsilon', 0.05)
        
        # 발행자 및 TF 브로드캐스터 초기화
        self.follower_pose_pub = self.create_publisher(PoseStamped, '/follower/estimated_pose', 10)
        self.path_pub = self.create_publisher(Path, '/debug/follower_estimated_path', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.lock = threading.Lock()

        # EKF 상태 변수
        self.x_f = np.zeros(9)
        self.P_f = np.eye(9) * 0.1
        self.Q_f = np.diag([
            1e-8, 1e-8, 1e-6,
            0.05**2, 0.05**2, 0.05**2,
            0.01**2, 0.01**2, (math.radians(0.1))**2
        ])
        odom_pos_var = 0.05**2
        odom_yaw_var = (math.radians(0.05))**2
        self.R_odom = np.diag([odom_pos_var, odom_pos_var, odom_yaw_var])
        
        # 초기화 관련 변수
        self.is_initialized = False
        self.last_imu_timestamp = None
        self.last_odom_pose = None
        self.initial_odom_pos = None
        self.initial_odom_yaw = 0.0

        # 경로 히스토리를 deque로 관리
        max_len = self.get_parameter('path_history_max_length').value
        self.estimated_path_history = deque(maxlen=max_len)

        # 구독자 초기화
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_predict_callback, sensor_qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_update_callback, sensor_qos)
        
        # 타이머 초기화
        path_pub_period = self.get_parameter('path_publish_period_sec').value
        self.path_publish_timer = self.create_timer(path_pub_period, self.publish_estimated_path)
        
        path_log_period = self.get_parameter('path_logging_period_sec').value
        self.path_log_timer = self.create_timer(path_log_period, self.log_path_callback)
        
        self.get_logger().info(f"Follower Estimator Node 시작됨. Path history max length: {max_len}")
        
    def imu_predict_callback(self, imu_msg):
        with self.lock:
            if not self.is_initialized:
                return
            
            current_timestamp = rclpy.time.Time.from_msg(imu_msg.header.stamp).nanoseconds / 1e9
            if self.last_imu_timestamp is None:
                self.last_imu_timestamp = current_timestamp
                return
            
            dt = current_timestamp - self.last_imu_timestamp
            
            if not (0 < dt < 0.5):
                self.get_logger().warn(
                    f"비정상적인 IMU dt 값 감지: {dt:.4f} 초. "
                    f"Current: {current_timestamp:.4f}, Last: {self.last_imu_timestamp:.4f}. "
                    "예측 단계를 건너뜁니다."
                )
                self.last_imu_timestamp = current_timestamp
                return
            
            self.last_imu_timestamp = current_timestamp
            
            q_orientation = np.array([imu_msg.orientation.x, imu_msg.orientation.y, imu_msg.orientation.z, imu_msg.orientation.w])
            accel_original = np.array([imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z])
            
            imu_rotation = Rotation.from_quat(q_orientation)
            pure_accel = accel_original - imu_rotation.inv().apply(GRAVITY)
            
            b_ax, b_ay, b_wz = self.x_f[6], self.x_f[7], self.x_f[8]
            ax_local = pure_accel[0] - b_ax
            ay_local = pure_accel[1] - b_ay
            
            theta_f = self.x_f[2]
            cos_th, sin_th = math.cos(theta_f), math.sin(theta_f)
            ax_world = ax_local * cos_th - ay_local * sin_th
            ay_world = ax_local * sin_th + ay_local * cos_th
            
            self.x_f[0] += self.x_f[3] * dt
            self.x_f[1] += self.x_f[4] * dt
            self.x_f[2] = normalize_angle(self.x_f[2] + self.x_f[5] * dt)
            self.x_f[3] += ax_world * dt
            self.x_f[4] += ay_world * dt
            
            F_f = np.eye(9)
            F_f[0, 3] = dt; F_f[1, 4] = dt; F_f[2, 5] = dt
            F_f[3, 2] = (-ax_local * sin_th - ay_local * cos_th) * dt
            F_f[4, 2] = ( ax_local * cos_th - ay_local * sin_th) * dt
            F_f[3, 6] = -cos_th * dt; F_f[3, 7] = sin_th * dt
            F_f[4, 6] = -sin_th * dt; F_f[4, 7] = -cos_th * dt
            
            self.P_f = F_f @ self.P_f @ F_f.T + self.Q_f * dt
            self.P_f = (self.P_f + self.P_f.T) / 2.0

            f_pose = PoseStamped()
            f_pose.header = imu_msg.header
            f_pose.header.frame_id = 'world'
            f_pose.pose.position.x = self.x_f[0]
            f_pose.pose.position.y = self.x_f[1]
            f_pose.pose.position.z = 0.0
            f_pose.pose.orientation = yaw_to_quaternion(self.x_f[2])
            self.follower_pose_pub.publish(f_pose)

            if self.last_odom_pose:
                T_world_base = np.eye(4)
                T_world_base[:3, :3] = Rotation.from_euler('z', self.x_f[2]).as_matrix()
                T_world_base[:2, 3] = self.x_f[:2]
                
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
        with self.lock:
            self.last_odom_pose = odom_msg.pose.pose
            
            current_q_odom = odom_msg.pose.pose.orientation
            current_odom_yaw = Rotation.from_quat([current_q_odom.x, current_q_odom.y, current_q_odom.z, current_q_odom.w]).as_euler('zyx')[0]
            current_odom_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])

            if not self.is_initialized:
                self.initial_odom_pos = current_odom_pos
                self.initial_odom_yaw = current_odom_yaw
                self.is_initialized = True
                self.get_logger().info(f"Follower EKF 초기화. Odom 기준점 설정: pos=({self.initial_odom_pos[0]:.2f}, {self.initial_odom_pos[1]:.2f}), yaw={math.degrees(self.initial_odom_yaw):.2f} deg. EKF는 (0,0)에서 시작.")
                return
            
            delta_pos_world = current_odom_pos - self.initial_odom_pos
            cos_init = math.cos(-self.initial_odom_yaw)
            sin_init = math.sin(-self.initial_odom_yaw)
            relative_x = delta_pos_world[0] * cos_init - delta_pos_world[1] * sin_init
            relative_y = delta_pos_world[0] * sin_init + delta_pos_world[1] * cos_init
            relative_yaw = normalize_angle(current_odom_yaw - self.initial_odom_yaw)
            
            z = np.array([relative_x, relative_y, relative_yaw])
            h_x = self.x_f[[0, 1, 2]]
            
            H_f = np.zeros((3, 9)); H_f[0, 0]=1; H_f[1, 1]=1; H_f[2, 2]=1
            
            y_err = z - h_x
            y_err[2] = normalize_angle(y_err[2])
            
            S = H_f @ self.P_f @ H_f.T + self.R_odom
            K = self.P_f @ H_f.T @ np.linalg.inv(S)
            
            self.x_f += K @ y_err
            self.x_f[2] = normalize_angle(self.x_f[2])
            
            I = np.eye(9)
            self.P_f = (I - K @ H_f) @ self.P_f @ (I - K @ H_f).T + K @ self.R_odom @ K.T
            
            if not np.all(np.isfinite(self.x_f)):
                self.get_logger().error("EKF 상태 벡터가 발산했습니다 (NaN or Inf). 노드를 재시작해야 합니다.")
                return

    def log_path_callback(self):
        """주기적으로 호출되어 EKF의 현재 위치를 경로 히스토리에 기록합니다."""
        with self.lock:
            if not self.is_initialized:
                return
            
            # 현재 추정된 위치를 경로 히스토리에 추가
            current_position = (self.x_f[0], self.x_f[1])
            self.estimated_path_history.append(current_position)

    def publish_estimated_path(self):
        """주기적으로 추정된 경로(Path)를 RDP 알고리즘으로 단순화하여 발행합니다."""
        with self.lock:
            if len(self.estimated_path_history) < 2:
                return
            
            path_points = list(self.estimated_path_history)
            epsilon = self.get_parameter('rdp_epsilon').value
            simplified_points = self._douglas_peucker(path_points, epsilon)
            
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'world'
            
            for point in simplified_points:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = point[0]
                pose.pose.position.y = point[1]
                pose.pose.orientation.w = 1.0
                path_msg.poses.append(pose)
            
            self.path_pub.publish(path_msg)
    
    def _douglas_peucker(self, points, epsilon):
        """RDP 알고리즘을 이용해 포인트 리스트를 단순화합니다."""
        if len(points) < 3:
            return points
            
        dmax, index = 0.0, 0
        p1, p_end = np.array(points[0]), np.array(points[-1])
        
        if np.linalg.norm(p_end - p1) < 1e-6:
            return [points[0]]

        for i in range(1, len(points) - 1):
            # 분모가 0이 되는 것을 방지하기 위해 np.linalg.norm(p_end - p1)을 미리 계산
            denominator = np.linalg.norm(p_end - p1)
            if denominator < 1e-9: # 매우 작은 값일 경우 건너뛰기
                continue
            d = np.linalg.norm(np.cross(p_end - p1, p1 - np.array(points[i]))) / denominator
            if d > dmax:
                index, dmax = i, d
        
        if dmax > epsilon:
            rec1 = self._douglas_peucker(points[:index + 1], epsilon)
            rec2 = self._douglas_peucker(points[index:], epsilon)
            return rec1[:-1] + rec2
        else:
            return [points[0], points[-1]]


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