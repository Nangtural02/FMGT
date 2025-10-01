# 파일명: path_generator_node.py
"""
Path Generator Node

이 노드는 리더의 '날것의' 위치 측정치(/leader/raw_point)들을 수집하여, 
Kalman Smoother를 이용해 전체적으로 가장 그럴듯한 리더의 과거 경로를 복원합니다.
스무딩은 주기적으로 실행되며, 결과물은 노이즈가 제거된 부드러운 경로입니다.

- 구독 (Subscriptions):
  - /leader/raw_point (geometry_msgs/PointStamped): Leader Estimator가 발행한 필터링되지 않은 위치

- 발행 (Publications):
  - /leader/full_trajectory (nav_msgs/Path): 스무딩 처리된 리더의 전체 경로

- 파라미터 (Parameters):
  - path_update_period_sec (double): 경로를 계산하고 발행하는 주기 (초)
  - min_points_for_path (int): 경로 생성을 시작하기 위한 최소 포인트 수
  - history_length (int): 경로 계산에 사용할 포인트의 최대 개수
  - kf_transition_covariance (double): 칼만 스무더의 상태 전이 공분산.
                                     값이 작을수록 더 부드럽고 직선적인 경로를 생성하려는 경향이 강해짐.
  - kf_observation_covariance (double): 칼만 스무더의 관측 공분산.
                                       값이 클수록 입력된 측정치를 덜 신뢰하고, 전체적인 추세선을 따라
                                       더욱 부드러운 경로를 생성함.
"""
import rclpy
import numpy as np
import math
import threading
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation
from pykalman import KalmanFilter

def yaw_to_quaternion(yaw):
    q = Rotation.from_euler('z', yaw).as_quat(); return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

class PathGeneratorNode(Node):
    def __init__(self):
        super().__init__('path_generator_node')

        # --- 파라미터 선언 ---
        self.declare_parameter('path_update_period_sec', 1.0)
        self.declare_parameter('min_points_for_path', 10)
        self.declare_parameter('history_length', 200)
        self.declare_parameter('kf_transition_covariance', 0.03*2)
        self.declare_parameter('kf_observation_covariance', 0.5)

        # --- 발행자 ---
        self.full_trajectory_pub = self.create_publisher(Path, '/leader/full_trajectory', 10)

        # --- 내부 상태 변수 ---
        self.lock = threading.Lock()
        self.point_history = []

        # --- 구독자 ---
        self.leader_point_sub = self.create_subscription(
            PointStamped, '/leader/raw_point', self.point_callback, 10)
        
        # --- 경로 생성 타이머 ---
        update_period = self.get_parameter('path_update_period_sec').value
        self.path_timer = self.create_timer(update_period, self.generate_smooth_path)
        self.get_logger().info("Path Generator Node (Refactored) 시작됨.")

    def point_callback(self, msg: PointStamped):
        """수신된 raw_point를 히스토리에 저장합니다."""
        with self.lock:
            self.point_history.append([msg.point.x, msg.point.y])
            history_length = self.get_parameter('history_length').value
            if len(self.point_history) > history_length:
                self.point_history.pop(0)

    def generate_smooth_path(self):
        """주기적으로 호출되어 히스토리를 바탕으로 부드러운 경로를 생성합니다."""
        with self.lock:
            min_points = self.get_parameter('min_points_for_path').value
            if len(self.point_history) < min_points:
                return

            header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
            
            try:
                # 파라미터에서 칼만 스무더 설정값 가져오기
                trans_cov = self.get_parameter('kf_transition_covariance').value
                obs_cov = self.get_parameter('kf_observation_covariance').value

                kf_smooth = KalmanFilter(
                    transition_matrices=np.eye(2), 
                    observation_matrices=np.eye(2),
                    transition_covariance=np.eye(2) * trans_cov, 
                    observation_covariance=np.eye(2) * obs_cov
                )
                
                measurements = list(self.point_history)
                (smoothed_means, _) = kf_smooth.smooth(measurements)
                smoothed_waypoints = [p for p in smoothed_means]

            except Exception as e:
                self.get_logger().warn(f"Kalman Smoothing 실패: {e}", throttle_duration_sec=5.0)
                smoothed_waypoints = self.point_history # 실패 시 원본 사용

            path_msg = self._create_waypoint_path_msg(header, smoothed_waypoints)
            self.full_trajectory_pub.publish(path_msg)

    def _create_waypoint_path_msg(self, header, waypoints):
        """웨이포인트 리스트로부터 Path 메시지를 생성합니다."""
        path_msg = Path(header=header)
        for i, wp in enumerate(waypoints):
            pose = PoseStamped(header=header)
            pose.pose.position.x, pose.pose.position.y = wp[0], wp[1]
            if i < len(waypoints) - 1:
                yaw = math.atan2(waypoints[i+1][1] - wp[1], waypoints[i+1][0] - wp[0])
            elif len(waypoints) > 1:
                yaw = math.atan2(wp[1] - waypoints[i-1][1], wp[0] - waypoints[i-1][0])
            else:
                yaw = 0.0
            pose.pose.orientation = yaw_to_quaternion(yaw)
            path_msg.poses.append(pose)
        return path_msg

def main(args=None):
    rclpy.init(args=args)
    node = PathGeneratorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()