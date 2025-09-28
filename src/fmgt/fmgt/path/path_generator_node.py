# 파일명: path_generator_node.py
import rclpy
import numpy as np
import math
import threading
from rclpy.node import Node

from geometry_msgs.msg import PointStamped, PoseStamped, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation
from pykalman import KalmanFilter # ★★★ 원본 코드와 동일한 라이브러리 사용 ★★★

def yaw_to_quaternion(yaw):
    q = Rotation.from_euler('z', yaw).as_quat(); return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

class PathGeneratorNode(Node):
    def __init__(self):
        super().__init__('path_generator_node')

        # --- 파라미터 ---
        self.declare_parameter('path_update_period_sec', 1.0) # 경로 업데이트 주기 (원본과 동일)
        self.declare_parameter('min_points_for_path', 5)      # 경로 생성을 위한 최소 포인트 수 (원본과 동일)
        self.declare_parameter('history_length', 1000)         # 저장할 최대 포인트 수

        # --- 발행자 ---
        self.full_trajectory_pub = self.create_publisher(Path, '/leader/full_trajectory', 10)

        self.lock = threading.Lock()
        self.point_history = []

        # --- 구독자 ---
        self.leader_point_sub = self.create_subscription(
            PointStamped,
            '/leader/raw_point',
            self.point_callback,
            10
        )

        # --- 경로 생성 타이머 ---
        update_period = self.get_parameter('path_update_period_sec').value
        self.path_timer = self.create_timer(update_period, self.generate_smooth_path)

        self.get_logger().info("Path Generator Node 시작됨.")

    def point_callback(self, msg: PointStamped):
        with self.lock:
            # 원본 코드에서는 'Moving' 상태일 때만 추가했지만,
            # 여기서는 우선 모든 포인트를 추가하고 스무딩으로 처리하도록 단순화.
            # 필요 시, 속도를 추정하여 움직일 때만 추가하는 로직을 여기에 넣을 수 있음.
            self.point_history.append([msg.point.x, msg.point.y])
            
            # 히스토리 길이 관리
            history_length = self.get_parameter('history_length').value
            if len(self.point_history) > history_length:
                self.point_history.pop(0)

    def generate_smooth_path(self):
        with self.lock:
            min_points = self.get_parameter('min_points_for_path').value
            if len(self.point_history) < min_points:
                return

            header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
            
            try:
                # ★★★ 원본 코드의 칼만 스무더 로직을 그대로 사용 ★★★
                kf_smooth = KalmanFilter(transition_matrices=np.eye(2), observation_matrices=np.eye(2),
                                         transition_covariance=0.05*np.eye(2), observation_covariance=0.5*np.eye(2))
                
                # smooth의 입력으로 사용할 데이터 복사
                measurements = list(self.point_history)
                
                (smoothed_means, _) = kf_smooth.smooth(measurements)
                smoothed_waypoints = [p for p in smoothed_means]

            except Exception as e:
                self.get_logger().warn(f"Kalman Smoothing 실패: {e}", throttle_duration_sec=5.0)
                # 실패 시, 필터링되지 않은 원본 포인트를 그대로 사용
                smoothed_waypoints = self.point_history

            path_msg = self._create_waypoint_path_msg(header, smoothed_waypoints)
            self.full_trajectory_pub.publish(path_msg)

    def _create_waypoint_path_msg(self, header, waypoints):
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