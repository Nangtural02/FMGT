# 파일명: path_generator_node.py
"""
Path Generator Node (Path Accumulator)

이 노드는 Leader Estimator가 발행한, EKF로 필터링된 리더의 위치 추정치
(/leader/raw_point)들을 수집하여, 이를 순서대로 연결한 경로(Path)를 생성합니다.

기존의 Kalman Smoother는 제거되었으며, 이 노드는 이제 안정적인 입력 포인트를
단순히 누적하여 경로 메시지를 만드는 '경로 누적기' 역할을 합니다.

- 구독 (Subscriptions):
  - /leader/raw_point (geometry_msgs/PointStamped): EKF로 필터링된 리더의 위치 추정치

- 발행 (Publications):
  - /leader/full_trajectory (nav_msgs/Path): 누적된 리더의 전체 경로

- 파라미터 (Parameters):
  - path_update_period_sec (double): 경로를 발행하는 주기 (초)
  - min_points_for_path (int): 경로 생성을 시작하기 위한 최소 포인트 수
  - history_length (int): 경로 계산에 사용할 포인트의 최대 개수
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

def yaw_to_quaternion(yaw):
    q = Rotation.from_euler('z', yaw).as_quat(); return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

class PathGeneratorNode(Node):
    def __init__(self):
        super().__init__('path_generator_node')

        # --- 파라미터 선언 ---
        # 칼만 스무더 관련 파라미터는 더 이상 사용되지 않음
        self.declare_parameter('path_update_period_sec', 1.0)
        self.declare_parameter('min_points_for_path', 10)
        self.declare_parameter('history_length', 1000)

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
        self.path_timer = self.create_timer(update_period, self.generate_path)
        self.get_logger().info("Path Generator Node (Accumulator Version) 시작됨.")

    def point_callback(self, msg: PointStamped):
        """수신된 포인트를 히스토리에 저장합니다."""
        with self.lock:
            self.point_history.append([msg.point.x, msg.point.y])
            history_length = self.get_parameter('history_length').value
            if len(self.point_history) > history_length:
                self.point_history.pop(0)

    def generate_path(self):
        """주기적으로 호출되어 히스토리를 바탕으로 경로 메시지를 생성 및 발행합니다."""
        with self.lock:
            min_points = self.get_parameter('min_points_for_path').value
            if len(self.point_history) < min_points:
                return

            header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
            
            # Kalman Smoother 로직이 제거되고, 저장된 히스토리를 그대로 사용
            waypoints = self.point_history
            
            path_msg = self._create_waypoint_path_msg(header, waypoints)
            self.full_trajectory_pub.publish(path_msg)

    def _create_waypoint_path_msg(self, header, waypoints):
        """웨이포인트 리스트로부터 Path 메시지를 생성합니다."""
        path_msg = Path(header=header)
        # 리스트가 비어있거나 포인트가 하나뿐인 엣지 케이스 처리
        if not waypoints:
            return path_msg
            
        poses = []
        for i, wp in enumerate(waypoints):
            pose = PoseStamped(header=header)
            pose.pose.position.x, pose.pose.position.y = wp[0], wp[1]
            
            # 각 웨이포인트의 방향(yaw) 계산
            if i < len(waypoints) - 1:
                # 다음 웨이포인트를 바라보는 방향
                yaw = math.atan2(waypoints[i+1][1] - wp[1], waypoints[i+1][0] - wp[0])
            elif len(waypoints) > 1:
                # 마지막 웨이포인트는 이전 웨이포인트의 방향을 그대로 유지
                yaw = math.atan2(wp[1] - waypoints[i-1][1], wp[0] - waypoints[i-1][0])
            else:
                # 웨이포인트가 하나뿐인 경우
                yaw = 0.0
            pose.pose.orientation = yaw_to_quaternion(yaw)
            poses.append(pose)
        
        path_msg.poses = poses
        return path_msg

def main(args=None):
    rclpy.init(args=args)
    node = PathGeneratorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()