# 파일명: path_postprocessor_node.py
"""
Path Post-processor Node

이 노드는 Path Generator가 생성한 경로를 로봇이 실제로 부드럽게 주행할 수 있도록
후처리하는 역할을 담당합니다. 3단계 파이프라인을 통해 경로를 가공합니다.

- 처리 파이프라인:
  1. 경로 단순화 (Douglas-Peucker): 노이즈로 인한 불필요한 굴곡을 제거하여 경로의 '형태'를 평탄화합니다.
  2. 경로 리샘플링: 단순화된 경로를 일정한 간격의 웨이포인트로 재구성하여 경로의 '밀도'를 균일화합니다.
  3. 경로 자르기: 리더와의 충돌을 방지하기 위해, 최종 경로의 끝을 'target_distance'만큼 잘라냅니다.

- 구독 (Subscriptions):
  - /leader/full_trajectory (nav_msgs/Path): Path Generator가 생성한 원본 경로

- 발행 (Publications):
  - /robot/goal_trajectory (nav_msgs/Path): 후처리된 최종 목표 경로

- 파라미터 (Parameters):
  - douglas_peucker_epsilon (double): 경로 단순화의 강도를 결정하는 임계값 (m). 클수록 경로가 더 직선화됨.
  - waypoint_spacing (double): 리샘플링 시 웨이포인트 사이의 목표 간격 (m).
  - target_distance (double): 리더와 유지할 목표 안전 거리 (m).
  - yaw_smoothing_window_size (int): Yaw 값 스무딩을 위한 이동 평균 필터의 윈도우 크기.
  - debug_heartbeat_period (double): 디버깅 상태를 로깅하는 주기 (초).
"""
import rclpy
import numpy as np
import math
import threading
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion, Pose
from nav_msgs.msg import Path
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation

def yaw_to_quaternion(yaw):
    """geometry_msgs/Quaternion 메시지를 Yaw 각도(라디안)로 변환합니다."""
    q = Rotation.from_euler('z', yaw).as_quat(); return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

class PathPostprocessorNode(Node):
    def __init__(self):
        super().__init__('path_postprocessor_node')

        self.declare_parameter('douglas_peucker_epsilon', 0.1)
        self.declare_parameter('waypoint_spacing', 0.05)
        self.declare_parameter('target_distance', 1.0)
        self.declare_parameter('yaw_smoothing_window_size', 5)
        self.declare_parameter('debug_heartbeat_period', 1.0)

        self.processed_path_pub = self.create_publisher(Path, '/robot/goal_trajectory', 10)
        self.lock = threading.Lock()
        
        self.debug_stats = {
            'last_input_size': 0, 'last_simplified_size': 0, 'last_resampled_size': 0,
            'last_truncated_size': 0, 'paths_processed_count': 0, 'last_processed_time': 0.0
        }

        self.path_sub = self.create_subscription(Path, '/leader/full_trajectory', self.path_callback, 10)
        heartbeat_period = self.get_parameter('debug_heartbeat_period').value
        self.heartbeat_timer = self.create_timer(heartbeat_period, self.heartbeat_callback)

        self.get_logger().info("Path Post-processor Node (Final Version) 시작됨.")

    def path_callback(self, raw_path_msg: Path):
        with self.lock:
            self.debug_stats['last_processed_time'] = self.get_clock().now().seconds_nanoseconds()[0]
            num_raw_poses = len(raw_path_msg.poses)
            self.debug_stats['last_input_size'] = num_raw_poses
            if num_raw_poses < 2: return

            points = [np.array([p.pose.position.x, p.pose.position.y]) for p in raw_path_msg.poses]

            epsilon = self.get_parameter('douglas_peucker_epsilon').value
            simplified_points = self.douglas_peucker(points, epsilon)
            self.debug_stats['last_simplified_size'] = len(simplified_points)
            if len(simplified_points) < 2: return

            resampled_points = self.resample_path(simplified_points)
            self.debug_stats['last_resampled_size'] = len(resampled_points)
            if len(resampled_points) < 2: return
            
            target_dist = self.get_parameter('target_distance').value
            truncated_points = self.truncate_path(resampled_points, target_dist)
            self.debug_stats['last_truncated_size'] = len(truncated_points)
            if len(truncated_points) < 2: return

            processed_poses = self.calculate_and_smooth_yaw(truncated_points)
            if not processed_poses: return

            processed_path_msg = Path(
                header=Header(stamp=self.get_clock().now().to_msg(), frame_id=raw_path_msg.header.frame_id),
                poses=processed_poses
            )
            self.processed_path_pub.publish(processed_path_msg)
            self.debug_stats['paths_processed_count'] += 1

    def douglas_peucker(self, points, epsilon):
        if len(points) < 3: return points
        dmax, index = 0.0, 0
        p1, p_end = points[0], points[-1]
        for i in range(1, len(points) - 1):
            # 직선(p1, p_end)과 점(points[i]) 사이의 거리를 계산합니다.
            # np.cross를 이용한 벡터 외적의 크기는 두 벡터가 만드는 평행사변형의 넓이와 같습니다.
            # 이 넓이를 밑변(p_end - p1)의 길이로 나누면 높이, 즉 점과 직선 사이의 거리가 됩니다.
            d = np.linalg.norm(np.cross(p_end - p1, p1 - points[i])) / np.linalg.norm(p_end - p1)
            if d > dmax: index, dmax = i, d
        
        if dmax > epsilon:
            # 임계값보다 먼 점이 있으면, 그 점을 기준으로 경로를 둘로 나누어 재귀적으로 처리
            rec1 = self.douglas_peucker(points[:index + 1], epsilon)
            rec2 = self.douglas_peucker(points[index:], epsilon)
            # 재귀 결과에서 중복되는 중간점을 제거하고 합침
            return rec1[:-1] + rec2
        else:
            # 모든 중간점이 임계값보다 가까우면, 중간점들을 모두 제거하고 시작점과 끝점만 반환
            return [p1, p_end]

    def resample_path(self, points: list) -> list:
        spacing = self.get_parameter('waypoint_spacing').value
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        if cumulative_distances[-1] < spacing: return points
        
        resampled_points, idx = [], 0
        for d in np.arange(0, cumulative_distances[-1], spacing):
            while idx < len(cumulative_distances) - 1 and d > cumulative_distances[idx + 1]: idx += 1
            p1, p2 = points[idx], points[idx + 1]
            dist_in_seg = d - cumulative_distances[idx]
            total_seg_dist = distances[idx]
            ratio = dist_in_seg / total_seg_dist if total_seg_dist > 1e-6 else 0.0
            resampled_points.append(p1 + ratio * (p2 - p1))
            
        if np.linalg.norm(resampled_points[-1] - points[-1]) > 1e-3:
             resampled_points.append(points[-1])
        return resampled_points

    def truncate_path(self, points: list, target_distance: float) -> list:
        """
        [최종 버그 수정] 경로의 끝에서부터 target_distance를 유지하도록 경로를 잘라냅니다.
        경로의 총 길이가 target_distance보다 짧으면, 시작점 하나만 반환하여 로봇을 정지시킵니다.
        """
        if len(points) < 2:
            return points

        # 경로의 끝에서부터 역방향으로 탐색하며 누적 거리를 계산합니다.
        total_dist_from_end = 0.0
        for i in range(len(points) - 2, -1, -1):
            p1 = points[i]
            p2 = points[i+1]
            total_dist_from_end += np.linalg.norm(p2 - p1)
            
            # 누적 거리가 target_distance를 넘으면, 현재 지점이 새로운 끝점이 됩니다.
            if total_dist_from_end >= target_distance:
                return points[:i + 1] # i 지점까지의 경로를 잘라서 반환
        
        # for 루프가 끝까지 실행되었다면, 경로의 전체 길이가 target_distance보다 짧다는 의미입니다.
        # 이 경우, 로봇이 움직이지 않도록 시작점 하나만 남긴 경로를 반환합니다.
        return [points[0]]

    def calculate_and_smooth_yaw(self, points: list) -> list:
        if len(points) < 2: return []
        
        raw_yaws = [math.atan2(p2[1]-p1[1], p2[0]-p1[0]) for p1, p2 in zip(points[:-1], points[1:])]
        raw_yaws.append(raw_yaws[-1])
        
        smoothed_yaws, window_size = [], self.get_parameter('yaw_smoothing_window_size').value
        for i in range(len(raw_yaws)):
            start = max(0, i - window_size // 2); end = min(len(raw_yaws), i + window_size // 2 + 1)
            avg_cos = np.mean([math.cos(y) for y in raw_yaws[start:end]])
            avg_sin = np.mean([math.sin(y) for y in raw_yaws[start:end]])
            smoothed_yaws.append(math.atan2(avg_sin, avg_cos))
            
        processed_poses = []
        for i, p in enumerate(points):
            current_pose = Pose()
            current_pose.position.x, current_pose.position.y = p[0], p[1]
            current_pose.orientation = yaw_to_quaternion(smoothed_yaws[i])
            
            stamped_pose = PoseStamped()
            stamped_pose.header.frame_id = "world"
            stamped_pose.pose = current_pose
            processed_poses.append(stamped_pose)
        return processed_poses

    def heartbeat_callback(self):
        stats = self.debug_stats
        now = self.get_clock().now().seconds_nanoseconds()[0]
        time_since_last = now - stats['last_processed_time'] if stats['last_processed_time'] > 0 else -1.0
        log_msg = (
            f"[Heartbeat] Paths processed: {stats['paths_processed_count']} | "
            f"Last Pipeline (In->DP->Re->Tr): "
            f"{stats['last_input_size']}->{stats['last_simplified_size']}->"
            f"{stats['last_resampled_size']}->{stats['last_truncated_size']} | "
            f"Since last: {time_since_last:.1f}s"
        )
        self.get_logger().info(log_msg)

def main(args=None):
    rclpy.init(args=args); node = PathPostprocessorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()