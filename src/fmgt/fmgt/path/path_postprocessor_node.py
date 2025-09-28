# 파일명: path_postprocessor_node.py (버그 수정 버전)
"""
Path Post-processor Node

(Docstring은 이전과 동일)
"""
import rclpy, numpy as np, math, threading
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation

def yaw_to_quaternion(yaw):
    q = Rotation.from_euler('z', yaw).as_quat(); return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class PathPostprocessorNode(Node):
    def __init__(self):
        super().__init__('path_postprocessor_node')

        self.declare_parameter('waypoint_spacing', 0.05) # 5cm 간격으로 리샘플링
        self.declare_parameter('yaw_smoothing_window_size', 5)

        # ★★★ 발행 토픽 이름을 Controller가 구독하는 이름으로 수정 ★★★
        self.processed_path_pub = self.create_publisher(Path, '/robot/goal_trajectory', 10)
        self.lock = threading.Lock()

        self.path_sub = self.create_subscription(Path, '/leader/full_trajectory', self.path_callback, 10)
        self.get_logger().info("Path Post-processor Node (Bug Fixed) 시작됨.")

    def path_callback(self, raw_path_msg: Path):
        with self.lock:
            if len(raw_path_msg.poses) < 2:
                return

            # 1. 경로 리샘플링
            resampled_points = self.resample_path_corrected(raw_path_msg.poses)
            if not resampled_points or len(resampled_points) < 2:
                self.get_logger().warn("리샘플링 후 포인트가 너무 적어 경로를 발행하지 않습니다.", throttle_duration_sec=2.0)
                return

            # 2. Yaw 계산 및 스무딩
            processed_poses = self.calculate_and_smooth_yaw(resampled_points)
            if not processed_poses:
                return

            # 3. 새로운 Path 메시지 생성 및 발행
            processed_path_msg = Path()
            processed_path_msg.header.stamp = self.get_clock().now().to_msg()
            processed_path_msg.header.frame_id = raw_path_msg.header.frame_id
            processed_path_msg.poses = processed_poses
            self.processed_path_pub.publish(processed_path_msg)

    def resample_path_corrected(self, poses: list) -> list:
        """
        ★ 버그가 수정되고 더 강건해진 리샘플링 함수 ★
        경로를 따라가며 일정한 간격으로 포인트를 다시 샘플링합니다.
        """
        points = [np.array([p.pose.position.x, p.pose.position.y]) for p in poses]
        spacing = self.get_parameter('waypoint_spacing').value
        
        # 경로의 각 세그먼트 사이의 거리를 미리 계산
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        # 경로 시작부터 각 점까지의 누적 거리
        cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
        
        if cumulative_distances[-1] < spacing:
            return [] # 전체 경로가 샘플링 간격보다 짧으면 아무것도 하지 않음

        resampled_points = []
        # 샘플링할 지점들의 누적 거리를 생성 (예: 0, 0.05, 0.10, 0.15...)
        sampling_distances = np.arange(0, cumulative_distances[-1], spacing)

        current_segment_idx = 0
        for d_sample in sampling_distances:
            # 현재 샘플링 거리가 어느 세그먼트에 속하는지 찾음
            while current_segment_idx < len(cumulative_distances) - 1 and d_sample > cumulative_distances[current_segment_idx + 1]:
                current_segment_idx += 1
            
            # 선형 보간(Linear Interpolation)을 사용하여 점의 위치 계산
            p1 = points[current_segment_idx]
            p2 = points[current_segment_idx + 1]
            
            # 현재 세그먼트 내에서의 비율 계산
            dist_in_segment = d_sample - cumulative_distances[current_segment_idx]
            total_segment_dist = distances[current_segment_idx]
            
            # 분모가 0이 되는 것을 방지
            if total_segment_dist < 1e-6:
                ratio = 0.0
            else:
                ratio = dist_in_segment / total_segment_dist
            
            # 보간된 점의 좌표
            new_point = p1 + ratio * (p2 - p1)
            resampled_points.append(new_point)
            
        # 마지막 점은 항상 포함
        if np.linalg.norm(resampled_points[-1] - points[-1]) > 1e-3:
             resampled_points.append(points[-1])
             
        return resampled_points

    def calculate_and_smooth_yaw(self, points: list) -> list:
        if len(points) < 2:
            return []

        raw_yaws = []
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i+1]
            raw_yaws.append(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
        raw_yaws.append(raw_yaws[-1])

        smoothed_yaws = []
        window_size = self.get_parameter('yaw_smoothing_window_size').value
        for i in range(len(raw_yaws)):
            start = max(0, i - window_size // 2)
            end = min(len(raw_yaws), i + window_size // 2 + 1)
            
            avg_cos = np.mean([math.cos(y) for y in raw_yaws[start:end]])
            avg_sin = np.mean([math.sin(y) for y in raw_yaws[start:end]])
            smoothed_yaws.append(math.atan2(avg_sin, avg_cos))

        processed_poses = []
        for i, p in enumerate(points):
            pose = PoseStamped()
            pose.header.frame_id = "world"
            pose.pose.position.x, pose.pose.position.y = p[0], p[1]
            pose.pose.orientation = yaw_to_quaternion(smoothed_yaws[i])
            processed_poses.append(pose)
            
        return processed_poses

def main(args=None):
    rclpy.init(args=args); node = PathPostprocessorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__': main()