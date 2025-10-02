# 파일명: path_generator_node.py
"""
Short-term Path Generator Node (Adaptive Bézier Curve Smoothing)

이 노드는 EKF로 필터링된 리더의 위치 스트림을 입력받아, 로봇이 추종할 짧고
매우 부드러운 '꼬리' 경로를 생성합니다. 2단계 파이프라인을 통해 안정성과
부드러움을 모두 확보합니다.

1. 경로 단순화 (Douglas-Peucker): EKF 출력의 미세한 떨림을 제거하고, 경로의
   핵심적인 형태를 나타내는 주요 꼭짓점들만 추출합니다.
2. 적응형 베지어 곡선 생성 (Adaptive Bézier Curve Generation): 단순화된 꼭짓점의
   개수에 따라 가장 적합한 베지어 곡선을 자동으로 선택하여 생성합니다.
   - 4개 이상: 3차 베지어 곡선 (부드러운 S자 곡선 표현에 유리)
   - 3개: 2차 베지어 곡선 (단일 코너링 상황에 최적화)
   - 2개: 직선 경로 (기본)

- 구독 (Subscriptions):
  - /leader/raw_point (geometry_msgs/PointStamped): EKF 기반 Leader Estimator의 출력.

- 발행 (Publications):
  - /controller/short_term_path (nav_msgs/Path): 제어기가 추종할 짧고 부드러운 경로.

- 파라미터 (Parameters):
  - history_length (int): 경로 생성에 사용할 최근 포인트의 개수.
  - min_points_for_path (int): 경로 생성을 시작하기 위한 최소 포인트 수.
  - douglas_peucker_epsilon (double): 경로 단순화의 강도를 결정하는 임계값(m).
  - bezier_resolution (int): 생성될 베지어 곡선 경로의 포인트 개수.
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
from collections import deque

def yaw_to_quaternion(yaw):
    q = Rotation.from_euler('z', yaw).as_quat(); return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

class PathGeneratorNode(Node):
    def __init__(self):
        super().__init__('path_generator_node')
        self.declare_parameter('history_length', 20)
        self.declare_parameter('min_points_for_path', 10)
        self.declare_parameter('douglas_peucker_epsilon', 0.3)
        self.declare_parameter('bezier_resolution', 50)

        self.path_pub = self.create_publisher(Path, '/controller/short_term_path', 10)
        self.lock = threading.Lock()
        
        history_size = self.get_parameter('history_length').value
        self.point_history = deque(maxlen=history_size)

        self.point_sub = self.create_subscription(PointStamped, '/leader/raw_point', self.point_callback, 10)
        self.path_timer = self.create_timer(0.2, self.generate_path_callback)
        self.get_logger().info("Short-term Path Generator (Adaptive Bézier) 시작됨.")

    def point_callback(self, msg: PointStamped):
        with self.lock:
            self.point_history.append(np.array([msg.point.x, msg.point.y]))

    def generate_path_callback(self):
        with self.lock:
            min_points = self.get_parameter('min_points_for_path').value
            if len(self.point_history) < min_points: return

            points = list(self.point_history)
            
            # 1단계: Douglas-Peucker로 경로 단순화
            simplified_points = self.douglas_peucker(points, self.get_parameter('douglas_peucker_epsilon').value)
            
            final_waypoints = []
            resolution = self.get_parameter('bezier_resolution').value
            t_points = np.linspace(0, 1, resolution)

            try:
                if len(simplified_points) >= 4:
                    # --- 2단계 (A): 제어점이 4개 이상이면, 3차 베지어 곡선 생성 ---
                    p0 = simplified_points[0]
                    p1 = simplified_points[1]
                    p2 = simplified_points[-2]
                    p3 = simplified_points[-1]
                    
                    final_waypoints = [
                        (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3
                        for t in t_points
                    ]
                
                elif len(simplified_points) == 3:
                    # --- 2단계 (B): 제어점이 3개이면, 2차 베지어 곡선 생성 ---
                    p0 = simplified_points[0]
                    p1 = simplified_points[1]
                    p2 = simplified_points[2]

                    final_waypoints = [
                        (1-t)**2*p0 + 2*(1-t)*t*p1 + t**2*p2
                        for t in t_points
                    ]

                elif len(simplified_points) >= 2:
                    # --- 2단계 (C): 제어점이 2개이면, 안정적인 직선 경로 사용 ---
                    final_waypoints = self.resample_path(simplified_points)
                
                else:
                    return # 경로 생성 불가

            except Exception as e:
                self.get_logger().warn(f"경로 생성 중 오류 발생: {e}", throttle_duration_sec=2.0)
                return

            if not final_waypoints: return

            header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
            path_msg = self._create_waypoint_path_msg(header, final_waypoints)
            self.path_pub.publish(path_msg)

    def douglas_peucker(self, points, epsilon):
        if len(points) < 3: return points
        dmax, index = 0.0, 0
        p1, p_end = points[0], points[-1]
        for i in range(1, len(points) - 1):
            d = np.linalg.norm(np.cross(p_end - p1, p1 - points[i])) / np.linalg.norm(p_end - p1)
            if d > dmax: index, dmax = i, d
        
        if dmax > epsilon:
            rec1 = self.douglas_peucker(points[:index + 1], epsilon)
            rec2 = self.douglas_peucker(points[index:], epsilon)
            return rec1[:-1] + rec2
        else:
            return [p1, p_end]

    def resample_path(self, points: list) -> list:
        spacing = 0.05
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

    def _create_waypoint_path_msg(self, header, waypoints):
        path_msg = Path(header=header)
        poses = []
        for i, wp in enumerate(waypoints):
            pose = PoseStamped(header=header)
            pose.pose.position.x, pose.pose.position.y = wp[0], wp[1]
            if i < len(waypoints) - 1:
                yaw = math.atan2(waypoints[i+1][1] - wp[1], waypoints[i+1][0] - wp[0])
            elif len(waypoints) > 1:
                yaw = math.atan2(wp[1] - waypoints[i-1][1], wp[0] - waypoints[i-1][0])
            else: yaw = 0.0
            pose.pose.orientation = yaw_to_quaternion(yaw)
            poses.append(pose)
        path_msg.poses = poses
        return path_msg

def main(args=None):
    rclpy.init(args=args); node = PathGeneratorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__': main()