# 파일명: leader_estimator_node.py
"""
Leader Estimator Node (Stateful & Robust Coordinate Transformer)

이 노드는 UWB 거리 측정 시 발생하는 전방/후방 모호성(ambiguity) 문제를 해결하기 위해
과거 N개의 리더 위치를 기억하는 '상태 유지(Stateful)' 방식으로 동작합니다.

- 주요 기능:
  1. UWB 거리로부터 두 개의 가능한 리더 위치(전방 해, 후방 해)를 계산합니다.
  2. 과거 N개 위치의 평균(중심)을 계산하고, 이 중심과 더 가까운 해를 현재의 올바른 위치로 선택합니다.
  3. 이를 통해 로봇이 리더를 지나치거나, 경계선 부근에서 발생하는 측정 노이즈에도 강건하게
     리더의 위치를 안정적으로 추정합니다.

- 구독 (Subscriptions):
  - /follower/estimated_pose (geometry_msgs/PoseStamped): 팔로워의 추정된 위치 및 자세
  - raw_uwb_distances (geometry_msgs/PointStamped): UWB 태그로부터의 거리 (x: d_a, y: d_b)

- 발행 (Publications):
  - /leader/raw_point (geometry_msgs/PointStamped): 월드 좌표계에서 계산된 리더의 위치 측정치

- 파라미터 (Parameters):
  - anchor_forward_offset (double): 로봇 중심에서 UWB 앵커 중앙까지의 전방 거리 (m)
  - anchor_width (double): 두 UWB 앵커 사이의 폭 (m)
  - history_size_for_solution (int): 해를 선택할 때 참조할 과거 위치 데이터의 개수
"""
import rclpy
import numpy as np
import math
import threading
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PointStamped, PoseStamped
from scipy.spatial.transform import Rotation
from collections import deque

class LeaderEstimatorNode(Node):
    def __init__(self):
        super().__init__('leader_estimator_node')
        
        # --- 파라미터 선언 ---
        self.declare_parameter('anchor_forward_offset', 0.25)
        self.declare_parameter('anchor_width', 0.4)
        self.declare_parameter('history_size_for_solution', 5)

        # --- 발행자 ---
        self.raw_point_pub = self.create_publisher(PointStamped, '/leader/raw_point', 10)
        
        # --- 내부 상태 변수 ---
        self.lock = threading.Lock()
        self.latest_follower_pose = None
        
        # 과거 유효 위치를 저장하기 위한 deque
        history_size = self.get_parameter('history_size_for_solution').value
        self.valid_history = deque(maxlen=history_size)
        
        # --- 구독자 ---
        self.follower_pose_sub = self.create_subscription(
            PoseStamped, '/follower/estimated_pose', self.follower_pose_callback, 10)
        self.uwb_sub = self.create_subscription(
            PointStamped, 'raw_uwb_distances', self.uwb_update_callback, 
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1))
            
        self.get_logger().info(f"Leader Estimator (Robust Stateful) 시작됨. History Size: {history_size}")

    def follower_pose_callback(self, msg: PoseStamped):
        """팔로워의 최신 포즈를 내부 변수에 저장합니다."""
        with self.lock:
            self.latest_follower_pose = msg

    def uwb_update_callback(self, uwb_msg):
        """UWB 거리 측정치를 받아 리더의 절대 위치를 계산하고 발행합니다."""
        with self.lock:
            if self.latest_follower_pose is None:
                return
            
            # 파라미터 가져오기
            anchor_forward_offset = self.get_parameter('anchor_forward_offset').value
            anchor_width = self.get_parameter('anchor_width').value
            
            pf_x, pf_y = self.latest_follower_pose.pose.position.x, self.latest_follower_pose.pose.position.y
            q = self.latest_follower_pose.pose.orientation
            follower_yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]

            try:
                d_a, d_b = uwb_msg.point.x, uwb_msg.point.y
                if not(d_a > 0.1 and d_b > 0.1): return
                
                # 삼각측량 계산
                Y_off = anchor_width / 2
                py_local = (d_b**2 - d_a**2) / (4 * Y_off)
                px_sq = d_a**2 - (py_local - Y_off)**2
                
                if px_sq < 0: # 해가 없는 경우 (삼각형이 안 만들어짐)
                    return

                # 1. 두 개의 가능한 해(solution) 계산
                px_sqrt = math.sqrt(px_sq)
                px_local_sol1 = anchor_forward_offset + px_sqrt # 전방 해
                px_local_sol2 = anchor_forward_offset - px_sqrt # 후방 해

                # 2. 두 해를 월드 좌표계로 변환
                cos_th, sin_th = math.cos(follower_yaw), math.sin(follower_yaw)
                rot_mat = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
                p_follower = np.array([pf_x, pf_y])
                
                p_world_sol1 = p_follower + rot_mat @ np.array([px_local_sol1, py_local])
                p_world_sol2 = p_follower + rot_mat @ np.array([px_local_sol2, py_local])

                # 3. 올바른 해 선택
                chosen_pos = None
                if not self.valid_history:
                    # 초기화: 기록이 없으면 전방 해를 우선적으로 선택
                    chosen_pos = p_world_sol1
                else:
                    # 상태 유지: 과거 기록의 평균(중심)과 더 가까운 해를 선택
                    history_centroid = np.mean(self.valid_history, axis=0)
                    dist_to_sol1 = np.linalg.norm(history_centroid - p_world_sol1)
                    dist_to_sol2 = np.linalg.norm(history_centroid - p_world_sol2)
                    
                    if dist_to_sol1 <= dist_to_sol2:
                        chosen_pos = p_world_sol1
                    else:
                        chosen_pos = p_world_sol2
                
                # 4. 선택된 위치로 상태 업데이트 및 발행
                self.valid_history.append(chosen_pos)
                
                raw_point_msg = PointStamped()
                raw_point_msg.header.stamp = uwb_msg.header.stamp
                raw_point_msg.header.frame_id = 'world'
                raw_point_msg.point.x, raw_point_msg.point.y = chosen_pos[0], chosen_pos[1]
                self.raw_point_pub.publish(raw_point_msg)

            except Exception as e:
                self.get_logger().warn(f"Leader estimator 계산 오류: {e}", throttle_duration_sec=2.0)
                return

def main(args=None):
    rclpy.init(args=args)
    node = LeaderEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()