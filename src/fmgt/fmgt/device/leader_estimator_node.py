# 파일명: leader_estimator_node.py
"""
Leader Estimator Node (EKF-based)

UWB 거리 측정치의 비선형성을 확장 칼만 필터(EKF)를 통해 효과적으로 처리하여,
리더의 위치를 강건하게 추정하는 노드입니다.

- 좌표계 및 앵커 정의:
  - 로봇 좌표계: ROS 2 표준(REP-103)에 따라 전방 +X, 좌측 +Y.
  - 앵커 A: 로봇 전방 좌측(+Y 방향)에 위치하며, 측정 거리는 d_a 입니다.
  - 앵커 B: 로봇 전방 우측(-Y 방향)에 위치하며, 측정 거리는 d_b 입니다.

- 주요 기능:
  1. UWB 센서의 원시 거리값(d_a, d_b)을 EKF의 측정값(measurement)으로 직접 사용합니다.
  2. 상태 변수는 리더의 월드 좌표계 위치 [x, y]만으로 구성하여 안정성을 높였습니다.
  3. 측정 함수 h(x)와 이를 편미분한 자코비안 행렬 H를 통해, 거리 측정 노이즈가
     좌표계 오차에 미치는 비선형적 영향을 올바르게 모델링합니다.
  4. 삼각측량 시 발생하는 두 개의 해(전방/후방)를 모두 계산한 뒤, 현재 EKF의
     예측 상태와 더 가까운 해를 기준으로 필터를 업데이트하여 물리적으로 더 타당한
     위치를 강건하게 추정합니다.

- 구독 (Subscriptions):
  - /follower/estimated_pose (geometry_msgs/PoseStamped): 팔로워 로봇의 추정된 위치 및 자세.
     EKF의 자코비안 및 측정 함수 계산에 사용됩니다.
  - raw_uwb_distances (geometry_msgs/PointStamped): UWB 태그로부터의 거리 측정치.
     메시지의 point.x 필드에 d_a(좌측), point.y 필드에 d_b(우측) 값이 담겨 있습니다.

- 발행 (Publications):
  - /leader/raw_point (geometry_msgs/PointStamped): EKF를 통해 최종적으로 필터링된
     리더의 위치 추정치를 발행합니다. (타 노드와의 호환성을 위해 토픽명은 유지합니다.)

- 파라미터 (Parameters):
  - anchor_forward_offset (double): 로봇의 중심(base_link)에서 두 UWB 앵커를 잇는
     가상의 선까지의 전방 거리(m)입니다.
  - anchor_width (double): 좌측 앵커(A)와 우측 앵커(B) 사이의 전체 폭(m)입니다.
  - ekf_process_noise (double): EKF의 프로세스 노이즈(Q) 값입니다. 리더의 움직임에 대한
     불확실성을 나타내며, 값이 클수록 리더가 빠르게 움직일 수 있다고 가정하여
     새로운 측정값을 더 많이 신뢰하게 됩니다.
  - ekf_measurement_noise (double): EKF의 측정 노이즈(R) 값입니다. UWB 거리 측정 자체의
     불확실성을 나타내며, 값이 클수록 UWB 측정치를 덜 신뢰하고 필터의 기존 예측값을
     더 많이 신뢰하게 됩니다.
"""
import rclpy
import numpy as np
import math
import threading
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PointStamped, PoseStamped
from scipy.spatial.transform import Rotation

class LeaderEstimatorNode(Node):
    def __init__(self):
        super().__init__('leader_estimator_node')
        
        # --- 파라미터 선언 ---
        self.declare_parameter('anchor_forward_offset', 0.25)
        self.declare_parameter('anchor_width', 0.4)
        self.declare_parameter('ekf_process_noise', 0.05)
        self.declare_parameter('ekf_measurement_noise', 0.4*2)

        # --- 발행자 ---
        self.raw_point_pub = self.create_publisher(PointStamped, '/leader/raw_point', 10)
        
        # --- 내부 상태 변수 ---
        self.lock = threading.Lock()
        self.latest_follower_pose = None
        self.last_update_time = None

        # --- EKF 상태 변수 ---
        self.x = np.zeros(2) 
        self.P = np.eye(2) * 1000.0
        q_val = self.get_parameter('ekf_process_noise').value
        self.Q = np.eye(2) * q_val
        r_val = self.get_parameter('ekf_measurement_noise').value
        self.R = np.eye(2) * r_val
        self.is_initialized = False
        
        # --- 구독자 ---
        self.follower_pose_sub = self.create_subscription(
            PoseStamped, '/follower/estimated_pose', self.follower_pose_callback, 10)
        self.uwb_sub = self.create_subscription(
            PointStamped, 'raw_uwb_distances', self.uwb_update_callback, 
            QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1))
            
        self.get_logger().info("Leader Estimator (EKF-based) 시작됨.")

    def follower_pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.latest_follower_pose = msg

    def uwb_update_callback(self, uwb_msg):
        with self.lock:
            if self.latest_follower_pose is None: return

            current_time = self.get_clock().now()
            
            if self.is_initialized and self.last_update_time is not None:
                dt = (current_time - self.last_update_time).nanoseconds / 1e9
                self.P += self.Q * dt
            
            self.last_update_time = current_time

            try:
                d_a, d_b = uwb_msg.point.x, uwb_msg.point.y
                if not(d_a > 0.1 and d_b > 0.1): return

                solutions = self._calculate_trilateration_solutions(d_a, d_b)
                if solutions is None:
                    self.get_logger().warn("삼각측량 해를 찾을 수 없습니다.", throttle_duration_sec=2.0)
                    return
                
                z1, z2 = solutions

                if not self.is_initialized:
                    self.x = z1
                    self.P = np.eye(2) * 0.1
                    self.is_initialized = True
                    self.get_logger().info(f"EKF 초기화 완료. 위치: {self.x}")
                    return

                dist1 = np.linalg.norm(z1 - self.x)
                dist2 = np.linalg.norm(z2 - self.x)
                z_measurement_coords = z1 if dist1 <= dist2 else z2

                h, H = self._calculate_jacobian_and_hx(self.x)
                if h is None: return

                # 실제 측정된 거리값 [d_a, d_b]를 사용
                z_measurement_distances = np.array([d_a, d_b])
                y = z_measurement_distances - h
                
                S = H @ self.P @ H.T + self.R
                K = self.P @ H.T @ np.linalg.inv(S)
                
                self.x += K @ y
                self.P = (np.eye(2) - K @ H) @ self.P

                raw_point_msg = PointStamped()
                raw_point_msg.header.stamp = uwb_msg.header.stamp
                raw_point_msg.header.frame_id = 'world'
                raw_point_msg.point.x, raw_point_msg.point.y = self.x[0], self.x[1]
                self.raw_point_pub.publish(raw_point_msg)

            except Exception as e:
                self.get_logger().warn(f"EKF 업데이트 오류: {e}", throttle_duration_sec=2.0)
                return

    def _calculate_trilateration_solutions(self, d_a, d_b):
        anchor_forward_offset = self.get_parameter('anchor_forward_offset').value
        anchor_width = self.get_parameter('anchor_width').value
        Y_off = anchor_width / 2
        
        # #####################################################################################
        # ## FIX: A=좌측(+Y), B=우측(-Y) 정의에 따른 올바른 삼각측량 수식으로 수정
        # ## d_a^2 = px^2 + (py - Y_off)^2
        # ## d_b^2 = px^2 + (py - (-Y_off))^2 = px^2 + (py + Y_off)^2
        # ## d_b^2 - d_a^2 = (py + Y_off)^2 - (py - Y_off)^2 = 4 * py * Y_off
        # ## --> py = (d_b^2 - d_a^2) / (4 * Y_off)
        # #####################################################################################
        py_local = (d_b**2 - d_a**2) / (4 * Y_off)
        
        px_sq = d_a**2 - (py_local - Y_off)**2
        if px_sq < 0: return None
        
        px_sqrt = math.sqrt(px_sq)
        p_local_sol1 = np.array([anchor_forward_offset + px_sqrt, py_local]) # 전방
        p_local_sol2 = np.array([anchor_forward_offset - px_sqrt, py_local]) # 후방

        pf_x, pf_y = self.latest_follower_pose.pose.position.x, self.latest_follower_pose.pose.position.y
        q = self.latest_follower_pose.pose.orientation
        follower_yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]
        cos_th, sin_th = math.cos(follower_yaw), math.sin(follower_yaw)
        rot_mat = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
        p_follower = np.array([pf_x, pf_y])
        
        p_world_sol1 = p_follower + rot_mat @ p_local_sol1
        p_world_sol2 = p_follower + rot_mat @ p_local_sol2
        
        return p_world_sol1, p_world_sol2

    def _calculate_jacobian_and_hx(self, state_x):
        anchor_forward_offset = self.get_parameter('anchor_forward_offset').value
        anchor_width = self.get_parameter('anchor_width').value
        
        pf_x, pf_y = self.latest_follower_pose.pose.position.x, self.latest_follower_pose.pose.position.y
        q = self.latest_follower_pose.pose.orientation
        follower_yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]
        cos_th, sin_th = math.cos(follower_yaw), math.sin(follower_yaw)

        # #####################################################################################
        # ## FIX: A=좌측(+Y), B=우측(-Y) 정의에 따라 로컬 앵커 좌표를 명확히 함
        # #####################################################################################
        p_anchor_a_local = np.array([anchor_forward_offset, +anchor_width / 2.0]) # 앵커 A (좌측)
        p_anchor_b_local = np.array([anchor_forward_offset, -anchor_width / 2.0]) # 앵커 B (우측)

        rot_mat = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
        p_follower = np.array([pf_x, pf_y])
        p_anchor_a_world = p_follower + rot_mat @ p_anchor_a_local
        p_anchor_b_world = p_follower + rot_mat @ p_anchor_b_local

        leader_pos = state_x
        dist_a = np.linalg.norm(leader_pos - p_anchor_a_world)
        dist_b = np.linalg.norm(leader_pos - p_anchor_b_world)
        
        if dist_a < 1e-6 or dist_b < 1e-6: return None, None

        hx = np.array([dist_a, dist_b])

        H = np.array([
            [(leader_pos[0] - p_anchor_a_world[0]) / dist_a, (leader_pos[1] - p_anchor_a_world[1]) / dist_a],
            [(leader_pos[0] - p_anchor_b_world[0]) / dist_b, (leader_pos[1] - p_anchor_b_world[1]) / dist_b]
        ])
        
        return hx, H

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