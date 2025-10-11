# 파일명: leader_estimator_4_anchor_node.py
"""
Leader Estimator Node (4-Anchor Non-linear EKF)

4개의 UWB 앵커를 사용하여 리더의 위치를 강건하게 추정하는 비선형 확장 칼만 필터(EKF)
노드입니다. 4개의 원시 거리값을 측정치로 직접 사용하고, 4x2 자코비안 행렬을 통해
각 측정치의 기여도를 정교하게 모델링하여 최고의 안정성과 정확도를 달성합니다.

- 핵심 로직:
  1. 앵커 월드 좌표 계산: 로봇의 현재 자세를 기준으로 4개 앵커의 월드 좌표를 실시간으로 계산.
  2. EKF 예측: 정지 모델을 사용하여 리더의 다음 위치를 예측.
  3. EKF 업데이트:
     - 측정값(z): 4개의 원시 UWB 거리값 [d_a, d_b, d_c, d_d].
     - 측정 함수(h(x)): 현재 예측 위치로부터 4개 앵커까지의 예상 거리를 계산.
     - 자코비안(H): 측정 함수를 상태 변수(x, y)로 편미분한 4x2 행렬.
     - 업데이트: 표준 EKF 공식을 사용하여 상태(위치)를 최종 보정.

- 구독 (Subscriptions):
  - /follower/estimated_pose (geometry_msgs/PoseStamped): 팔로워의 현재 위치 및 자세.
  - /uwb/distances_4_anchor (geometry_msgs/PoseStamped): 4개 앵커의 거리 측정치.

- 발행 (Publications):
  - /leader/estimated_point_4_anchor (geometry_msgs/PointStamped): 최종 필터링된 리더 위치.
  - /debug/anchor_marker (visualization_msgs/Marker): RViz 시각화를 위한 4개 앵커의 월드 좌표.

- 파라미터 (Parameters):
  - anchor_pos.a, .b, .c, .d (list): 각 앵커의 로봇 중심 기준 로컬 좌표 [x, y].
  - ekf_process_noise (double): EKF의 프로세스 노이즈(Q) 값.
  - ekf_measurement_noise (double): EKF의 측정 노이즈(R) 값.
"""
import rclpy
import numpy as np
import math
import threading
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped, Point
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation

class LeaderEstimator4AnchorNode(Node):
    def __init__(self):
        super().__init__('leader_estimator_4_anchor_node')
        
        # --- 파라미터 선언 ---
        self.declare_parameter('anchor_pos.a', [0.25, 0.2])   # 전방 좌측
        self.declare_parameter('anchor_pos.b', [0.25, -0.2])  # 전방 우측
        self.declare_parameter('anchor_pos.c', [-0.25, 0.2])  # 후방 좌측
        self.declare_parameter('anchor_pos.d', [-0.25, -0.2]) # 후방 우측
        self.declare_parameter('ekf_process_noise', 0.3**2)
        self.declare_parameter('ekf_measurement_noise', 0.3**2)

        # --- 발행자 ---
        self.point_pub = self.create_publisher(PointStamped, '/leader/raw_point', 10)
        self.anchor_marker_pub = self.create_publisher(Marker, '/debug/anchor_marker', 10)
        
        # --- 내부 상태 변수 ---
        self.lock = threading.Lock()
        self.latest_follower_pose = None
        self.last_update_time = None

        # --- EKF 상태 변수 ---
        self.x = np.zeros(2) 
        self.P = np.eye(2) * 1000.0
        self.Q = np.eye(2) * self.get_parameter('ekf_process_noise').value
        # R은 이제 4x4 행렬
        r_val = self.get_parameter('ekf_measurement_noise').value
        self.R = np.eye(4) * r_val
        self.is_initialized = False

        # --- 구독자 ---
        self.follower_pose_sub = self.create_subscription(
            PoseStamped, '/follower/estimated_pose', self.follower_pose_callback, 10)
        self.uwb_sub = self.create_subscription(
            PoseStamped, '/uwb/distances_4_anchor', self.uwb_callback, 10)
            
        self.get_logger().info("Leader Estimator (4-Anchor EKF) 시작됨.")

    def follower_pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.latest_follower_pose = msg

    def uwb_callback(self, msg: PoseStamped):
        with self.lock:
            if self.latest_follower_pose is None: return

            current_time = self.get_clock().now()
            if self.is_initialized and self.last_update_time is not None:
                dt = (current_time - self.last_update_time).nanoseconds / 1e9
                self.P += self.Q * dt
            self.last_update_time = current_time

            # --- 1. 입력 데이터 추출 ---
            d_a = msg.pose.position.x
            d_b = msg.pose.position.y
            d_c = msg.pose.position.z
            d_d = msg.pose.orientation.x
            if not (d_a > 0.1 and d_b > 0.1 and d_c > 0.1 and d_d > 0.1): return
            
            # 4개의 거리 측정값 벡터 z
            z = np.array([d_a, d_b, d_c, d_d])

            # --- 2. 앵커 월드 좌표 계산 ---
            p_f = np.array([self.latest_follower_pose.pose.position.x, self.latest_follower_pose.pose.position.y])
            q = self.latest_follower_pose.pose.orientation
            yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]
            rot_mat = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])

            p_a_local = np.array(self.get_parameter('anchor_pos.a').value)
            p_b_local = np.array(self.get_parameter('anchor_pos.b').value)
            p_c_local = np.array(self.get_parameter('anchor_pos.c').value)
            p_d_local = np.array(self.get_parameter('anchor_pos.d').value)
            
            anchor_world_coords = {
                'a': p_f + rot_mat @ p_a_local,
                'b': p_f + rot_mat @ p_b_local,
                'c': p_f + rot_mat @ p_c_local,
                'd': p_f + rot_mat @ p_d_local,
            }
            self.publish_anchor_marker(anchor_world_coords, msg.header)

            # --- 3. EKF 업데이트 ---
            if not self.is_initialized:
                # 초기화: 최소 제곱법으로 초기 위치 추정
                self.x = self.multilateration_least_squares(anchor_world_coords, z)
                if self.x is None: return
                self.P = np.eye(2) * 0.1
                self.is_initialized = True
                self.get_logger().info(f"EKF (4-Anchor) 초기화 완료.")
                return
            
            # 측정 함수 h(x)와 자코비안 H 계산
            h, H = self.calculate_hx_and_jacobian(self.x, anchor_world_coords)
            if h is None: return

            y = z - h
            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T @ np.linalg.inv(S)
            
            self.x += K @ y
            self.P = (np.eye(2) - K @ H) @ self.P

            # --- 4. 결과 발행 ---
            point_msg = PointStamped()
            point_msg.header = msg.header
            point_msg.header.frame_id = 'world'
            point_msg.point.x, point_msg.point.y = self.x[0], self.x[1]
            self.point_pub.publish(point_msg)

    def calculate_hx_and_jacobian(self, state_x, anchor_coords):
        p_a = anchor_coords['a']; p_b = anchor_coords['b']
        p_c = anchor_coords['c']; p_d = anchor_coords['d']
        
        dist_a = np.linalg.norm(state_x - p_a)
        dist_b = np.linalg.norm(state_x - p_b)
        dist_c = np.linalg.norm(state_x - p_c)
        dist_d = np.linalg.norm(state_x - p_d)

        if any(d < 1e-6 for d in [dist_a, dist_b, dist_c, dist_d]): return None, None

        hx = np.array([dist_a, dist_b, dist_c, dist_d])
        
        H = np.array([
            [(state_x[0] - p_a[0]) / dist_a, (state_x[1] - p_a[1]) / dist_a],
            [(state_x[0] - p_b[0]) / dist_b, (state_x[1] - p_b[1]) / dist_b],
            [(state_x[0] - p_c[0]) / dist_c, (state_x[1] - p_c[1]) / dist_c],
            [(state_x[0] - p_d[0]) / dist_d, (state_x[1] - p_d[1]) / dist_d]
        ])
        return hx, H

    def multilateration_least_squares(self, anchor_coords, distances):
        # 초기화 시 사용되는 최소 제곱법 다변측위
        try:
            anchors = list(anchor_coords.values())
            A = []
            b = []
            for i in range(len(anchors) - 1):
                A.append(2 * (anchors[i+1] - anchors[0]))
                b.append(distances[0]**2 - distances[i+1]**2 + anchors[i+1][0]**2 - anchors[0][0]**2 + anchors[i+1][1]**2 - anchors[0][1]**2)
            A = np.array(A)
            b = np.array(b)
            # (A.T * A)^-1 * A.T * b
            result = np.linalg.inv(A.T @ A) @ A.T @ b
            return result
        except np.linalg.LinAlgError:
            return None

    def publish_anchor_marker(self, anchor_coords, header):
        marker = Marker()
        marker.header = header
        marker.header.frame_id = "world"
        marker.ns = "anchor_positions"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05  # 선 두께
        marker.color.a = 0.8; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0 # 초록색

        # 사각형을 그리기 위해 A-B-D-C-A 순서로 점 추가
        points = [anchor_coords['a'], anchor_coords['b'], anchor_coords['d'], anchor_coords['c'], anchor_coords['a']]
        for p in points:
            marker.points.append(Point(x=p[0], y=p[1], z=0.1))
        
        self.anchor_marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = LeaderEstimator4AnchorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()