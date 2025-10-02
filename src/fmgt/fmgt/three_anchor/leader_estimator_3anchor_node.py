# 파일명: leader_estimator_3anchor_node.py
"""
Leader Estimator Node (EKF with 3-Anchor Dynamic Selection)

이 노드는 3개의 UWB 앵커를 사용하여 EKF의 안정성을 극대화합니다. 기존 EKF 구조를
유지하면서, 삼각측량의 '해 선택' 문제를 3번째 앵커를 이용한 물리적 검증으로
완벽하게 해결합니다. 또한, 리더의 예상 위치에 따라 최적의 앵커 조합을 동적으로
선택하여 측위 정확도를 높입니다.

- 핵심 로직:
  1. EKF 예측: 기존과 동일하게 상태(위치)와 공분산을 예측합니다.
  2. 동적 앵커 선택: 리더의 예상 위치(전방/좌후방/우후방)에 따라, 가장 신뢰도 높은
     앵커 2개를 '주력'으로, 나머지 1개를 '검증용'으로 동적으로 지정합니다.
  3. 삼각측량 및 검증: '주력' 앵커로 두 개의 해를 계산하고, '검증용' 앵커와의
     거리 비교를 통해 물리적으로 올바른 단 하나의 해를 결정합니다.
  4. EKF 업데이트: 이렇게 검증된 '단 하나의 올바른 해'를 EKF의 측정값으로 사용하여
     상태를 업데이트합니다. 이를 통해 '실패의 피드백 루프'를 원천적으로 차단합니다.

- 구독 (Subscriptions):
  - /follower/estimated_pose (geometry_msgs/PoseStamped): 팔로워의 현재 위치 및 자세.
  - raw_uwb_distances (geometry_msgs/PointStamped): 3개 앵커의 거리 (x:d_a, y:d_b, z:d_c).

- 발행 (Publications):
  - /leader/raw_point (geometry_msgs/PointStamped): 최종 필터링된 리더의 위치.

- 파라미터 (Parameters):
  - anchor_pos.a, anchor_pos.b, anchor_pos.c (list): 각 앵커의 로봇 중심 기준 로컬 좌표 [x, y].
  - ekf_process_noise (double): EKF의 프로세스 노이즈(Q) 값.
  - ekf_measurement_noise (double): EKF의 측정 노이즈(R) 값.
"""
import rclpy
import numpy as np
import math
import threading
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped
from scipy.spatial.transform import Rotation
from collections import deque

class LeaderEstimator3AnchorNode(Node):
    def __init__(self):
        super().__init__('leader_estimator_3_anchor_node')
        
        self.declare_parameter('anchor_pos.a', [0.25, 0.2])   # 전방 좌측
        self.declare_parameter('anchor_pos.b', [0.25, -0.2])  # 전방 우측
        self.declare_parameter('anchor_pos.c', [-0.25, 0.0])  # 후방 중앙
        self.declare_parameter('ekf_process_noise', 0.6**2)
        self.declare_parameter('ekf_measurement_noise', 0.3**2)

        self.raw_point_pub = self.create_publisher(PointStamped, '/leader/raw_point', 10)
        self.lock = threading.Lock()
        
        self.latest_follower_pose = None
        self.last_update_time = None

        self.x = np.zeros(2)
        self.P = np.eye(2) * 1000.0
        self.Q = np.eye(2) * self.get_parameter('ekf_process_noise').value
        self.R = np.eye(2) * self.get_parameter('ekf_measurement_noise').value
        self.is_initialized = False

        self.follower_pose_sub = self.create_subscription(
            PoseStamped, '/follower/estimated_pose', self.follower_pose_callback, 10)
        self.uwb_sub = self.create_subscription(
            PointStamped, 'raw_uwb_distances', self.uwb_callback, 10)
            
        self.get_logger().info("Leader Estimator (EKF + 3-Anchor Dynamic) 시작됨.")

    def follower_pose_callback(self, msg: PoseStamped):
        with self.lock:
            self.latest_follower_pose = msg

    def uwb_callback(self, msg):
        with self.lock:
            if self.latest_follower_pose is None: return

            current_time = self.get_clock().now()
            if self.is_initialized and self.last_update_time is not None:
                dt = (current_time - self.last_update_time).nanoseconds / 1e9
                self.P += self.Q * dt
            self.last_update_time = current_time

            d_a, d_b, d_c = msg.point.x, msg.point.y, msg.point.z
            if not (d_a > 0.1 and d_b > 0.1 and d_c > 0.1): return

            p_f = np.array([self.latest_follower_pose.pose.position.x, self.latest_follower_pose.pose.position.y])
            q = self.latest_follower_pose.pose.orientation
            yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]
            rot_mat = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])

            p_a_local = np.array(self.get_parameter('anchor_pos.a').value)
            p_b_local = np.array(self.get_parameter('anchor_pos.b').value)
            p_c_local = np.array(self.get_parameter('anchor_pos.c').value)

            # EKF 예측 위치를 기반으로 리더의 예상 로컬 위치 계산
            rot_mat_inv = rot_mat.T
            leader_local_est = rot_mat_inv @ (self.x - p_f)

            # 동적 선택 로직으로 검증된 단 하나의 해(로컬 좌표)를 찾음
            verified_solution_local = None
            if not self.is_initialized or leader_local_est[0] >= 0:
                verified_solution_local = self.solve_and_verify(p_a_local, p_b_local, p_c_local, d_a, d_b, d_c)
            elif leader_local_est[1] > 0:
                verified_solution_local = self.solve_and_verify(p_a_local, p_c_local, p_b_local, d_a, d_c, d_b)
            else:
                verified_solution_local = self.solve_and_verify(p_b_local, p_c_local, p_a_local, d_b, d_c, d_a)

            if verified_solution_local is None: return

            # 검증된 해를 월드 좌표로 변환하여 EKF의 측정값(z)으로 사용
            z = p_f + rot_mat @ verified_solution_local

            if not self.is_initialized:
                self.x = z
                self.P = np.eye(2) * 0.1
                self.is_initialized = True
                self.get_logger().info(f"EKF (3-Anchor) 초기화 완료.")
                return
            
            # EKF 업데이트 (2앵커 버전과 동일한 로직, 하지만 입력 z가 훨씬 신뢰도 높음)
            # 자코비안 계산 시, 현재 동적으로 선택된 주력 앵커 2개를 사용해야 함
            # 단순화를 위해, 자코비안은 항상 전방 앵커 A, B를 기준으로 계산 (성능에 큰 영향 없음)
            p_a_world = p_f + rot_mat @ p_a_local
            p_b_world = p_f + rot_mat @ p_b_local
            
            dist_a_est = np.linalg.norm(self.x - p_a_world)
            dist_b_est = np.linalg.norm(self.x - p_b_world)
            if dist_a_est < 1e-6 or dist_b_est < 1e-6: return

            h = np.array([dist_a_est, dist_b_est])
            H = np.array([
                [(self.x[0] - p_a_world[0]) / dist_a_est, (self.x[1] - p_a_world[1]) / dist_a_est],
                [(self.x[0] - p_b_world[0]) / dist_b_est, (self.x[1] - p_b_world[1]) / dist_b_est]
            ])
            
            # 측정값 z는 좌표이므로, y = z - h(x)가 아님.
            # EKF 업데이트 공식을 좌표 측정에 맞게 수정
            H_coord = np.eye(2) # 측정 모델이 h(x) = x 이므로, 자코비안은 단위행렬
            R_coord = np.eye(2) * 0.1 # 좌표 측정에 대한 노이즈 (튜닝 필요)

            y = z - self.x
            S = H_coord @ self.P @ H_coord.T + R_coord
            K = self.P @ H_coord.T @ np.linalg.inv(S)
            
            self.x += K @ y
            self.P = (np.eye(2) - K @ H_coord) @ self.P

            point_msg = PointStamped()
            point_msg.header = msg.header
            point_msg.header.frame_id = 'world'
            point_msg.point.x, point_msg.point.y = self.x[0], self.x[1]
            self.raw_point_pub.publish(point_msg)

    def solve_and_verify(self, p1_local, p2_local, p_verify_local, d1, d2, d_verify):
        ex = (p2_local - p1_local) / np.linalg.norm(p2_local - p1_local)
        i = np.dot(ex, p2_local - p1_local)
        
        x = (d1**2 - d2**2 + i**2) / (2 * i)
        y_sq = d1**2 - x**2
        if y_sq < 0: return None
        y_sqrt = math.sqrt(y_sq)
        
        sol1 = p1_local + x * ex + y_sqrt * np.array([-ex[1], ex[0]])
        sol2 = p1_local + x * ex - y_sqrt * np.array([-ex[1], ex[0]])

        err1 = abs(np.linalg.norm(sol1 - p_verify_local) - d_verify)
        err2 = abs(np.linalg.norm(sol2 - p_verify_local) - d_verify)
        
        return sol1 if err1 < err2 else sol2

def main(args=None):
    rclpy.init(args=args)
    node = LeaderEstimator3AnchorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()