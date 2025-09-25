# 파일명: point_postprocessor_node.py
"""
Point Post-processor Node

이 노드는 리더의 위치 추정을 담당합니다. 2단계 필터링을 통해 안정적인 제어 목표점을 생성합니다.
1. KF(칼만 필터): '날것의' 위치 측정치의 노이즈를 1차적으로 제거합니다.
2. 이동 평균 필터 + 문턱값: KF 출력의 미세한 떨림을 2차적으로 제거하고, 
   '의미 있는' 움직임이 감지될 때만 최종 목표 지점을 갱신합니다.

- 구독 (Subscriptions):
  - /leader/raw_point (geometry_msgs/PointStamped): Leader Estimator가 발행한 리더의 위치 측정치

- 발행 (Publications):
  - /controller/goal_point (geometry_msgs/PointStamped): 제어기가 사용할 최종 목표 지점

- 파라미터 (Parameters):
  - leader_process_variance (double): 리더 KF의 프로세스 노이즈 공분산. 리더의 예측 불확실성.
  - leader_measurement_variance (double): 리더 KF의 측정 노이즈 공분산. UWB 측정치의 신뢰도.
  - leader_history_size (int): 이동 평균 필터의 윈도우(샘플) 크기.
  - goal_update_threshold (double): 최종 목표점을 갱신하기 위한 최소 거리 (m).
"""
import rclpy, numpy as np, math, threading
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from collections import deque

class PointPostprocessorNode(Node):
    def __init__(self):
        super().__init__('point_postprocessor_node')
        
        self.declare_parameter('leader_process_variance', 0.2**2)
        self.declare_parameter('leader_measurement_variance', 0.3**2)
        self.declare_parameter('leader_history_size', 10)
        self.declare_parameter('goal_update_threshold', 0.15)

        self.goal_point_pub = self.create_publisher(PointStamped, '/controller/goal_point', 10)
        self.lock = threading.Lock()
        
        self.x_l = np.zeros(2); self.P_l = np.eye(2) * 1000
        q_var = self.get_parameter('leader_process_variance').value
        self.Q_l = np.diag([q_var, q_var])
        r_var = self.get_parameter('leader_measurement_variance').value
        self.R_l = np.diag([r_var, r_var])
        self.is_leader_initialized = False; self.last_update_timestamp = None

        history_size = self.get_parameter('leader_history_size').value
        self.kf_pos_history = deque(maxlen=history_size)
        self.final_goal_point = None

        self.raw_point_sub = self.create_subscription(PointStamped, '/leader/raw_point', self.raw_point_callback, 10)
        self.get_logger().info("Point Post-processor Node (KF + Moving Avg) 시작됨.")

    def raw_point_callback(self, msg: PointStamped):
        with self.lock:
            current_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if not self.is_leader_initialized:
                self.x_l = np.array([msg.point.x, msg.point.y])
                self.final_goal_point = self.x_l.copy()
                self.last_update_timestamp = current_timestamp
                self.is_leader_initialized = True
                self.get_logger().info(f"리더 필터 초기화 완료: ({self.x_l[0]:.2f}, {self.x_l[1]:.2f})")
                return

            dt = current_timestamp - self.last_update_timestamp
            self.last_update_timestamp = current_timestamp
            if not (0 < dt < 0.5): return
            
            self.P_l += self.Q_l * dt

            z = np.array([msg.point.x, msg.point.y])
            H_l = np.eye(2); y_err = z - self.x_l
            S = H_l @ self.P_l @ H_l.T + self.R_l
            try:
                K = self.P_l @ H_l.T @ np.linalg.inv(S)
                self.x_l += K @ y_err
                self.P_l = (np.eye(2) - K @ H_l) @ self.P_l
            except np.linalg.LinAlgError: return

            self.kf_pos_history.append(self.x_l)
            smoothed_pos = np.mean(self.kf_pos_history, axis=0)
            
            update_threshold = self.get_parameter('goal_update_threshold').value
            if np.linalg.norm(smoothed_pos - self.final_goal_point) > update_threshold:
                self.final_goal_point = smoothed_pos

            goal_point_msg = PointStamped()
            goal_point_msg.header = msg.header
            goal_point_msg.point.x, goal_point_msg.point.y = self.final_goal_point[0], self.final_goal_point[1]
            self.goal_point_pub.publish(goal_point_msg)

def main(args=None):
    rclpy.init(args=args); node = PointPostprocessorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__': main()