# 파일명: point_postprocessor_node.py
"""
Point Post-processor Node

이 노드는 EKF로 1차 필터링된 리더의 위치를 입력받아, 최종적으로 로봇 제어기가
사용할 안정적인 목표점을 생성하는 역할을 합니다. 2단계 후처리 필터링을 수행합니다.

1. 이동 평균 필터 (Moving Average Filter): EKF 출력에 남아있을 수 있는 미세한
   떨림이나 노이즈를 제거하여 경로를 부드럽게 만듭니다.
2. 문턱값 기반 업데이트 (Threshold-based Update): 이동 평균 필터를 거친 위치가
   기존 목표점에서 '의미 있는' 거리(threshold) 이상으로 벗어났을 때만 최종 목표점을
   갱신합니다. 이를 통해 로봇이 불필요한 미세 움직임을 수행하는 것을 방지하고,
   리더가 실제로 움직였을 때만 반응하도록 합니다.

- 구독 (Subscriptions):
  - /leader/raw_point (geometry_msgs/PointStamped): EKF 기반 Leader Estimator가
    발행한, 1차 필터링된 리더의 위치 추정치.

- 발행 (Publications):
  - /controller/goal_point (geometry_msgs/PointStamped): 제어기가 추종할 최종 목표 지점.

- 파라미터 (Parameters):
  - leader_history_size (int): 이동 평균 필터의 윈도우(샘플) 크기입니다. 값이 클수록
    경로는 부드러워지지만, 리더의 움직임에 대한 반응은 약간 느려집니다.
  - goal_update_threshold (double): 최종 목표점을 갱신하기 위한 최소 거리(m)입니다.
    리더의 움직임이 이 값보다 작을 경우, 로봇은 현재 목표점을 유지하며 대기합니다.
"""
import rclpy
import numpy as np
import threading
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from collections import deque

class PointPostprocessorNode(Node):
    def __init__(self):
        super().__init__('point_postprocessor_node')
        
        # --- 파라미터 선언 (KF 관련 파라미터 제거) ---
        self.declare_parameter('leader_history_size', 15)
        self.declare_parameter('goal_update_threshold', 0.3)

        # --- 발행자 ---
        self.goal_point_pub = self.create_publisher(PointStamped, '/controller/goal_point', 10)
        
        # --- 내부 상태 변수 (KF 관련 변수 제거) ---
        self.lock = threading.Lock()
        
        history_size = self.get_parameter('leader_history_size').value
        # 입력된 포인트(EKF 출력)를 저장하기 위한 deque
        self.point_history = deque(maxlen=history_size)
        
        # 최종적으로 발행될 목표 지점
        self.final_goal_point = None
        self.is_initialized = False

        # --- 구독자 ---
        self.raw_point_sub = self.create_subscription(
            PointStamped, 
            '/leader/raw_point', 
            self.raw_point_callback, 
            10
        )
        self.get_logger().info("Point Post-processor Node (Moving Avg + Threshold) 시작됨.")

    def raw_point_callback(self, msg: PointStamped):
        with self.lock:
            current_point = np.array([msg.point.x, msg.point.y])

            # --- 1. 초기화 ---
            # 첫 번째 메시지를 받으면, 내부 상태를 초기화합니다.
            if not self.is_initialized:
                # deque를 현재 위치로 채워서 초기 평균이 튀는 것을 방지
                for _ in range(self.point_history.maxlen):
                    self.point_history.append(current_point)
                
                self.final_goal_point = current_point
                self.is_initialized = True
                self.get_logger().info(f"후처리 필터 초기화 완료: ({current_point[0]:.2f}, {current_point[1]:.2f})")
            
            # --- 2. 이동 평균 필터링 ---
            # 칼만 필터 대신, EKF 출력을 히스토리에 바로 추가합니다.
            self.point_history.append(current_point)
            # 히스토리의 평균을 계산하여 부드러운 위치를 얻습니다.
            smoothed_pos = np.mean(self.point_history, axis=0)
            
            # --- 3. 문턱값 기반 목표점 갱신 ---
            update_threshold = self.get_parameter('goal_update_threshold').value
            # 부드러워진 위치가 기존 목표점에서 일정 거리 이상 벗어났을 때만 목표를 갱신합니다.
            if np.linalg.norm(smoothed_pos - self.final_goal_point) > update_threshold:
                self.final_goal_point = smoothed_pos

            # --- 4. 최종 목표점 발행 ---
            goal_point_msg = PointStamped()
            goal_point_msg.header = msg.header
            goal_point_msg.point.x, goal_point_msg.point.y = self.final_goal_point[0], self.final_goal_point[1]
            self.goal_point_pub.publish(goal_point_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PointPostprocessorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()