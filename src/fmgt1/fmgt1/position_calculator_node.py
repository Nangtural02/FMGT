# position_calculator_node.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
# 커스텀 메시지 임포트
from uwb_interfaces.msg import PolarCoordinate
import math
import numpy as np
from collections import deque

# --- 설정 값 및 필터 클래스는 이전과 동일 ---
ANCHOR_DISTANCE = 0.5
ANCHOR_A_POS = (-ANCHOR_DISTANCE / 2, 0)
ANCHOR_B_POS = (ANCHOR_DISTANCE / 2, 0)
FILTER_TYPE = 'KF'
MAF_WINDOW_SIZE = 10
KF_PROCESS_NOISE = 1e-4
KF_MEASUREMENT_NOISE = 4e-2

class NoFilter:
    def update(self, measurement): return measurement
class MovingAverageFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = deque(maxlen=window_size)
    def update(self, measurement):
        self.data.append(measurement)
        return sum(self.data) / len(self.data)
class KalmanFilter1D:
    def __init__(self, process_noise, measurement_noise, initial_value=0.0):
        self.Q, self.R, self.P, self.x = process_noise, measurement_noise, 1.0, initial_value
    def update(self, measurement):
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (measurement - self.x)
        self.P = (1 - K) * self.P
        return self.x
# ----------------------------------------------


class PositionCalculatorNode(Node):
    def __init__(self):
        super().__init__('position_calculator_node')
        
        self.subscription = self.create_subscription(
            Point,
            'raw_uwb_distances',
            self.listener_callback,
            10)
        
        # 발행자: 극좌표 정보를 발행하도록 변경
        self.publisher_ = self.create_publisher(PolarCoordinate, 'relative_polar_position', 10)

        # 필터 초기화 (이전과 동일)
        if FILTER_TYPE == 'MAF':
            self.filter_a = MovingAverageFilter(MAF_WINDOW_SIZE)
            self.filter_b = MovingAverageFilter(MAF_WINDOW_SIZE)
        elif FILTER_TYPE == 'KF':
            self.filter_a = KalmanFilter1D(KF_PROCESS_NOISE, KF_MEASUREMENT_NOISE)
            self.filter_b = KalmanFilter1D(KF_PROCESS_NOISE, KF_MEASUREMENT_NOISE)
        else:
            self.filter_a = NoFilter()
            self.filter_b = NoFilter()

        self.get_logger().info(f'Position Calculator Node가 시작되었습니다. (Filter: {FILTER_TYPE})')

    def listener_callback(self, msg):
        raw_dist_a = msg.x
        raw_dist_b = msg.y
        
        filt_dist_a = self.filter_a.update(raw_dist_a)
        filt_dist_b = self.filter_b.update(raw_dist_b)

        # 1. 데카르트 좌표 계산
        cartesian_pos = self.calculate_cartesian_position(filt_dist_a, filt_dist_b)

        if cartesian_pos[0] is not None:
            x, y = cartesian_pos
            
            # 2. 데카르트 -> 극좌표 변환
            r = math.sqrt(x**2 + y**2)
            theta = math.atan2(y,x)
            # theta_rad = math.atan2(y, x) # y, x 순서가 중요!
            # theta = math.degrees(theta_rad)

            # 3. PolarCoordinate 메시지로 발행
            polar_msg = PolarCoordinate()
            polar_msg.r = r
            polar_msg.theta = theta
            
            self.publisher_.publish(polar_msg)
            # self.get_logger().info(f'Publishing Polar: r={r:.2f} m, theta={math.degrees(theta):.1f} deg')

    def calculate_cartesian_position(self, d_a, d_b):
        L = ANCHOR_DISTANCE
        if d_a <= 0 or d_b <= 0 or d_a + d_b < L or abs(d_a - d_b) > L: return (None, None)
        try:
            x = (d_a**2 - d_b**2) / (2 * L)
            y_squared = d_a**2 - (x - ANCHOR_A_POS[0])**2
            if y_squared < 0: return (None, None)
            y = math.sqrt(y_squared)
            return (x, y)
        except (ValueError, ZeroDivisionError): return (None, None)

def main(args=None):
    rclpy.init(args=args)
    node = PositionCalculatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()