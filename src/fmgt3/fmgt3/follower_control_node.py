# fmgt/follower_control_node.py (데드존 + 직진 관성 + 안전 종료)
import rclpy, math, time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TwistStamped, Twist
import message_filters

class FollowerControlNode(Node):
    def __init__(self):
        super().__init__('follower_control_node')

        # --- 제어 파라미터 ---
        self.declare_parameter('target_distance', 0.8)
        self.declare_parameter('kp_pos', 0.7)
        self.declare_parameter('kp_vel', 0.4)
        self.declare_parameter('alpha_lpf', 0.2)
        self.declare_parameter('k_omega', 2.5)
        self.declare_parameter('max_linear_speed', 1.0)
        self.declare_parameter('max_angular_speed', 1.0)
        # ★★★ 문제 해결을 위한 새 파라미터 ★★★
        self.declare_parameter('deadzone_distance', 0.1) # 목표 거리 ±10cm는 무시
        self.declare_parameter('deadzone_angle_deg', 5.0)  # 목표 각도 ±5도는 무시
        self.declare_parameter('forward_angle_threshold_deg', 20.0) # 이 각도 이내여야 전진

        self.v_cmd_smoothed = np.zeros(2)

        # 구독자
        pos_sub = message_filters.Subscriber(self, PointStamped, 'relative_position')
        vel_sub = message_filters.Subscriber(self, TwistStamped, 'relative_velocity')
        self.ts = message_filters.TimeSynchronizer([pos_sub, vel_sub], 10)
        self.ts.registerCallback(self.control_loop_callback)
        
        # 발행자
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info('Advanced Follower Node 시작됨.')

    def control_loop_callback(self, pos_msg, vel_msg):
        # 파라미터 가져오기
        target_dist = self.get_parameter('target_distance').get_parameter_value().double_value
        kp_pos = self.get_parameter('kp_pos').get_parameter_value().double_value
        kp_vel = self.get_parameter('kp_vel').get_parameter_value().double_value
        alpha = self.get_parameter('alpha_lpf').get_parameter_value().double_value
        k_omega = self.get_parameter('k_omega').get_parameter_value().double_value
        max_linear = self.get_parameter('max_linear_speed').get_parameter_value().double_value
        max_angular = self.get_parameter('max_angular_speed').get_parameter_value().double_value
        deadzone_dist = self.get_parameter('deadzone_distance').get_parameter_value().double_value
        deadzone_angle = math.radians(self.get_parameter('deadzone_angle_deg').get_parameter_value().double_value)
        forward_angle_threshold = math.radians(self.get_parameter('forward_angle_threshold_deg').get_parameter_value().double_value)

        # 상태 추출
        p_est = np.array([pos_msg.point.x, pos_msg.point.y])
        v_rel = np.array([vel_msg.twist.linear.x, vel_msg.twist.linear.y])
        
        current_dist = np.linalg.norm(p_est)
        if current_dist < 1e-6: self.publisher_.publish(Twist()); return
        
        # ★★★ 문제 1 해결: 데드존 적용 ★★★
        error_dist = current_dist - target_dist
        if abs(error_dist) < deadzone_dist:
            error_dist = 0.0 # 거리 오차가 데드존 안에 있으면 0으로 취급

        p_hat = p_est / current_dist
        v_correct = kp_pos * error_dist * p_hat
        v_follow = kp_vel * v_rel
        v_cmd_raw = v_correct + v_follow
        self.v_cmd_smoothed = alpha * v_cmd_raw + (1 - alpha) * self.v_cmd_smoothed

        # 로봇 명령 변환
        twist_msg = Twist()
        twist_msg.linear.x = self.v_cmd_smoothed[1]
        
        # 각도 오차 계산
        target_angle = math.atan2(p_est[0], p_est[1]) # y축 기준 각도
        
        # ★★★ 문제 1 & 2 해결: 각도 데드존 및 직진 관성 ★★★
        # 각도 오차가 아주 작으면 회전 명령을 0으로 만듦 (떨림 방지)
        if abs(target_angle) < deadzone_angle:
            twist_msg.angular.z = 0.0
        else:
            twist_msg.angular.z = k_omega * self.v_cmd_smoothed[0]

        # 정렬이 잘 되었을 때만 전진, 아니면 회전에만 집중 (지그재그 방지)
        if abs(target_angle) > forward_angle_threshold:
            twist_msg.linear.x = 0.0
            
        # 속도 제한
        twist_msg.linear.x = max(0.0, min(max_linear, twist_msg.linear.x))
        twist_msg.angular.z = max(-max_angular, min(max_angular, twist_msg.angular.z))

        self.publisher_.publish(twist_msg)

    def shutdown_sequence(self):
        """노드 종료 시 안전하게 로봇을 멈추는 함수"""
        self.get_logger().info('Controller 종료 시퀀스 시작...')
        # 속도가 0인 Twist 메시지를 여러 번 보내 확실하게 정지
        stop_msg = Twist()
        for _ in range(5):
            if rclpy.ok():
                self.publisher_.publish(stop_msg)
                time.sleep(0.02) # 20ms 간격
            else:
                break
        self.get_logger().info('로봇 정지 명령 발행 완료.')

def main(args=None):
    rclpy.init(args=args)
    node = FollowerControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt! 노드를 종료합니다...')
    finally:
        # ★★★ 문제 3 해결: 올바른 종료 순서 ★★★
        # rclpy.shutdown() 전에 노드의 정리 작업을 먼저 수행
        node.shutdown_sequence()
        node.destroy_node()
        # 이미 종료되었을 수 있으므로 try_shutdown 사용
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()