# follower_control_node_stateful.py
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
import math

class FollowerControlNode(Node):
    def __init__(self):
        super().__init__('follower_control_node')

        # --- 제어 파라미터 선언 ---
        # 영역(Dead Zone) 정의
        self.declare_parameter('target_distance', 0.8)      # 목표 거리 (미터)
        self.declare_parameter('distance_tolerance', 0.3)   # 거리 허용 오차 (e.g., 0.8 ± 0.3m)
        self.declare_parameter('angle_tolerance_deg', 10.0) # 정면으로 간주할 각도 허용 오차 (도)

        # 속도 및 게인 정의
        self.declare_parameter('forward_speed', 0.15)       # 전진 시 고정 속도 (m/s)
        self.declare_parameter('rotation_speed', 0.5)       # 회전 시 고정 속도 (rad/s)
        self.declare_parameter('kp_angle', 2.0)             # 회전 제어를 위한 P 게인
        
        self.declare_parameter('max_angular_speed', 2.84)   # 최대 각속도

        # 내부 상태 변수
        self.robot_state = 'IDLE'

        # 구독자 및 발행자
        self.subscription = self.create_subscription(
            PointStamped, 'relative_polar_position', self.control_loop_callback, 10)
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)

        self.get_logger().info('상태 기반 Follower Control Node가 시작되었습니다.')

    def control_loop_callback(self, msg):
        # 파라미터 값 가져오기
        target_dist = self.get_parameter('target_distance').get_parameter_value().double_value
        dist_tol = self.get_parameter('distance_tolerance').get_parameter_value().double_value
        angle_tol = math.radians(self.get_parameter('angle_tolerance_deg').get_parameter_value().double_value)
        fwd_speed = self.get_parameter('forward_speed').get_parameter_value().double_value
        rot_speed = self.get_parameter('rotation_speed').get_parameter_value().double_value
        kp_angle = self.get_parameter('kp_angle').get_parameter_value().double_value
        max_angular = self.get_parameter('max_angular_speed').get_parameter_value().double_value

        # 1. 상대 위치 데이터 및 오차 계산
        pos_x = msg.point.x
        pos_y = msg.point.y
        current_dist = math.sqrt(pos_x**2 + pos_y**2)

        # 2. 상태 결정 (State Determination)
        is_too_far = current_dist > (target_dist + dist_tol)
        is_aligned = abs(pos_x) < (current_dist * math.sin(angle_tol)) # 기하학적으로 계산된 각도 오차

        # 상태 전환 로직
        if not is_too_far:
            self.robot_state = 'IDLE'
        elif is_too_far and not is_aligned:
            self.robot_state = 'ROTATING'
        elif is_too_far and is_aligned:
            self.robot_state = 'MOVING_FORWARD'

        # 3. 상태에 따른 행동 결정 (Action Selection)
        twist_msg = Twist()
        if self.robot_state == 'IDLE':
            # 목표: 정지
            pass # 기본값이 0이므로 아무것도 안함
        elif self.robot_state == 'ROTATING':
            # 목표: 리더를 향해 회전
            # P제어로 부드러운 회전, 고정 속도로 최대치 제한
            angular_vel = -kp_angle * pos_x
            twist_msg.angular.z = max(-rot_speed, min(rot_speed, angular_vel))
        elif self.robot_state == 'MOVING_FORWARD':
            # 목표: 앞으로 전진
            twist_msg.linear.x = fwd_speed

        # 4. 제어 명령 발행
        self.publisher_.publish(twist_msg)

        # 현재 상태 로깅
        self.get_logger().info(
            f'State: {self.robot_state} | '
            f'Dist: {current_dist:.2f}m (Target: {target_dist:.2f}m) | '
            f'Aligned: {is_aligned} | '
            f'LinVel: {twist_msg.linear.x:.2f} | AngVel: {twist_msg.angular.z:.2f}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = FollowerControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 로봇을 멈추기
        stop_msg = Twist()
        if rclpy.ok() and node.publisher_.handle is not None:
             node.publisher_.publish(stop_msg)
        node.get_logger().info('노드 종료. 로봇을 정지합니다.')
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()