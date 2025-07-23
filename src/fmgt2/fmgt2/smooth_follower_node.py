# fmgt/smooth_follower_node.py
import rclpy, math
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist

class FollowerControlNode(Node):
    def __init__(self):
        super().__init__('smooth_follower_node') # 런치 파일의 이름과 맞춤

        # --- 파라미터 선언 ---
        self.declare_parameter('target_distance', 0.8)
        self.declare_parameter('distance_tolerance', 0.3)
        self.declare_parameter('angle_tolerance_deg', 15.0)
        self.declare_parameter('forward_speed', 0.5)
        self.declare_parameter('kp_angle', 2.0)
        self.declare_parameter('rotation_speed', 0.8)

        self.robot_state = 'IDLE'

        # ★★★ EKF가 발행하는 'estimated_position'을 구독 ★★★
        self.subscription = self.create_subscription(
            PointStamped, 'relative_position', self.control_loop_callback, 10)
        
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info('상태 기반 Follower Node 시작됨.')

    def control_loop_callback(self, msg):
        target_dist = self.get_parameter('target_distance').get_parameter_value().double_value
        dist_tol = self.get_parameter('distance_tolerance').get_parameter_value().double_value
        angle_tol_rad = math.radians(self.get_parameter('angle_tolerance_deg').get_parameter_value().double_value)
        fwd_speed = self.get_parameter('forward_speed').get_parameter_value().double_value
        rot_speed = self.get_parameter('rotation_speed').get_parameter_value().double_value
        kp_angle = self.get_parameter('kp_angle').get_parameter_value().double_value
        pos_x, pos_y = msg.point.x, msg.point.y

        if pos_y <= 0.1:
            self.robot_state = 'IDLE'
            self.publisher_.publish(Twist())
            return
            
        current_dist = math.sqrt(pos_x**2 + pos_y**2)
        is_too_far = current_dist > (target_dist + dist_tol)
        is_aligned = abs(pos_x) < (current_dist * math.sin(angle_tol_rad))

        if not is_too_far: self.robot_state = 'IDLE'
        elif not is_aligned: self.robot_state = 'ROTATING'
        else: self.robot_state = 'MOVING_FORWARD'

        twist_msg = Twist()
        if self.robot_state == 'ROTATING':
            angular_vel = kp_angle * pos_x
            twist_msg.angular.z = max(-rot_speed, min(rot_speed, angular_vel))
        elif self.robot_state == 'MOVING_FORWARD':
            twist_msg.linear.x = fwd_speed
        self.publisher_.publish(twist_msg)

    def shutdown(self):
        self.get_logger().info('Follower 종료 중... 로봇 정지 명령 발행.')
        if rclpy.ok():
            self.publisher_.publish(Twist())
            time.sleep(0.1)

# main 함수는 반드시 필요합니다.
import time # shutdown을 위해 추가

def main(args=None):
    rclpy.init(args=args)
    node = FollowerControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()