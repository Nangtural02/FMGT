# fmgt4/follower_control_node.py
import rclpy, math, time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist, PoseStamped
from scipy.spatial.transform import Rotation
import message_filters
from collections import deque

def quaternion_to_yaw(q):
    try:
        return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx', degrees=False)[0]
    except Exception:
        return 0.0

def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

class FollowerControlNode(Node):
    def __init__(self):
        super().__init__('follower_control_node')
        # 파라미터 선언 (당신이 수정한 값 반영)
        self.declare_parameter('d_limit', 1.0)
        self.declare_parameter('d_hysteresis', 0.2)
        self.declare_parameter('approach_kp_angle', 2.0)
        self.declare_parameter('approach_fwd_speed', 0.5)
        self.declare_parameter('approach_rot_speed', 0.2)
        self.declare_parameter('approach_angle_tol_deg', 20.0)
        self.declare_parameter('max_linear_speed', 1.5)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('leader_history_size', 100)
        self.declare_parameter('goal_update_threshold', 0.5)
        self.declare_parameter('follow_kp_pos', 0.7)

        # 내부 상태 변수
        self.robot_state = 'Stop'
        self.goal_point = None
        history_size = self.get_parameter('leader_history_size').get_parameter_value().integer_value
        self.leader_pos_history = deque(maxlen=history_size)

        # 구독자
        leader_pos_sub = message_filters.Subscriber(self, PointStamped, '/leader/estimated_position')
        follower_pose_sub = message_filters.Subscriber(self, PoseStamped, '/follower/estimated_pose')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [leader_pos_sub, follower_pose_sub], 
            queue_size=10, slop=0.2
        )
        self.ts.registerCallback(self.control_loop_callback)
        
        # 발행자
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.goal_point_pub_ = self.create_publisher(PointStamped, '/debug/goal_point', 10)

        self.get_logger().info('Follower Control Node (Final Version with Stop Logic) 시작됨.')

    def control_loop_callback(self, l_pos_msg, f_pose_msg):
        # 파라미터 가져오기
        d_limit = self.get_parameter('d_limit').get_parameter_value().double_value
        d_hyst = self.get_parameter('d_hysteresis').get_parameter_value().double_value
        
        # --- 1. 상태 측정 및 데이터 수집 ---
        p_leader_current = np.array([l_pos_msg.point.x, l_pos_msg.point.y])
        p_follower = np.array([f_pose_msg.pose.position.x, f_pose_msg.pose.position.y])
        follower_yaw = quaternion_to_yaw(f_pose_msg.pose.orientation)
        self.leader_pos_history.append(p_leader_current)

        # --- 2. 목표 지점(Goal Point) 계획 ---
        if len(self.leader_pos_history) > 0:
            smoothed_leader_pos = np.mean(self.leader_pos_history, axis=0)
        else:
            return

        if self.goal_point is None:
            self.goal_point = smoothed_leader_pos

        goal_update_threshold = self.get_parameter('goal_update_threshold').get_parameter_value().double_value
        if np.linalg.norm(smoothed_leader_pos - self.goal_point) > goal_update_threshold:
            self.goal_point = smoothed_leader_pos

        # --- 3. 제어 로직 ---
        p_relative_to_goal = self.goal_point - p_follower
        current_dist = np.linalg.norm(p_relative_to_goal)

        # 상태 전환 로직 (Stop 상태는 이제 사용되지 않지만, 디버깅을 위해 유지)
        d_upper = d_limit + d_hyst
        d_lower = d_limit - d_hyst
        if len(self.leader_pos_history) == self.leader_pos_history.maxlen:
            path_variance = np.var(self.leader_pos_history, axis=0)
            is_leader_stopped = np.sum(path_variance) < 0.01
        else:
            is_leader_stopped = False

        if current_dist > d_upper:
            self.robot_state = 'Approaching'
        elif current_dist < d_lower and is_leader_stopped:
            self.robot_state = 'Stop' # 이 조건에 도달하기 어려움
        else:
            self.robot_state = 'Following'

        twist_msg = Twist()
        log_angle_err, log_dist_err = 0.0, 0.0

        if self.robot_state == 'Approaching':
            approach_angle_tol = math.radians(self.get_parameter('approach_angle_tol_deg').value)
            angle_to_goal = math.atan2(p_relative_to_goal[1], p_relative_to_goal[0])
            angle_error = normalize_angle(angle_to_goal - follower_yaw)
            log_angle_err = math.degrees(angle_error)
            
            if abs(angle_error) > approach_angle_tol:
                twist_msg.linear.x = 0.0
                approach_kp_angle = self.get_parameter('approach_kp_angle').value
                approach_rot_speed = self.get_parameter('approach_rot_speed').value
                # ★★★ 변경점: 당신의 테스트 결과를 반영하여 (+) 부호 사용 ★★★
                angular_vel = approach_kp_angle * angle_error
                twist_msg.angular.z = np.clip(angular_vel, -approach_rot_speed, approach_rot_speed)
            else:
                fwd_speed = self.get_parameter('approach_fwd_speed').value
                twist_msg.linear.x = fwd_speed
                dynamic_max_angular = 0.2 * fwd_speed
                approach_kp_angle = self.get_parameter('approach_kp_angle').value
                angular_vel = approach_kp_angle * angle_error
                twist_msg.angular.z = np.clip(angular_vel, -dynamic_max_angular, dynamic_max_angular)

        elif self.robot_state == 'Following':
            kp_pos = self.get_parameter('follow_kp_pos').value
            max_linear = self.get_parameter('max_linear_speed').value
            max_angular = self.get_parameter('max_angular_speed').value
            
            distance_error = current_dist - d_limit
            log_dist_err = distance_error
            
            # ★★★ 변경점: 완전 정지 로직 추가 ★★★
            # 목표 거리보다 가까워지면 모든 움직임을 멈춤
            if distance_error <= 0.0:
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                self.robot_state = 'Stop' # 상태를 명시적으로 변경
            else:
                angle_to_goal = math.atan2(p_relative_to_goal[1], p_relative_to_goal[0])
                angle_error = normalize_angle(angle_to_goal - follower_yaw)
                log_angle_err = math.degrees(angle_error)
                
                linear_speed = np.clip(kp_pos * distance_error, 0.0, max_linear)
                twist_msg.linear.x = linear_speed
                
                dynamic_max_angular = 0.2 * linear_speed
                kp_angle = self.get_parameter('approach_kp_angle').value
                angular_vel = kp_angle * angle_error
                twist_msg.angular.z = np.clip(angular_vel, -dynamic_max_angular, dynamic_max_angular)
        
        self.publisher_.publish(twist_msg)

        # --- 디버깅 시각화 및 로그 ---
        goal_msg = PointStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg()
        goal_msg.header.frame_id = 'world'
        goal_msg.point.x = self.goal_point[0]
        goal_msg.point.y = self.goal_point[1]
        self.goal_point_pub_.publish(goal_msg)

        log_msg = (
            f"State: {self.robot_state} | "
            f"DistToGoal: {current_dist:.2f}m | DistErr: {log_dist_err:.2f}m | "
            f"AngleErr: {log_angle_err:.1f}deg | "
            f"Cmd: [{twist_msg.linear.x:.2f}m/s, {twist_msg.angular.z:.2f}rad/s]"
        )
        self.get_logger().info(log_msg, throttle_duration_sec=0.2)

    def shutdown_sequence(self):
        self.get_logger().info('Controller 종료 시퀀스 시작...')
        stop_msg = Twist()
        for _ in range(5):
            if rclpy.ok(): self.publisher_.publish(stop_msg); time.sleep(0.02)
        self.get_logger().info('로봇 정지 명령 발행 완료.')

def main(args=None):
    rclpy.init(args=args)
    node = FollowerControlNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.shutdown_sequence(); node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()