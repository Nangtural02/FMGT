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
        self.declare_parameter('leader_history_size', 20)
        self.declare_parameter('goal_update_threshold', 0.3)
        
        # 거리 기준
        self.declare_parameter('approach_dist_thresh', 3.0)
        self.declare_parameter('follow_dist_thresh', 0.8)
        
        # 제어 게인
        self.declare_parameter('kp_angle', 2.0)
        self.declare_parameter('kp_pos', 0.5)
        
        # 상태별 속도 제한
        self.declare_parameter('approach_max_linear_speed', 0.7)
        self.declare_parameter('approach_max_angular_speed', 0.1)
        self.declare_parameter('follow_max_linear_speed', 0.4)
        self.declare_parameter('follow_max_angular_speed', 0.3)
        
        # Align 상태
        self.declare_parameter('align_rot_speed', 0.5)
        self.declare_parameter('align_angle_complete_deg', 3.0)
        self.declare_parameter('align_trigger_angle_deg', 30.0)

        # 내부 상태 변수
        self.robot_state = 'Stop'
        self.goal_point = None
        history_size = self.get_parameter('leader_history_size').get_parameter_value().integer_value
        self.leader_pos_history = deque(maxlen=history_size)
        self.state_before_align = 'Stop'
        self.align_target_point = None

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

        self.get_logger().info('Follower Control Node (Final Logic) 시작됨.')

    def control_loop_callback(self, l_pos_msg, f_pose_msg):
        # --- 1. 상태 측정 및 목표 지점 계획 ---
        p_leader_current = np.array([l_pos_msg.point.x, l_pos_msg.point.y])
        p_follower = np.array([f_pose_msg.pose.position.x, f_pose_msg.pose.position.y])
        follower_yaw = quaternion_to_yaw(f_pose_msg.pose.orientation)
        self.leader_pos_history.append(p_leader_current)

        if len(self.leader_pos_history) > 0:
            smoothed_leader_pos = np.mean(self.leader_pos_history, axis=0)
        else: return

        if self.goal_point is None: self.goal_point = smoothed_leader_pos

        goal_update_threshold = self.get_parameter('goal_update_threshold').get_parameter_value().double_value
        if np.linalg.norm(smoothed_leader_pos - self.goal_point) > goal_update_threshold:
            self.goal_point = smoothed_leader_pos
        
        # --- 2. 상태 전환 로직 (4-State Machine) ---
        p_relative_to_goal = self.goal_point - p_follower
        current_dist_to_goal = np.linalg.norm(p_relative_to_goal)
        
        target_point_for_angle = self.align_target_point if self.robot_state == 'Aligning' and self.align_target_point is not None else self.goal_point
        p_relative_for_angle = target_point_for_angle - p_follower
        angle_to_target = math.atan2(p_relative_for_angle[1], p_relative_for_angle[0])
        angle_error = normalize_angle(angle_to_target - follower_yaw)
        
        current_state = self.robot_state
        align_trigger_rad = math.radians(self.get_parameter('align_trigger_angle_deg').value)
        approach_dist = self.get_parameter('approach_dist_thresh').value
        follow_dist = self.get_parameter('follow_dist_thresh').value
        
        if current_state == 'Aligning':
            align_complete_rad = math.radians(self.get_parameter('align_angle_complete_deg').value)
            if abs(angle_error) < align_complete_rad:
                self.robot_state = self.state_before_align
        else:
            if abs(angle_error) > align_trigger_rad:
                self.state_before_align = current_state if current_state != 'Stop' else 'Following'
                self.robot_state = 'Aligning'
                self.align_target_point = self.goal_point.copy()
            elif current_dist_to_goal > approach_dist:
                self.robot_state = 'Approaching'
            elif current_dist_to_goal < follow_dist:
                self.robot_state = 'Stop'
            else:
                self.robot_state = 'Following'

        # --- 3. 행동 결정 ---
        twist_msg = Twist()
        log_angle_err = math.degrees(angle_error)
        log_dist_err = current_dist_to_goal - follow_dist

        if self.robot_state == 'Aligning':
            kp_angle = self.get_parameter('kp_angle').value
            max_rot = self.get_parameter('align_rot_speed').value
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = np.clip(kp_angle * angle_error, -max_rot, max_rot)

        elif self.robot_state == 'Approaching' or self.robot_state == 'Following':
            kp_pos = self.get_parameter('kp_pos').value
            kp_angle = self.get_parameter('kp_angle').value
            
            if self.robot_state == 'Approaching':
                max_linear = self.get_parameter('approach_max_linear_speed').value
                max_angular = self.get_parameter('approach_max_angular_speed').value
            else: # Following
                max_linear = self.get_parameter('follow_max_linear_speed').value
                max_angular = self.get_parameter('follow_max_angular_speed').value
            
            twist_msg.linear.x = np.clip(kp_pos * log_dist_err, 0.0, max_linear)
            twist_msg.angular.z = np.clip(kp_angle * angle_error, -max_angular, max_angular)
        
        self.publisher_.publish(twist_msg)

        # --- 4. 디버깅 ---
        goal_msg = PointStamped()
        goal_msg.header.stamp = self.get_clock().now().to_msg(); goal_msg.header.frame_id = 'world'
        goal_msg.point.x = self.goal_point[0]; goal_msg.point.y = self.goal_point[1]
        self.goal_point_pub_.publish(goal_msg)

        log_msg = (
            f"State: {self.robot_state} | DistToGoal: {current_dist_to_goal:.2f}m | "
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