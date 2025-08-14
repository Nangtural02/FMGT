import rclpy, math, time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TwistStamped, Twist, PoseStamped
from sensor_msgs.msg import Imu
from scipy.spatial.transform import Rotation
import message_filters

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
        self.declare_parameter('d_limit', 0.8)
        self.declare_parameter('d_hysteresis', 0.2)
        self.declare_parameter('approach_kp_angle', 2.0)
        self.declare_parameter('approach_fwd_speed', 0.22)
        self.declare_parameter('approach_rot_speed', 1.0)
        self.declare_parameter('approach_angle_tol_deg', 15.0)
        self.declare_parameter('follow_kp_pos', 0.7)
        self.declare_parameter('follow_kp_vel', 0.4)
        self.declare_parameter('follow_alpha_lpf', 0.2)
        self.declare_parameter('follow_k_omega', 2.5)
        self.declare_parameter('max_linear_speed', 0.22)
        self.declare_parameter('max_angular_speed', 1.5)

        self.robot_state = 'Stop'
        self.v_cmd_smoothed = np.zeros(2)
        self.prev_leader_pos = None
        self.prev_leader_timestamp = None
        self.last_known_yaw = 0.0

        leader_pos_sub = message_filters.Subscriber(self, PointStamped, '/leader/estimated_position')
        follower_pos_sub = message_filters.Subscriber(self, PointStamped, '/follower/estimated_position')
        follower_vel_sub = message_filters.Subscriber(self, TwistStamped, '/follower/estimated_velocity')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [leader_pos_sub, follower_pos_sub, follower_vel_sub], 
            queue_size=10, slop=0.2
        )
        self.ts.registerCallback(self.control_loop_callback)
        
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.debug_cmd_pub_ = self.create_publisher(PoseStamped, '/debug/control_vector', 10)
        self.get_logger().info('Follower Control Node (Final Version) 시작됨.')

    def control_loop_callback(self, l_pos_msg, f_pos_msg, f_vel_msg):
        d_limit = self.get_parameter('d_limit').get_parameter_value().double_value
        d_hyst = self.get_parameter('d_hysteresis').get_parameter_value().double_value
        
        p_leader = np.array([l_pos_msg.point.x, l_pos_msg.point.y])
        p_follower = np.array([f_pos_msg.point.x, f_pos_msg.point.y])
        v_follower = np.array([f_vel_msg.twist.linear.x, f_vel_msg.twist.linear.y])
        
        # EKF의 최종 Yaw 추정치를 사용
        if np.linalg.norm(v_follower) > 0.01:
            follower_yaw = math.atan2(v_follower[1], v_follower[0])
            self.last_known_yaw = follower_yaw
        else:
            follower_yaw = self.last_known_yaw
        
        current_time = l_pos_msg.header.stamp.sec + l_pos_msg.header.stamp.nanosec * 1e-9
        if self.prev_leader_pos is not None and self.prev_leader_timestamp is not None:
            dt = current_time - self.prev_leader_timestamp
            if dt > 1e-6: v_leader = (p_leader - self.prev_leader_pos) / dt
            else: v_leader = np.zeros(2)
        else: v_leader = np.zeros(2)
        self.prev_leader_pos = p_leader; self.prev_leader_timestamp = current_time
        
        p_relative = p_leader - p_follower
        v_relative = v_leader - v_follower
        current_dist = np.linalg.norm(p_relative)

        d_upper = d_limit + d_hyst; d_lower = d_limit - d_hyst
        is_leader_stopped = np.linalg.norm(v_leader) < 0.05

        if current_dist > d_upper: self.robot_state = 'Approaching'
        elif current_dist < d_lower and is_leader_stopped: self.robot_state = 'Stop'
        else: self.robot_state = 'Following'

        twist_msg = Twist()
        log_angle_err = 0.0; log_dist_err = 0.0

        if self.robot_state == 'Approaching':
            approach_angle_tol = math.radians(self.get_parameter('approach_angle_tol_deg').value)
            angle_to_leader = math.atan2(p_relative[1], p_relative[0])
            angle_error = normalize_angle(angle_to_leader - follower_yaw)
            log_angle_err = math.degrees(angle_error)
            if abs(angle_error) > approach_angle_tol:
                kp_angle = self.get_parameter('approach_kp_angle').value
                max_rot = self.get_parameter('approach_rot_speed').value
                twist_msg.angular.z = np.clip(-kp_angle * angle_error, -max_rot, max_rot)
            else:
                twist_msg.linear.x = self.get_parameter('approach_fwd_speed').value
        
        elif self.robot_state == 'Following':
            kp_pos = self.get_parameter('follow_kp_pos').value; kp_vel = self.get_parameter('follow_kp_vel').value
            alpha = self.get_parameter('follow_alpha_lpf').value; k_omega = self.get_parameter('follow_k_omega').value
            max_linear = self.get_parameter('max_linear_speed').value; max_angular = self.get_parameter('max_angular_speed').value
            error_dist = current_dist - d_limit; log_dist_err = error_dist
            p_hat = p_relative / current_dist if current_dist > 1e-6 else np.zeros(2)
            v_correct = kp_pos * error_dist * p_hat
            v_follow = kp_vel * v_relative
            v_cmd_raw = v_correct + v_follow
            self.v_cmd_smoothed = alpha * v_cmd_raw + (1 - alpha) * self.v_cmd_smoothed
            cos_th, sin_th = math.cos(follower_yaw), math.sin(follower_yaw)
            rot_matrix_inv = np.array([[cos_th, sin_th], [-sin_th, cos_th]])
            v_cmd_local = rot_matrix_inv @ self.v_cmd_smoothed
            twist_msg.linear.x = np.clip(v_cmd_local[0], 0.0, max_linear)
            twist_msg.angular.z = np.clip(-k_omega * v_cmd_local[1], -max_angular, max_angular)

        self.publisher_.publish(twist_msg)

        if np.linalg.norm(self.v_cmd_smoothed) > 1e-3:
            debug_pose = PoseStamped()
            debug_pose.header.stamp = self.get_clock().now().to_msg()
            debug_pose.header.frame_id = 'world'
            debug_pose.pose.position.x = p_follower[0]; debug_pose.pose.position.y = p_follower[1]
            debug_pose.pose.position.z = 0.1
            cmd_angle = math.atan2(self.v_cmd_smoothed[1], self.v_cmd_smoothed[0])
            q = Rotation.from_euler('z', cmd_angle).as_quat()
            debug_pose.pose.orientation.x = q[0]; debug_pose.pose.orientation.y = q[1]
            debug_pose.pose.orientation.z = q[2]; debug_pose.pose.orientation.w = q[3]
            self.debug_cmd_pub_.publish(debug_pose)
        
        log_msg = (
            f"State: {self.robot_state} | Dist: {current_dist:.2f}m | "
            f"L_Vel: [{v_leader[0]:.2f}, {v_leader[1]:.2f}] | "
            f"Cmd: [{twist_msg.linear.x:.2f}m/s, {twist_msg.angular.z:.2f}rad/s]"
        )
        self.get_logger().info(log_msg)

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

if __name__ == '__main__': main()