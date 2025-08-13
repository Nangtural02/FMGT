# fmgt4/follower_control_node.py
import rclpy, math, time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, TwistStamped, Twist
from sensor_msgs.msg import Imu
import message_filters

def quaternion_to_yaw(q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class FollowerControlNodeV2(Node):
    def __init__(self):
        super().__init__('follower_control_node_v2')
        # 파라미터는 기존 코드와 동일하게 유지
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

        # 내부 상태 변수
        self.robot_state = 'Stop'
        self.v_cmd_smoothed = np.zeros(2)
        # ★★★ 변경점: 리더 속도 추정을 위한 변수 추가 ★★★
        self.prev_leader_pos = None
        self.prev_leader_timestamp = None

        # ★★★ 변경점: EKF 출력에 맞게 구독자 재구성 ★★★
        # /leader/estimated_velocity는 더 이상 사용하지 않음
        leader_pos_sub = message_filters.Subscriber(self, PointStamped, '/leader/estimated_position')
        follower_pos_sub = message_filters.Subscriber(self, PointStamped, '/follower/estimated_position')
        follower_vel_sub = message_filters.Subscriber(self, TwistStamped, '/follower/estimated_velocity')
        follower_imu_sub = message_filters.Subscriber(self, Imu, '/imu/data')  # Yaw를 위해 유지

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [leader_pos_sub, follower_pos_sub, follower_vel_sub, follower_imu_sub],
            queue_size=10, slop=0.2
        )
        self.ts.registerCallback(self.control_loop_callback)

        # 발행자
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.get_logger().info('Follower Control Node (EKF Compatible) 시작됨.')

    def control_loop_callback(self, l_pos_msg, f_pos_msg, f_vel_msg, imu_msg):
        # 파라미터 가져오기 (기존과 동일)
        d_limit = self.get_parameter('d_limit').get_parameter_value().double_value
        d_hyst = self.get_parameter('d_hysteresis').get_parameter_value().double_value

        # 상태 계산
        p_leader = np.array([l_pos_msg.point.x, l_pos_msg.point.y])
        p_follower = np.array([f_pos_msg.point.x, f_pos_msg.point.y])
        v_follower = np.array([f_vel_msg.twist.linear.x, f_vel_msg.twist.linear.y])
        follower_yaw = quaternion_to_yaw(imu_msg.orientation)

        # ★★★ 변경점: 리더 속도를 위치 변화로부터 직접 추정 ★★★
        current_time = l_pos_msg.header.stamp.sec + l_pos_msg.header.stamp.nanosec * 1e-9
        if self.prev_leader_pos is not None and self.prev_leader_timestamp is not None:
            dt = current_time - self.prev_leader_timestamp
            if dt > 1e-6:  # 0으로 나누기 방지
                v_leader = (p_leader - self.prev_leader_pos) / dt
            else:
                v_leader = np.zeros(2)
        else:
            v_leader = np.zeros(2)
        self.prev_leader_pos = p_leader
        self.prev_leader_timestamp = current_time

        p_relative = p_leader - p_follower
        v_relative = v_leader - v_follower
        current_dist = np.linalg.norm(p_relative)

        # 상태 전환 로직 (기존과 동일)
        d_upper = d_limit + d_hyst
        d_lower = d_limit - d_hyst
        is_leader_stopped = np.linalg.norm(v_leader) < 0.05

        if self.robot_state != 'Approaching' and current_dist > d_upper:
            self.robot_state = 'Approaching'
        elif current_dist < d_lower and is_leader_stopped:
            self.robot_state = 'Stop'
        elif self.robot_state != 'Stop' or current_dist > d_lower:
            self.robot_state = 'Following'

        # 행동 결정 (기존 로직 대부분 재사용)
        twist_msg = Twist()
        log_angle_err, log_dist_err = 0.0, 0.0

        if self.robot_state == 'Approaching':
            approach_angle_tol = math.radians(
                self.get_parameter('approach_angle_tol_deg').get_parameter_value().double_value)
            angle_to_leader = math.atan2(p_relative[1], p_relative[0])
            angle_error = normalize_angle(angle_to_leader - follower_yaw)
            log_angle_err = math.degrees(angle_error)

            if abs(angle_error) > approach_angle_tol:
                approach_kp_angle = self.get_parameter('approach_kp_angle').get_parameter_value().double_value
                approach_rot_speed = self.get_parameter('approach_rot_speed').get_parameter_value().double_value
                angular_vel = approach_kp_angle * angle_error  # 부호는 실제 로봇 방향에 맞게 튜닝
                twist_msg.angular.z = np.clip(angular_vel, -approach_rot_speed, approach_rot_speed)
            else:
                twist_msg.linear.x = self.get_parameter('approach_fwd_speed').get_parameter_value().double_value

        elif self.robot_state == 'Following':
            kp_pos = self.get_parameter('follow_kp_pos').get_parameter_value().double_value
            kp_vel = self.get_parameter('follow_kp_vel').get_parameter_value().double_value
            alpha = self.get_parameter('follow_alpha_lpf').get_parameter_value().double_value
            k_omega = self.get_parameter('follow_k_omega').get_parameter_value().double_value
            max_linear = self.get_parameter('max_linear_speed').get_parameter_value().double_value
            max_angular = self.get_parameter('max_angular_speed').get_parameter_value().double_value

            error_dist = current_dist - d_limit
            log_dist_err = error_dist

            p_hat = p_relative / current_dist if current_dist > 1e-6 else np.zeros(2)

            v_correct = kp_pos * error_dist * p_hat
            v_follow = kp_vel * v_relative
            v_cmd_raw = v_correct + v_follow
            self.v_cmd_smoothed = alpha * v_cmd_raw + (1 - alpha) * self.v_cmd_smoothed

            cos_th, sin_th = math.cos(follower_yaw), math.sin(follower_yaw)
            rot_matrix_inv = np.array([[cos_th, sin_th], [-sin_th, cos_th]])
            v_cmd_local = rot_matrix_inv @ self.v_cmd_smoothed

            twist_msg.linear.x = v_cmd_local[0]
            twist_msg.angular.z = k_omega * v_cmd_local[1]  # 부호는 실제 로봇 방향에 맞게 튜닝

            twist_msg.linear.x = np.clip(twist_msg.linear.x, 0.0, max_linear)
            twist_msg.angular.z = np.clip(twist_msg.angular.z, -max_angular, max_angular)

        # 'Stop' 상태에서는 twist_msg가 비어 있으므로 자동으로 정지

        self.publisher_.publish(twist_msg)

        log_msg = (
            f"State: {self.robot_state} | "
            f"Dist: {current_dist:.2f}m | DistErr: {log_dist_err:.2f}m | "
            f"AngleErr: {log_angle_err:.1f}deg | "
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
    node = FollowerControlNodeV2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_sequence(); node.destroy_node(); rclpy.try_shutdown()


if __name__ == '__main__': main()