# 파일명: follower_control_node.py
import rclpy
import numpy as np
import math
import time
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker

def quaternion_to_yaw(q):
    try: return Rotation.from_quat([q.x,q.y,q.z,q.w]).as_euler('zyx',degrees=False)[0]
    except Exception: return 0.0
def normalize_angle(angle): return (angle+math.pi)%(2*math.pi)-math.pi

class FollowerControlNode(Node):
    def __init__(self):
        super().__init__('follower_control_node')
        
        # --- 파라미터 선언 ---
        self.declare_parameter('debug_mode', True)
        # ★★★ 사용자가 제안한 로직을 위한 핵심 파라미터 ★★★
        self.declare_parameter('skip_distance_threshold', 1.0) 
        self.declare_parameter('max_linear_speed', 2.0)
        self.declare_parameter('max_angular_speed', 0.8)
        self.declare_parameter('goal_reached_dist', 0.8)
        self.declare_parameter('kp_angle', 2.5)
        self.declare_parameter('kp_pos', 1.0)

        self.debug_mode = self.get_parameter('debug_mode').value
        self.current_path = []
        self.robot_pose = None
        
        # --- 발행자 ---
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.marker_pub = self.create_publisher(Marker, '/debug/target_point', 10)
        
        # --- 구독자 ---
        self.path_sub = self.create_subscription(Path, '/leader/control_trajectory', self.path_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/follower/estimated_pose', self.pose_callback, 10)

        self.get_logger().info(f'Simple Waypoint Follower (User Logic) 시작됨.')

    def path_callback(self, control_path_msg):
        self.current_path = control_path_msg.poses
    
    def pose_callback(self, follower_pose_msg):
        self.robot_pose = follower_pose_msg
        self.control_loop()

    def control_loop(self):
        if self.robot_pose is None or not self.current_path:
            self.stop_robot()
            return
        
        robot_pos = np.array([self.robot_pose.pose.position.x, self.robot_pose.pose.position.y])
        robot_yaw = quaternion_to_yaw(self.robot_pose.pose.orientation)
        
        # --- 사용자가 제안한 "거리 기반 스킵" 로직 ---
        skip_dist = self.get_parameter('skip_distance_threshold').value
        
        # 경로의 앞에서부터 웨이포인트를 확인하며, 스킵할 웨이포인트를 제거
        while len(self.current_path) > 1: # 마지막 웨이포인트는 남겨둠
            wp_to_check = self.current_path[0]
            wp_pos = np.array([wp_to_check.pose.position.x, wp_to_check.pose.position.y])
            dist_to_wp = np.linalg.norm(robot_pos - wp_pos)
            
            if dist_to_wp < skip_dist:
                if self.debug_mode:
                    self.get_logger().warn(f"Waypoint ({wp_pos[0]:.2f}, {wp_pos[1]:.2f}) is too close ({dist_to_wp:.2f}m < {skip_dist:.2f}m). SKIPPING.")
                self.current_path.pop(0) # 너무 가까우면 경로에서 제거
            else:
                # 거리가 충분히 먼 첫번째 웨이포인트를 찾으면 루프 중단
                break
        
        # 따라갈 목표 웨이포인트는 항상 경로의 첫번째
        target_wp_pose = self.current_path[0]
        target_wp = np.array([target_wp_pose.pose.position.x, target_wp_pose.pose.position.y])
        
        dist_to_target = np.linalg.norm(robot_pos - target_wp)

        # 최종 목표 도달 확인 (경로에 점이 하나만 남았을 때)
        if len(self.current_path) == 1 and dist_to_target < self.get_parameter('goal_reached_dist').value:
            self.stop_robot()
            if self.debug_mode: self.get_logger().info("Final waypoint reached. Stopping.")
            return

        # --- P 제어 로직 ---
        angle_to_target = math.atan2(target_wp[1] - robot_pos[1], target_wp[0] - robot_pos[0])
        angle_error = normalize_angle(angle_to_target - robot_yaw)
        
        max_angular = self.get_parameter('max_angular_speed').value
        max_linear = self.get_parameter('max_linear_speed').value
        kp_angle = self.get_parameter('kp_angle').value
        kp_pos = self.get_parameter('kp_pos').value

        angular_z = np.clip(kp_angle * angle_error, -max_angular, max_angular)
        
        angle_err_abs = abs(angle_error)
        # 방향 오차가 90도 이상이면 전진 안함
        speed_reduction = max(0.0, math.cos(angle_err_abs)) if angle_err_abs < math.pi / 2 else 0.0
        
        linear_x = np.clip(kp_pos * dist_to_target * speed_reduction, 0.0, max_linear)
        
        twist_msg = Twist()
        twist_msg.linear.x = linear_x
        twist_msg.angular.z = angular_z
        self.publisher_.publish(twist_msg)

        if self.debug_mode:
            self.publish_marker(target_wp)
            log_msg = (f"[Control] Target WP:({target_wp[0]:.2f},{target_wp[1]:.2f}) | "
                       f"Dist:{dist_to_target:.2f}m | AngleErr:{math.degrees(angle_error):.1f}deg | "
                       f"Cmd:[{linear_x:.2f}m/s, {angular_z:.2f}rad/s]")
            self.get_logger().info(log_msg, throttle_duration_sec=0.2)

    def stop_robot(self):
        self.publisher_.publish(Twist())
    
    def publish_marker(self, target_point):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target_waypoint"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = target_point[0]
        marker.pose.position.y = target_point[1]
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25
        marker.scale.y = 0.25
        marker.scale.z = 0.25
        marker.color.a = 0.8
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0 # Green
        self.marker_pub.publish(marker)

    def shutdown_sequence(self):
        self.get_logger().info('Controller 종료 시퀀스 시작...')
        stop_msg=Twist()
        for _ in range(5):
            if rclpy.ok(): self.publisher_.publish(stop_msg); time.sleep(0.02)
        self.get_logger().info('로봇 정지 명령 발행 완료.')

def main(args=None):
    rclpy.init(args=args)
    node = FollowerControlNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('키보드 인터럽트 감지됨. 종료 시퀀스 실행.')
        node.shutdown_sequence()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()