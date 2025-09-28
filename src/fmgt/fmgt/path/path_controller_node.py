# 파일명: path_controller_node.py
"""
Path Controller Node

이 노드는 Post-processor가 발행한 최종 목표 경로('/robot/goal_trajectory')를 따라가도록
로봇의 속도를 제어합니다. Look-ahead Distance 기반의 제어 방식을 사용하여 경로를 부드럽게
추종합니다.

- 주요 기능:
  1. 경로 수신 및 현재 로봇 위치에서 일정 거리(lookahead_distance) 앞의 목표점 탐색
  2. 로봇의 현재 위치와 목표점 사이의 거리 및 각도 오차 계산
  3. P 제어 기반으로 로봇의 선속도/각속도('cmd_vel') 계산 및 발행

- 구독 (Subscriptions):
  - /robot/goal_trajectory (nav_msgs/Path): 후처리된 최종 목표 경로
  - /follower/estimated_pose (geometry_msgs/PoseStamped): 팔로워의 현재 위치 및 자세

- 발행 (Publications):
  - cmd_vel (geometry_msgs/Twist): 로봇 구동을 위한 속도 명령
  - /debug/target_point (visualization_msgs/Marker): 현재 추종 중인 목표 지점 (디버깅용)

- 파라미터 (Parameters):
  - debug_mode (bool): 디버그 로그 및 Rviz 마커 발행 여부
  - lookahead_distance (double): 현재 로봇 위치에서 경로상의 목표점을 찾는 탐색 거리 (m)
  - goal_reached_dist (double): 최종 목표점에 도달했다고 판단하는 거리 (m)
  - max_linear_speed (double): 최대 선속도 (m/s)
  - max_angular_speed (double): 최대 각속도 (rad/s)
  - kp_angle (double): 각도 오차에 대한 비례(P) 제어 게인
"""
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
    """geometry_msgs/Quaternion 메시지를 Yaw 각도(라디안)로 변환합니다."""
    return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]

def normalize_angle(angle):
    """각도를 -pi 에서 +pi 사이로 정규화합니다."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

class PathControllerNode(Node):
    def __init__(self):
        super().__init__('path_controller_node')
        
        # --- 파라미터 선언 ---
        self.declare_parameter('debug_mode', True)
        self.declare_parameter('lookahead_distance', 0.4)
        self.declare_parameter('goal_reached_dist', 0.15)
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('kp_angle', 2.5)

        # --- 상태 변수 초기화 ---
        self.current_path = []
        self.robot_pose = None
        
        # --- 발행자 ---
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        if self.get_parameter('debug_mode').value:
            self.marker_pub = self.create_publisher(Marker, '/robot/target_point', 10)
        
        # --- 구독자 ---
        # QoS 문제를 해결하고 토픽 이름을 아키텍처에 맞게 수정
        self.path_sub = self.create_subscription(Path, '/robot/goal_trajectory', self.path_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/follower/estimated_pose', self.pose_callback, 10)

        self.get_logger().info('Path Controller Node (Refactored) 시작됨.')

    def path_callback(self, path_msg: Path):
        """새로운 경로를 수신하면 내부 상태를 업데이트합니다."""
        if len(path_msg.poses) > 1:
            self.current_path = path_msg.poses
            self.get_logger().info(f'새로운 경로 수신. 총 {len(self.current_path)}개의 웨이포인트.')
        else:
            self.current_path = [] # 너무 짧은 경로는 무시하여 오작동 방지

    def pose_callback(self, pose_msg: PoseStamped):
        """로봇의 새로운 포즈를 수신할 때마다 메인 제어 루프를 실행합니다."""
        self.robot_pose = pose_msg
        self.control_loop()

    def control_loop(self):
        """메인 제어 로직을 관리하는 함수."""
        if self.robot_pose is None or not self.current_path:
            self.stop_robot()
            return
        
        robot_pos = np.array([self.robot_pose.pose.position.x, self.robot_pose.pose.position.y])
        
        target_point, is_final_goal = self.find_target_waypoint(robot_pos)
        
        if target_point is None:
            self.stop_robot() # 경로 상에 더 이상 따라갈 점이 없음
            return
        
        # 최종 목표점에 거의 도달했는지 확인
        goal_reached_dist = self.get_parameter('goal_reached_dist').value
        if is_final_goal and np.linalg.norm(target_point - robot_pos) < goal_reached_dist:
            self.get_logger().info("최종 목표점 도달. 정지합니다.")
            self.stop_robot()
            self.current_path = [] # 경로 초기화
            return
            
        twist_msg = self.calculate_control_commands(robot_pos, target_point)
        self.cmd_vel_pub.publish(twist_msg)

        if self.get_parameter('debug_mode').value:
            self.publish_marker(target_point)
            # 디버깅 로그 추가
            log_msg = (f"[Control] Target:({target_point[0]:.2f},{target_point[1]:.2f}) | "
                       f"Path left:{len(self.current_path)} | "
                       f"Cmd:[{twist_msg.linear.x:.2f}m/s, {twist_msg.angular.z:.2f}rad/s]")
            self.get_logger().info(log_msg, throttle_duration_sec=0.2)


    def find_target_waypoint(self, robot_pos):
        """경로 상에서 현재 로봇이 따라가야 할 목표 지점(look-ahead point)을 찾습니다."""
        lookahead_dist = self.get_parameter('lookahead_distance').value
        
        # 가장 가까운 웨이포인트부터 탐색 시작
        min_dist = float('inf')
        closest_idx = -1
        for i, pose in enumerate(self.current_path):
            wp_pos = np.array([pose.pose.position.x, pose.pose.position.y])
            dist = np.linalg.norm(robot_pos - wp_pos)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 가장 가까운 점부터 시작해서 lookahead_distance보다 멀어지는 첫 점을 찾음
        for i in range(closest_idx, len(self.current_path)):
            pose = self.current_path[i]
            wp_pos = np.array([pose.pose.position.x, pose.pose.position.y])
            dist_to_wp = np.linalg.norm(robot_pos - wp_pos)

            if dist_to_wp > lookahead_dist:
                is_final = (i == len(self.current_path) - 1)
                return wp_pos, is_final
        
        # 경로 끝까지 탐색했는데도 적절한 점을 못찾으면(경로가 짧거나 거의 도달) 마지막 점을 목표로 함
        last_pose = self.current_path[-1]
        last_wp_pos = np.array([last_pose.pose.position.x, last_pose.pose.position.y])
        return last_wp_pos, True

    def calculate_control_commands(self, robot_pos, target_point):
        """로봇의 현재 위치와 목표 지점을 바탕으로 Twist 메시지를 계산합니다."""
        robot_yaw = quaternion_to_yaw(self.robot_pose.pose.orientation)
        
        angle_to_target = math.atan2(target_point[1] - robot_pos[1], target_point[0] - robot_pos[0])
        angle_error = normalize_angle(angle_to_target - robot_yaw)
        
        kp_angle = self.get_parameter('kp_angle').value
        max_angular = self.get_parameter('max_angular_speed').value
        max_linear = self.get_parameter('max_linear_speed').value

        angular_z = np.clip(kp_angle * angle_error, -max_angular, max_angular)

        # 각도 오차가 크면 속도를 줄여 안정적으로 회전하도록 함
        angle_err_abs = abs(angle_error)
        # pi/2 (90도) 이상 차이나면 전진하지 않고 제자리 회전
        speed_reduction = max(0.0, math.cos(angle_err_abs)) if angle_err_abs < math.pi / 2 else 0.0
        linear_x = max_linear * speed_reduction

        twist_msg = Twist()
        twist_msg.linear.x = linear_x
        twist_msg.angular.z = angular_z
        return twist_msg

    def stop_robot(self):
        """로봇을 정지시키는 Twist 메시지를 발행합니다."""
        self.cmd_vel_pub.publish(Twist())

    def publish_marker(self, target_point):
        """디버깅을 위해 Rviz에 현재 목표점을 마커로 표시합니다."""
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "target_waypoint"; marker.id = 0
        marker.type = Marker.SPHERE; marker.action = Marker.ADD
        marker.pose.position.x = target_point[0]
        marker.pose.position.y = target_point[1]
        marker.pose.position.z = 0.1
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.25; marker.scale.y = 0.25; marker.scale.z = 0.25
        marker.color.a = 0.8; marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = PathControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('키보드 인터럽트 감지됨.')
    finally:
        node.get_logger().info('노드 종료. 로봇 정지.')
        node.stop_robot() # 노드 종료 시 안전하게 로봇 정지
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()