# 파일명: path_controller_node.py
"""
Path Controller Node (with Alignment Handling & Anti-Reverse Logic)

이 노드는 Post-processor가 발행한 최종 목표 경로('/robot/goal_trajectory')를 따라가도록
로봇의 속도를 제어합니다. 또한 Leader Estimator로부터 Align 필요 신호를 받아,
경로 추종을 잠시 멈추고 제자리 회전을 수행하여 시스템이 안정화되도록 돕습니다.

- 주요 기능:
  1. Align 신호 수신 및 처리: '/align_needed'가 True이면, 경로 추종을 멈추고 제자리 회전.
  2. 경로 겹침 방지: 로봇이 항상 경로의 진행 방향으로만 목표점을 찾도록 하여 역주행을 방지.
  3. 속도 계산 및 발행: 목표점을 향하도록 'cmd_vel'을 계산하여 발행.

- 구독 (Subscriptions):
  - /robot/goal_trajectory (nav_msgs/Path): 후처리된 최종 목표 경로
  - /follower/estimated_pose (geometry_msgs/PoseStamped): 팔로워의 현재 위치 및 자세
  - /align_needed (std_msgs/Bool): Leader Estimator로부터의 Align Mode 요청 신호

- 발행 (Publications):
  - cmd_vel (geometry_msgs/Twist): 로봇 구동을 위한 속도 명령
  - /debug/target_point (geometry_msgs/PointStamped): 현재 추종 중인 목표 지점 (디버깅용)

- 파라미터 (Parameters):
  - debug_mode (bool): 디버그 로그 및 목표점 발행 여부
  - lookahead_distance (double): 현재 로봇 위치에서 경로상의 목표점을 찾는 탐색 거리 (m)
  - goal_reached_dist (double): 최종 목표점에 도달했다고 판단하는 거리 (m)
  - max_linear_speed (double): 최대 선속도 (m/s)
  - max_angular_speed (double): 최대 각속도 (rad/s)
  - kp_angle (double): 각도 오차에 대한 비례(P) 제어 게인
  - kp_linear (double): 선속도 제어를 위한 비례(P) 제어 게인
  - align_angular_speed (double): Align Mode 중 제자리 회전 각속도 (rad/s)
"""
import rclpy
import numpy as np
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PointStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool
from scipy.spatial.transform import Rotation

def quaternion_to_yaw(q):
    return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]
def normalize_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

class PathControllerNode(Node):
    def __init__(self):
        super().__init__('path_controller_node')
        
        self.declare_parameter('debug_mode', True)
        self.declare_parameter('lookahead_distance', 0.4)
        self.declare_parameter('goal_reached_dist', 0.15)
        self.declare_parameter('max_linear_speed', 0.5)
        self.declare_parameter('max_angular_speed', 1.0)
        self.declare_parameter('kp_angle', 2.5)
        self.declare_parameter('kp_linear', 0.8)
        self.declare_parameter('align_angular_speed', 0.3)

        self.current_path = []
        self.robot_pose = None
        self.is_align_needed = False
        
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        if self.get_parameter('debug_mode').value:
            self.target_point_pub = self.create_publisher(PointStamped, '/debug/target_point', 10)
        
        self.path_sub = self.create_subscription(Path, '/robot/goal_trajectory', self.path_callback, 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/follower/estimated_pose', self.pose_callback, 10)
        self.align_sub = self.create_subscription(Bool, '/align_needed', self.align_callback, 10)

        self.get_logger().info('Path Controller Node (with Alignment Handling) 시작됨.')
    
    def align_callback(self, msg: Bool):
        if msg.data and not self.is_align_needed:
            self.get_logger().warn("Align 신호 수신! 제자리 회전을 시작합니다.")
            self.current_path = []
        elif not msg.data and self.is_align_needed:
            self.get_logger().info("Align 완료 신호 수신! 경로 추종을 재개합니다.")
        self.is_align_needed = msg.data

    def path_callback(self, path_msg: Path):
        if not self.is_align_needed and len(path_msg.poses) > 1:
            self.current_path = path_msg.poses
            self.get_logger().info(f'새로운 경로 수신. 총 {len(self.current_path)}개의 웨이포인트.')

    def pose_callback(self, pose_msg: PoseStamped):
        self.robot_pose = pose_msg
        self.control_loop()

    def control_loop(self):
        if self.robot_pose is None:
            self.stop_robot()
            return
            
        if self.is_align_needed:
            twist_msg = Twist(angular={'z': self.get_parameter('align_angular_speed').value})
            self.cmd_vel_pub.publish(twist_msg)
            return

        if not self.current_path:
            self.stop_robot()
            return
        
        robot_pos = np.array([self.robot_pose.pose.position.x, self.robot_pose.pose.position.y])
        target_point, is_final_goal = self.find_target_waypoint(robot_pos)
        
        if target_point is None:
            self.stop_robot()
            return
        
        goal_reached_dist = self.get_parameter('goal_reached_dist').value
        if is_final_goal and np.linalg.norm(target_point - robot_pos) < goal_reached_dist:
            self.get_logger().info("최종 목표점 도달. 정지합니다.")
            self.stop_robot()
            self.current_path = []
            return
            
        twist_msg = self.calculate_control_commands(robot_pos, target_point)
        self.cmd_vel_pub.publish(twist_msg)

        if self.get_parameter('debug_mode').value:
            self.publish_target_point(target_point)

    def find_target_waypoint(self, robot_pos):
        """
        [로직 수정 완료] 경로 겹침 및 역주행을 방지하는 강건한 목표점 탐색 함수.
        """
        # 경로를 numpy 배열로 변환
        path_points = np.array([[pose.pose.position.x, pose.pose.position.y] for pose in self.current_path])
        
        # 1단계: 경로 위에서 로봇과 가장 가까운 지점(projection)을 찾고, 그 지점이 속한 선분(segment)의 인덱스를 찾음
        closest_dist_sq = float('inf')
        search_start_index = 0
        
        for i in range(len(path_points) - 1):
            p1, p2 = path_points[i], path_points[i+1]
            segment_vec = p2 - p1
            # 벡터의 제곱 길이를 계산하여 불필요한 sqrt 연산 방지
            segment_len_sq = np.dot(segment_vec, segment_vec)
            
            if segment_len_sq < 1e-12:
                # 매우 짧은 선분은 건너뜀
                dist_sq = np.sum((robot_pos - p1)**2)
            else:
                # 로봇 위치를 선분에 투영(projection)
                t = max(0, min(1, np.dot(robot_pos - p1, segment_vec) / segment_len_sq))
                projection = p1 + t * segment_vec
                dist_sq = np.sum((robot_pos - projection)**2)

            if dist_sq < closest_dist_sq:
                closest_dist_sq = dist_sq
                # 탐색을 시작할 인덱스는 가장 가까운 선분의 '끝점' 인덱스
                search_start_index = i + 1

        # 2단계: 위에서 찾은 시작 인덱스부터 경로의 끝까지 순회하며 lookahead_distance를 만족하는 점을 찾음
        lookahead_dist = self.get_parameter('lookahead_distance').value
        for i in range(search_start_index, len(path_points)):
            wp_pos = path_points[i]
            if np.linalg.norm(robot_pos - wp_pos) > lookahead_dist:
                return wp_pos, (i == len(path_points) - 1)
        
        # 경로 끝까지 적절한 점을 못찾으면 마지막 점을 목표로 함
        return path_points[-1], True

    def calculate_control_commands(self, robot_pos, target_point):
        robot_yaw = quaternion_to_yaw(self.robot_pose.pose.orientation)
        angle_to_target = math.atan2(target_point[1] - robot_pos[1], target_point[0] - robot_pos[0])
        angle_error = normalize_angle(angle_to_target - robot_yaw)
        
        kp_angle = self.get_parameter('kp_angle').value
        max_angular = self.get_parameter('max_angular_speed').value
        max_linear = self.get_parameter('max_linear_speed').value
        
        angular_z = np.clip(kp_angle * angle_error, -max_angular, max_angular)
        
        angle_err_abs = abs(angle_error)
        speed_reduction = max(0.0, math.cos(angle_err_abs)) if angle_err_abs < math.pi / 2 else 0.0
        
        # 만약 거리 비례 제어를 원하면 아래 주석 처리된 코드를 사용할 수 있음.
        # dist_to_target = np.linalg.norm(target_point - robot_pos)
        # linear_x = np.clip(self.get_parameter('kp_linear').value * dist_to_target * speed_reduction, 0.0, max_linear)
        linear_x = max_linear * speed_reduction

        twist_msg = Twist()
        twist_msg.linear.x, twist_msg.angular.z = linear_x, angular_z
        return twist_msg

    def stop_robot(self):
        self.cmd_vel_pub.publish(Twist())

    def publish_target_point(self, target_point):
        point_msg = PointStamped()
        point_msg.header.frame_id = "world"
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.point.x, point_msg.point.y, point_msg.point.z = target_point[0], target_point[1], 0.1
        self.target_point_pub.publish(point_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PathControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info('노드 종료. 로봇 정지.')
        node.stop_robot()
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()