# 파일명: path_controller_node.py
"""
Path Controller Node (Corrected Pure Pursuit)

이 노드는 상황에 따라 Point 추종과 Pure Pursuit을 전환하는 하이브리드 제어기입니다.
Pure Pursuit 로직의 치명적인 버그를 수정하여, 로봇의 현재 위치를 기준으로
기하학적으로 올바른 Lookahead Point를 계산합니다.

- 제어 로직 (상황 판단 기반):
  1. 원거리 추종 (Stable Point Following): 경로에서 멀리 떨어져 있을 때, 안정적인
     단일 목표점을 직접 추종합니다.
  2. 근거리 추종 (Corrected Pure Pursuit): 경로에 가까울 때, 로봇 중심의 원과
     경로 선분의 교차점을 계산하여 정확한 Lookahead Point를 찾아 부드럽게 추종합니다.
  3. 안전 거리 유지 (P-Control): 모든 제어 로직의 기반으로, 리더와의 안전 거리를
     유지하며 충돌을 방지합니다.

- 구독 (Subscriptions):
  - /controller/short_term_path (nav_msgs/Path): Path Generator가 생성한 단기 경로.
  - /controller/goal_point (geometry_msgs/PointStamped): Point Post-processor가 생성한 안정적인 목표점.
  - /follower/estimated_pose (geometry_msgs/PoseStamped): 팔로워의 현재 위치 및 자세.

- 발행 (Publications):
  - cmd_vel (geometry_msgs/Twist): 로봇 구동을 위한 속도 명령.
  - /debug/final_target_point (geometry_msgs/PointStamped): 현재 추종 중인 최종 목표점 (디버깅용).

- 파라미터 (Parameters):
  (Docstring은 이전 버전과 동일하므로 생략)
"""
import rclpy, math, numpy as np, message_filters
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist, PoseStamped
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation

def quaternion_to_yaw(q): return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]
def normalize_angle(angle): return (angle + math.pi) % (2 * math.pi) - math.pi

class PathControllerNode(Node):
    def __init__(self):
        super().__init__('path_controller_node')
        self.declare_parameter('follow_dist_thresh', 1.0)
        self.declare_parameter('lookahead_distance', 2.5)
        self.declare_parameter('approach_dist_thresh', 3.0)
        self.declare_parameter('kp_angle', 2.0)
        self.declare_parameter('kp_pos', 0.5)
        self.declare_parameter('approach_max_linear_speed', 1.0)
        self.declare_parameter('follow_max_linear_speed', 0.7)
        self.declare_parameter('max_angular_speed', 1.0)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.debug_goal_pub = self.create_publisher(PointStamped, '/debug/final_target_point', 10)
        self.current_control_mode = 'Init'

        path_sub = message_filters.Subscriber(self, Path, '/controller/short_term_path')
        stable_goal_sub = message_filters.Subscriber(self, PointStamped, '/controller/goal_point')
        follower_pose_sub = message_filters.Subscriber(self, PoseStamped, '/follower/estimated_pose')
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [path_sub, stable_goal_sub, follower_pose_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.control_loop_callback)
        self.get_logger().info('Path Controller Node (Corrected Hybrid) 시작됨.')

    def control_loop_callback(self, path_msg, stable_goal_msg, follower_pose_msg):
        if not path_msg.poses:
            self.cmd_vel_pub.publish(Twist()); return

        p_follower = np.array([follower_pose_msg.pose.position.x, follower_pose_msg.pose.position.y])
        follower_yaw = quaternion_to_yaw(follower_pose_msg.pose.orientation)
        path_points = np.array([[p.pose.position.x, p.pose.position.y] for p in path_msg.poses])
        
        distances_to_path = np.linalg.norm(path_points - p_follower, axis=1)
        min_dist_to_path = np.min(distances_to_path)
        lookahead_dist = self.get_parameter('lookahead_distance').value
        
        final_target = None
        new_mode = ''

        if min_dist_to_path > lookahead_dist:
            final_target = np.array([stable_goal_msg.point.x, stable_goal_msg.point.y])
            new_mode = 'Point Following'
        else:
            # --- 올바르게 수정된 Pure Pursuit 로직 사용 ---
            final_target = self.find_lookahead_point_correct(path_points, p_follower, lookahead_dist)
            new_mode = 'Pure Pursuit'

        if new_mode != self.current_control_mode:
            self.get_logger().info(f"Switching to [{new_mode}] mode.")
            self.current_control_mode = new_mode

        if final_target is None:
            # Lookahead Point를 못 찾은 경우(경로가 너무 짧음 등), 경로의 끝점을 목표로 삼음
            final_target = path_points[-1]

        # ... (이하 제어 로직은 동일)
        current_dist_to_target = np.linalg.norm(final_target - p_follower)
        angle_to_target = math.atan2(final_target[1] - p_follower[1], final_target[0] - p_follower[0])
        angle_error = normalize_angle(angle_to_target - follower_yaw)
        
        follow_dist = self.get_parameter('follow_dist_thresh').value
        if current_dist_to_target < follow_dist:
            self.cmd_vel_pub.publish(Twist()); return

        kp_angle = self.get_parameter('kp_angle').value
        max_angular = self.get_parameter('max_angular_speed').value
        angular_vel = np.clip(kp_angle * angle_error, -max_angular, max_angular)

        approach_dist = self.get_parameter('approach_dist_thresh').value
        follow_max_speed = self.get_parameter('follow_max_linear_speed').value
        approach_max_speed = self.get_parameter('approach_max_linear_speed').value
        dynamic_max_linear_speed = np.interp(
            current_dist_to_target,
            [follow_dist, approach_dist],
            [follow_max_speed, approach_max_speed]
        )
        
        kp_pos = self.get_parameter('kp_pos').value
        dist_error = current_dist_to_target - follow_dist
        base_linear_vel = np.clip(kp_pos * dist_error, 0.0, dynamic_max_linear_speed)
        
        deceleration_factor = max(0.0, math.cos(angle_error))
        linear_vel = base_linear_vel * deceleration_factor

        twist_msg = Twist()
        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self.cmd_vel_pub.publish(twist_msg)

        debug_msg = PointStamped()
        debug_msg.header = path_msg.header
        debug_msg.point.x, debug_msg.point.y = final_target[0], final_target[1]
        self.debug_goal_pub.publish(debug_msg)

    def find_lookahead_point_correct(self, path_points, robot_pos, lookahead_dist):
        """ 기하학적으로 올바른 Pure Pursuit 목표점을 찾습니다. """
        # 경로상의 모든 선분을 순회하며 교차점을 찾음
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]

            # 선분 벡터 계산
            d = p2 - p1
            # 로봇 위치에서 선분 시작점까지의 벡터
            f = p1 - robot_pos
            
            # 원과 선의 교차점을 찾기 위한 2차 방정식의 계수 계산
            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            c = np.dot(f, f) - lookahead_dist**2
            
            discriminant = b**2 - 4*a*c
            
            if discriminant >= 0:
                # 교차점이 존재
                discriminant = math.sqrt(discriminant)
                
                # 두 개의 가능한 해
                t1 = (-b - discriminant) / (2*a)
                t2 = (-b + discriminant) / (2*a)
                
                # 해(t)가 0과 1 사이에 있어야 선분 위에 교차점이 존재
                if 0 <= t1 <= 1:
                    return p1 + t1 * d
                if 0 <= t2 <= 1:
                    return p1 + t2 * d
        
        # 경로 전체에서 교차점을 찾지 못한 경우
        return None

def main(args=None):
    rclpy.init(args=args); node = PathControllerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__': main()