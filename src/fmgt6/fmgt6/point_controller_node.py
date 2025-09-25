# 파일명: point_controller_node.py
"""
Point Controller Node

이 노드는 Post-processor가 발행한 안정적인 목표 지점을 따라가도록 로봇의 속도를
제어하는 역할만 담당합니다. 내부 상태 머신(Stop, Approaching, Following, Aligning)을
기반으로 상황에 맞는 속도(cmd_vel)를 계산하여 발행합니다.

- 구독 (Subscriptions):
  - /controller/goal_point (geometry_msgs/PointStamped): 최종 목표 지점
  - /follower/estimated_pose (geometry_msgs/PoseStamped): 팔로워의 현재 위치 및 자세

- 발행 (Publications):
  - cmd_vel (geometry_msgs/Twist): 로봇 구동을 위한 속도 명령
  - /debug/controller_goal (geometry_msgs/PointStamped): 현재 추종 중인 목표 지점 (디버깅용)

- 파라미터 (Parameters):
  - approach_dist_thresh (double): 'Approaching' 상태로 진입하는 거리 임계값 (m)
  - follow_dist_thresh (double): 목표 추종 거리 (m). 이보다 가까워지면 'Stop' 상태가 됨.
  - kp_angle (double): 각도 오차에 대한 비례(P) 제어 게인
  - kp_pos (double): 거리 오차에 대한 비례(P) 제어 게인
  - approach_max_linear_speed (double): 'Approaching' 상태에서의 최대 선속도 (m/s)
  - approach_max_angular_speed (double): 'Approaching' 상태에서의 최대 각속도 (rad/s)
  - follow_max_linear_speed (double): 'Following' 상태에서의 최대 선속도 (m/s)
  - follow_max_angular_speed (double): 'Following' 상태에서의 최대 각속도 (rad/s)
  - align_rot_speed (double): 'Aligning' 상태에서의 회전 속도 (rad/s)
  - align_angle_complete_deg (double): 'Aligning'을 완료했다고 판단하는 각도 오차 (degrees)
  - align_trigger_angle_deg (double): 'Aligning' 상태로 진입하는 각도 오차 (degrees)
"""
import rclpy, math, numpy as np, message_filters
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist, PoseStamped
from scipy.spatial.transform import Rotation

def quaternion_to_yaw(q): return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]
def normalize_angle(angle): return (angle + math.pi) % (2 * math.pi) - math.pi

class PointControllerNode(Node):
    def __init__(self):
        super().__init__('point_controller_node')
        self.declare_parameter('approach_dist_thresh', 3.0)
        self.declare_parameter('follow_dist_thresh', 1.2)
        self.declare_parameter('kp_angle', 2.0)
        self.declare_parameter('kp_pos', 0.5)
        self.declare_parameter('approach_max_linear_speed', 3.0)
        self.declare_parameter('approach_max_angular_speed', 0.5)
        self.declare_parameter('follow_max_linear_speed', 1.0)
        self.declare_parameter('follow_max_angular_speed', 0.2)
        self.declare_parameter('align_rot_speed', 1.0)
        self.declare_parameter('align_angle_complete_deg', 3.0)
        self.declare_parameter('align_trigger_angle_deg', 30.0)

        self.robot_state = 'Stop'; self.state_before_align = 'Stop'; self.align_target_point = None

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.debug_goal_pub = self.create_publisher(PointStamped, '/debug/controller_goal', 10)

        goal_point_sub = message_filters.Subscriber(self, PointStamped, '/controller/goal_point')
        follower_pose_sub = message_filters.Subscriber(self, PoseStamped, '/follower/estimated_pose')
        
        self.ts = message_filters.ApproximateTimeSynchronizer([goal_point_sub, follower_pose_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.control_loop_callback)
        self.get_logger().info('Point Controller Node (Control Logic Only) 시작됨.')

    def control_loop_callback(self, goal_point_msg, follower_pose_msg):
        goal_point = np.array([goal_point_msg.point.x, goal_point_msg.point.y])
        p_follower = np.array([follower_pose_msg.pose.position.x, follower_pose_msg.pose.position.y])
        follower_yaw = quaternion_to_yaw(follower_pose_msg.pose.orientation)
        
        current_dist_to_goal = np.linalg.norm(goal_point - p_follower)
        target_point_for_angle = self.align_target_point if self.robot_state == 'Aligning' else goal_point
        angle_to_target = math.atan2(target_point_for_angle[1] - p_follower[1], target_point_for_angle[0] - p_follower[0])
        angle_error = normalize_angle(angle_to_target - follower_yaw)
        
        align_trigger_rad = math.radians(self.get_parameter('align_trigger_angle_deg').value)
        approach_dist = self.get_parameter('approach_dist_thresh').value
        follow_dist = self.get_parameter('follow_dist_thresh').value
        
        if self.robot_state == 'Aligning':
            if abs(angle_error) < math.radians(self.get_parameter('align_angle_complete_deg').value): self.robot_state = self.state_before_align
        else:
            if abs(angle_error) > align_trigger_rad:
                self.state_before_align = self.robot_state if self.robot_state != 'Stop' else 'Following'
                self.robot_state, self.align_target_point = 'Aligning', goal_point.copy()
            elif current_dist_to_goal > approach_dist: self.robot_state = 'Approaching'
            elif current_dist_to_goal < follow_dist: self.robot_state = 'Stop'
            else: self.robot_state = 'Following'

        twist_msg = Twist()
        if self.robot_state == 'Aligning':
            max_rot = self.get_parameter('align_rot_speed').value
            twist_msg.angular.z = np.clip(self.get_parameter('kp_angle').value * angle_error, -max_rot, max_rot)
        elif self.robot_state in ['Approaching', 'Following']:
            dist_error = current_dist_to_goal - follow_dist
            if self.robot_state == 'Approaching':
                max_linear = self.get_parameter('approach_max_linear_speed').value
                max_angular = self.get_parameter('approach_max_angular_speed').value
            else: # Following
                max_linear = self.get_parameter('follow_max_linear_speed').value
                max_angular = self.get_parameter('follow_max_angular_speed').value
            
            twist_msg.linear.x = np.clip(self.get_parameter('kp_pos').value * dist_error, 0.0, max_linear)
            twist_msg.angular.z = np.clip(self.get_parameter('kp_angle').value * angle_error, -max_angular, max_angular)
        
        self.cmd_vel_pub.publish(twist_msg)
        self.debug_goal_pub.publish(goal_point_msg)

def main(args=None):
    rclpy.init(args=args); node = PointControllerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__': main()