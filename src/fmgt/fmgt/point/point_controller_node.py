# 파일명: point_controller_node.py
"""
Point Controller Node (Unified Control Logic)

이 노드는 Post-processor가 발행한 안정적인 목표 지점을 따라가도록 로봇의 속도를
제어합니다. 기존의 불연속적인 상태 머신(Stop, Approaching, Following, Aligning)을
제거하고, 목표점과의 거리와 각도 오차에 기반한 하나의 통합된 제어 법칙을 사용합니다.
이를 통해 로봇의 움직임을 훨씬 더 부드럽고 예측 가능하게 만듭니다.

- 통합 제어 로직:
  1. 각속도 제어: 목표점을 향한 각도 오차에 비례하여 회전 속도를 결정합니다.
  2. 선속도 제어 (2-Factor):
     a. 동적 속도 스케일링: 목표점과의 거리에 따라 최대 선속도를 동적으로 조절합니다.
        (멀리 있을수록 빠르고, 가까울수록 느리게)
     b. 각도 기반 감속: 목표점을 정면으로 바라보지 않을수록(각도 오차가 클수록)
        선속도를 자동으로 감속시켜, 회전과 전진이 부드럽게 연동되도록 합니다.

- 구독 (Subscriptions):
  - /controller/goal_point (geometry_msgs/PointStamped): 최종 목표 지점
  - /follower/estimated_pose (geometry_msgs/PoseStamped): 팔로워의 현재 위치 및 자세

- 발행 (Publications):
  - cmd_vel (geometry_msgs/Twist): 로봇 구동을 위한 속도 명령
  - /debug/controller_goal (geometry_msgs/PointStamped): 현재 추종 중인 목표 지점 (디버깅용)

- 파라미터 (Parameters):
  - approach_dist_thresh (double): 이 거리(m) 이상부터 최대 선속도로 주행합니다.
  - follow_dist_thresh (double): 이 거리(m) 안으로 들어오면 로봇이 완전히 정지합니다.
  - kp_angle (double): 각도 오차에 대한 비례(P) 제어 게인.
  - kp_pos (double): 거리 오차에 대한 비례(P) 제어 게인.
  - approach_max_linear_speed (double): 로봇이 낼 수 있는 최대 선속도(m/s)입니다.
  - follow_max_linear_speed (double): 목표점에 가까워졌을 때의 목표 선속도(m/s)입니다.
  - follow_max_angular_speed (double): 로봇이 낼 수 있는 최대 각속도(rad/s)입니다.
  - (align_* 파라미터는 더 이상 사용되지 않지만, 호환성을 위해 남겨둘 수 있습니다.)
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
        # --- 파라미터 선언 ---
        self.declare_parameter('approach_dist_thresh', 3.0)
        self.declare_parameter('follow_dist_thresh', 1.2)
        self.declare_parameter('kp_angle', 2.0)
        self.declare_parameter('kp_pos', 0.5)
        self.declare_parameter('approach_max_linear_speed', 1.5)
        self.declare_parameter('approach_max_angular_speed', 1.0) # 각속도는 하나의 max값만 사용
        self.declare_parameter('follow_max_linear_speed', 0.6)

        # --- 상태 머신 변수 제거 ---
        # self.robot_state = 'Stop' ... etc.

        # --- 발행자 및 구독자 ---
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.debug_goal_pub = self.create_publisher(PointStamped, '/debug/controller_goal', 10)

        goal_point_sub = message_filters.Subscriber(self, PointStamped, '/controller/goal_point')
        follower_pose_sub = message_filters.Subscriber(self, PoseStamped, '/follower/estimated_pose')
        
        self.ts = message_filters.ApproximateTimeSynchronizer([goal_point_sub, follower_pose_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.control_loop_callback)
        self.get_logger().info('Point Controller Node (Unified Control) 시작됨.')

    def control_loop_callback(self, goal_point_msg, follower_pose_msg):
        # --- 1. 현재 상태 계산 ---
        goal_point = np.array([goal_point_msg.point.x, goal_point_msg.point.y])
        p_follower = np.array([follower_pose_msg.pose.position.x, follower_pose_msg.pose.position.y])
        follower_yaw = quaternion_to_yaw(follower_pose_msg.pose.orientation)
        
        current_dist_to_goal = np.linalg.norm(goal_point - p_follower)
        angle_to_target = math.atan2(goal_point[1] - p_follower[1], goal_point[0] - p_follower[0])
        angle_error = normalize_angle(angle_to_target - follower_yaw)
        
        # --- 2. 정지 조건 확인 ---
        follow_dist = self.get_parameter('follow_dist_thresh').value
        if current_dist_to_goal < follow_dist:
            self.cmd_vel_pub.publish(Twist()) # 정지 명령 발행
            self.debug_goal_pub.publish(goal_point_msg)
            return

        # --- 3. 통합 제어 로직 ---
        twist_msg = Twist()

        # 3a. 각속도 계산
        kp_angle = self.get_parameter('kp_angle').value
        # Approaching과 Following 상태가 통합되었으므로, 하나의 최대 각속도 값만 사용
        max_angular = self.get_parameter('approach_max_angular_speed').value
        twist_msg.angular.z = np.clip(kp_angle * angle_error, -max_angular, max_angular)

        # 3b. 선속도 계산
        # 거리에 따라 최대 선속도를 동적으로 조절
        approach_dist = self.get_parameter('approach_dist_thresh').value
        follow_max_speed = self.get_parameter('follow_max_linear_speed').value
        approach_max_speed = self.get_parameter('approach_max_linear_speed').value
        
        dynamic_max_linear_speed = np.interp(
            current_dist_to_goal,
            [follow_dist, approach_dist],
            [follow_max_speed, approach_max_speed]
        )
        
        # 거리 오차에 기반한 기본 선속도 계산
        kp_pos = self.get_parameter('kp_pos').value
        dist_error = current_dist_to_goal - follow_dist
        base_linear_vel = np.clip(kp_pos * dist_error, 0.0, dynamic_max_linear_speed)
        
        # 각도 오차에 따라 선속도를 감속시키는 계수 계산 (cos 함수 이용)
        # 정면을 볼수록(angle_error=0) 계수는 1, 90도 이상 벗어나면 계수는 0
        deceleration_factor = max(0.0, math.cos(angle_error))
        
        # 최종 선속도 결정
        twist_msg.linear.x = base_linear_vel * deceleration_factor
        
        # --- 4. 최종 속도 명령 발행 ---
        self.cmd_vel_pub.publish(twist_msg)
        self.debug_goal_pub.publish(goal_point_msg)

def main(args=None):
    rclpy.init(args=args); node = PointControllerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__': main()