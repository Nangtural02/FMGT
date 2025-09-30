# 파일명: robot_simulator_node.py
"""
Robot Simulator Node

이 노드는 리더(Leader)와 팔로워(Follower) 로봇의 움직임을 시뮬레이션하고,
센서 데이터(UWB, Pose)와 Ground Truth 정보를 발행합니다.

- 주요 기능:
  1. 차동 구동 방식의 팔로워 로봇과 직교 좌표계 이동 방식의 리더 시뮬레이션
  2. 팔로워의 추정 위치('/follower/estimated_pose') 발행
  3. 두 로봇 간의 거리를 시뮬레이션한 UWB 데이터('/raw_uwb_distances') 발행
  4. 디버깅을 위한 리더의 실제 위치('/leader/ground_truth') 및 전체 경로('/debug/ground_truth_path') 발행

- 구독 (Subscriptions):
  - cmd_vel (geometry_msgs/Twist): 팔로워 로봇의 속도 명령
  - leader_teleop/cmd_vel (geometry_msgs/Twist): 리더 로봇의 속도 명령 (키보드 조작용)

- 발행 (Publications):
  - /follower/estimated_pose (geometry_msgs/PoseStamped): 팔로워의 시뮬레이션된 위치 및 자세
  - raw_uwb_distances (geometry_msgs/PointStamped): 노이즈가 포함된 시뮬레이션 UWB 거리
  - /leader/ground_truth (geometry_msgs/PointStamped): 리더의 현재 실제 위치 (Ground Truth)
  - /debug/ground_truth_path (nav_msgs/Path): RDP 알고리즘으로 단순화된 리더의 전체 실제 경로

- 파라미터 (Parameters):
  - sim_rate (double): 시뮬레이션 루프의 실행 주기 (Hz)
  - anchor_forward_offset (double): 팔로워 중심에서 UWB 앵커까지의 전방 거리 (m)
  - anchor_width (double): 팔로워의 두 UWB 앵커 사이의 폭 (m)
  - init_follower_pose (list[double]): 팔로워의 초기 위치 및 방향 [x, y, yaw]
  - init_leader_pose (list[double]): 리더의 초기 위치 및 방향 [x, y, yaw]
  - uwb_noise_std (double): UWB 거리 측정에 추가할 노이즈의 표준 편차
  - path_publish_period_sec (double): Ground Truth 경로를 발행하는 주기 (초)
  - rdp_epsilon (double): Ground Truth 경로 단순화(RDP)를 위한 임계값 (m)
"""
import rclpy
import numpy as np
import math
import threading
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion, Twist, PointStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation

def normalize_angle(angle):
    """각도를 -pi 에서 +pi 사이로 정규화합니다."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def yaw_to_quaternion(yaw):
    """Yaw 각도(라디안)를 geometry_msgs/Quaternion 메시지로 변환합니다."""
    q = Rotation.from_euler('z', yaw).as_quat()
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

class RobotSimulatorNode(Node):
    def __init__(self):
        super().__init__('robot_simulator_node')

        # --- 파라미터 선언 ---
        self.declare_parameter('sim_rate', 20.0)
        self.declare_parameter('anchor_forward_offset', 0.25)
        self.declare_parameter('anchor_width', 0.4)
        self.declare_parameter('init_follower_pose', [0.0, 0.0, 0.0])
        self.declare_parameter('init_leader_pose', [1.5, 0.0, 0.0])
        self.declare_parameter('uwb_noise_std', 0.05)
        self.declare_parameter('path_publish_period_sec', 1.0)
        self.declare_parameter('rdp_epsilon', 0.05)

        # --- 상태 변수 초기화 ---
        self.follower_pose = np.array(self.get_parameter('init_follower_pose').value)
        self.leader_pose = np.array(self.get_parameter('init_leader_pose').value)
        self.follower_cmd_vel = Twist()
        self.leader_cmd_vel = Twist()
        self.gt_path_history = []
        self.lock = threading.Lock()

        # --- 발행자 ---
        self.follower_pose_pub = self.create_publisher(PoseStamped, '/follower/estimated_pose', 10)
        self.uwb_pub = self.create_publisher(PointStamped, 'raw_uwb_distances', 10)
        self.gt_leader_pub = self.create_publisher(PointStamped, '/leader/ground_truth', 10)
        self.gt_path_pub = self.create_publisher(Path, '/debug/ground_truth_path', 10)

        # --- 구독자 ---
        self.create_subscription(Twist, 'cmd_vel', self.follower_cmd_callback, 10)
        self.create_subscription(Twist, 'leader_teleop/cmd_vel', self.leader_cmd_callback, 10)

        # --- 타이머 ---
        sim_rate = self.get_parameter('sim_rate').value
        self.dt = 1.0 / sim_rate
        self.sim_timer = self.create_timer(self.dt, self.simulation_loop)
        
        path_pub_period = self.get_parameter('path_publish_period_sec').value
        self.path_timer = self.create_timer(path_pub_period, self.publish_gt_path)

        self.get_logger().info("Robot Simulator (Refactored) 시작됨.")

    def follower_cmd_callback(self, msg):
        with self.lock: self.follower_cmd_vel = msg

    def leader_cmd_callback(self, msg):
        with self.lock: self.leader_cmd_vel = msg
        
    def _update_diff_drive_pose(self, pose, cmd_vel, dt):
        """차동 구동 로봇의 포즈를 업데이트합니다."""
        vx, wz = cmd_vel.linear.x, cmd_vel.angular.z
        pose[0] += vx * math.cos(pose[2]) * dt
        pose[1] += vx * math.sin(pose[2]) * dt
        pose[2] = normalize_angle(pose[2] + wz * dt)
        return pose

    def _update_cartesian_pose(self, pose, cmd_vel, dt):
        """직교 좌표계(Cartesian) 모델의 포즈를 업데이트합니다."""
        pose[0] += cmd_vel.linear.x * dt
        pose[1] += cmd_vel.linear.y * dt
        return pose

    def simulation_loop(self):
        """메인 시뮬레이션 루프."""
        with self.lock:
            now = self.get_clock().now()
            
            # 1. 포즈 업데이트
            self.leader_pose = self._update_cartesian_pose(self.leader_pose, self.leader_cmd_vel, self.dt)
            self.follower_pose = self._update_diff_drive_pose(self.follower_pose, self.follower_cmd_vel, self.dt)
            
            # Ground Truth 경로 기록
            self.gt_path_history.append(self.leader_pose[:2].copy())

            # 2. 센서 데이터 및 Ground Truth 발행
            self._publish_follower_pose(now)
            self._publish_uwb_distances(now)
            self._publish_leader_ground_truth(now)
    
    def _publish_follower_pose(self, stamp):
        """팔로워의 PoseStamped 메시지를 발행합니다."""
        msg = PoseStamped()
        msg.header.stamp = stamp.to_msg(); msg.header.frame_id = 'world'
        msg.pose.position.x, msg.pose.position.y = self.follower_pose[0], self.follower_pose[1]
        msg.pose.orientation = yaw_to_quaternion(self.follower_pose[2])
        self.follower_pose_pub.publish(msg)

    def _publish_uwb_distances(self, stamp):
        """UWB 거리 측정치를 시뮬레이션하여 발행합니다."""
        anchor_fwd = self.get_parameter('anchor_forward_offset').value
        anchor_width = self.get_parameter('anchor_width').value
        noise_std = self.get_parameter('uwb_noise_std').value
        
        f_yaw = self.follower_pose[2]
        R = np.array([[math.cos(f_yaw), -math.sin(f_yaw)], [math.sin(f_yaw), math.cos(f_yaw)]])
        f_pos = self.follower_pose[:2]
        
        anchors_local = np.array([[anchor_fwd, anchor_width / 2.0], [anchor_fwd, -anchor_width / 2.0]])
        anchors_world = (R @ anchors_local.T).T + f_pos
        
        leader_pos = self.leader_pose[:2]
        d_a = np.linalg.norm(leader_pos - anchors_world[0]) + np.random.normal(0, noise_std)
        d_b = np.linalg.norm(leader_pos - anchors_world[1]) + np.random.normal(0, noise_std)
        
        msg = PointStamped()
        msg.header.stamp = stamp.to_msg(); msg.header.frame_id = 'follower_uwb_center' # Or a relevant frame
        msg.point.x, msg.point.y = max(0.0, d_a), max(0.0, d_b)
        self.uwb_pub.publish(msg)
            
    def _publish_leader_ground_truth(self, stamp):
        """리더의 현재 Ground Truth 위치(PointStamped)를 발행합니다."""
        msg = PointStamped()
        msg.header.stamp = stamp.to_msg(); msg.header.frame_id = 'world'
        msg.point.x, msg.point.y = self.leader_pose[0], self.leader_pose[1]
        self.gt_leader_pub.publish(msg)

    def publish_gt_path(self):
        """주기적으로 Ground Truth 경로(Path)를 RDP 알고리즘으로 단순화하여 발행합니다."""
        with self.lock:
            if len(self.gt_path_history) < 2: return

            epsilon = self.get_parameter('rdp_epsilon').value
            simplified_points = self._douglas_peucker(self.gt_path_history, epsilon)
            
            path_msg = Path()
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.header.frame_id = 'world'
            
            for point in simplified_points:
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x, pose.pose.position.y = point[0], point[1]
                pose.pose.orientation.w = 1.0 # 기본 방향
                path_msg.poses.append(pose)
            
            self.gt_path_pub.publish(path_msg)
    
    def _douglas_peucker(self, points, epsilon):
        """RDP 알고리즘을 이용해 포인트 리스트를 단순화합니다."""
        if len(points) < 3: return points
        dmax, index = 0.0, 0
        p1, p_end = np.array(points[0]), np.array(points[-1])
        for i in range(1, len(points) - 1):
            d = np.linalg.norm(np.cross(p_end - p1, p1 - np.array(points[i]))) / np.linalg.norm(p_end - p1)
            if d > dmax: index, dmax = i, d
        
        if dmax > epsilon:
            rec1 = self._douglas_peucker(points[:index + 1], epsilon)
            rec2 = self._douglas_peucker(points[index:], epsilon)
            return rec1[:-1] + rec2
        else:
            return [points[0], points[-1]]

def main(args=None):
    rclpy.init(args=args)
    node = RobotSimulatorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()