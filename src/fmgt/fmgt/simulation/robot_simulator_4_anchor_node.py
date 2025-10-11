# 파일명: robot_simulator_node.py
"""
Robot Simulator Node (Advanced, 4-Anchor, Hard Simulation Mode)

이 노드는 리더(Leader)와 팔로워(Follower) 로봇의 움직임을 시뮬레이션하고,
4-Anchor 시스템에 맞는 센서 데이터(UWB, Pose)와 Ground Truth 정보를 발행합니다.
'hard_simulation_mode'를 통해 현실적인 UWB 오차(정적/동적 NLOS, 다중 경로)를
시뮬레이션하는 고도화된 기능을 포함합니다.

- 주요 기능:
  1. 차동 구동 팔로워와 직교 좌표계 리더 시뮬레이션
  2. 팔로워의 추정 위치('/follower/estimated_pose') 발행
  3. 4개 앵커 UWB 데이터 발행 ('/uwb/distances_4_anchor')
     - Hard Simulation Mode: 정적/동적 NLOS, 다중 경로 에러를 포함한 현실적인 오차 모델 적용
  4. 디버깅을 위한 리더와 팔로워의 실제 위치 및 전체 경로 발행

- 파라미터 (Parameters):
  - hard_simulation_mode (bool): 현실적인 오차 모델 활성화 여부
  - ... (자세한 내용은 코드 내 파라미터 선언부 참조)
"""
import rclpy
import numpy as np
import math
import threading
import random
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Quaternion, Twist, PointStamped
from nav_msgs.msg import Path
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def yaw_to_quaternion(yaw):
    q = Rotation.from_euler('z', yaw).as_quat()
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

class RobotSimulatorNode(Node):
    def __init__(self):
        super().__init__('robot_simulator_node')

        # --- 기본 파라미터 ---
        self.declare_parameter('sim_rate', 50.0)
        self.declare_parameter('init_follower_pose', [0.0, 0.0, 0.0])
        self.declare_parameter('init_leader_pose', [1.5, 0.0, 0.0])
        self.declare_parameter('path_publish_period_sec', 1.0)
        self.declare_parameter('rdp_epsilon', 0.05)

        # --- UWB 기본 파라미터 (Ideal Mode) ---
        self.declare_parameter('anchor_pos.a', [0.25, 0.2])
        self.declare_parameter('anchor_pos.b', [0.25, -0.2])
        self.declare_parameter('anchor_pos.c', [-0.25, 0.2])
        self.declare_parameter('anchor_pos.d', [-0.25, -0.2])
        self.declare_parameter('uwb_noise_std', 0.05) # 이상적인 환경에서의 기본 노이즈

        # --- Hard Simulation Mode 파라미터 ---
        self.declare_parameter('hard_simulation_mode', True)
        self.declare_parameter('enable_static_nlos', True)
        self.declare_parameter('enable_dynamic_nlos', True)
        self.declare_parameter('enable_multipath_error', True)

        self.declare_parameter('robot_blocker_dims', [0.7, 0.5]) # 로봇 몸체 차폐 영역 [가로, 세로]
        
        self.declare_parameter('gmm_multipath_probability', 0.05)
        self.declare_parameter('gmm_multipath_bias', 0.8)
        self.declare_parameter('gmm_multipath_std', 0.6)

        self.declare_parameter('p_los_to_nlos', 0.01)
        self.declare_parameter('p_nlos_to_los', 0.1)
        
        # --- 논문 기반 NLOS 오차 모델 파라미터 (bias_mean, bias_std, noise_std) ---
        self.nlos_models = {
            'soft':  {'bias_mean': 0.35, 'bias_std': 0.15, 'noise_std': 0.3},
            'hard':  {'bias_mean': 1.65, 'bias_std': 0.85, 'noise_std': 0.8},
            'human': {'bias_mean': 0.7,  'bias_std': 0.3,  'noise_std': 0.5}
        }
        self.declare_parameter('static_nlos_type', 'hard') # 로봇 몸체는 'hard' NLOS로 간주
        self.declare_parameter('dynamic_nlos_type', 'human') # 동적 장애물은 'human'으로 간주

        # --- 상태 변수 ---
        self.follower_pose = np.array(self.get_parameter('init_follower_pose').value)
        self.leader_pose = np.array(self.get_parameter('init_leader_pose').value)
        self.follower_cmd_vel, self.leader_cmd_vel = Twist(), Twist()
        self.gt_path_history, self.follower_path_history = [], []
        self.dynamic_nlos_states = { 'a': 'LOS', 'b': 'LOS', 'c': 'LOS', 'd': 'LOS' }
        self.lock = threading.Lock()

        # --- 발행자 & 구독자 ---
        self.follower_pose_pub = self.create_publisher(PoseStamped, '/follower/estimated_pose', 10)
        self.uwb_pub = self.create_publisher(PoseStamped, '/uwb/distances_4_anchor', 10)
        self.gt_leader_pub = self.create_publisher(PointStamped, '/leader/ground_truth', 10)
        self.gt_path_pub = self.create_publisher(Path, '/debug/ground_truth_path', 10)
        self.follower_path_pub = self.create_publisher(Path, '/debug/follower_ground_truth_path', 10)
        self.create_subscription(Twist, 'cmd_vel', self.follower_cmd_callback, 10)
        self.create_subscription(Twist, 'leader_teleop/cmd_vel', self.leader_cmd_callback, 10)

        # --- 타이머 ---
        sim_rate = self.get_parameter('sim_rate').value; self.dt = 1.0 / sim_rate
        self.sim_timer = self.create_timer(self.dt, self.simulation_loop)
        path_pub_period = self.get_parameter('path_publish_period_sec').value
        self.path_timer = self.create_timer(path_pub_period, self.publish_ground_truth_paths)

        self.get_logger().info("Robot Simulator (Advanced Full Version) 시작됨.")
        is_hard_mode = self.get_parameter('hard_simulation_mode').value
        self.get_logger().info(f"--- Hard Simulation Mode: {'ON' if is_hard_mode else 'OFF'} ---")

    # (생략) ... follower_cmd_callback, leader_cmd_callback, 포즈 업데이트 함수들은 이전과 동일 ...
    def follower_cmd_callback(self, msg):
        with self.lock: self.follower_cmd_vel = msg

    def leader_cmd_callback(self, msg):
        with self.lock: self.leader_cmd_vel = msg
        
    def _update_diff_drive_pose(self, pose, cmd_vel, dt):
        vx, wz = cmd_vel.linear.x, cmd_vel.angular.z
        pose[0] += vx * math.cos(pose[2]) * dt; pose[1] += vx * math.sin(pose[2]) * dt
        pose[2] = normalize_angle(pose[2] + wz * dt)
        return pose

    def _update_cartesian_pose(self, pose, cmd_vel, dt):
        pose[0] += cmd_vel.linear.x * dt; pose[1] += cmd_vel.linear.y * dt
        return pose

    def simulation_loop(self):
        with self.lock:
            now = self.get_clock().now()
            self.leader_pose = self._update_cartesian_pose(self.leader_pose, self.leader_cmd_vel, self.dt)
            self.follower_pose = self._update_diff_drive_pose(self.follower_pose, self.follower_cmd_vel, self.dt)
            self.gt_path_history.append(self.leader_pose[:2].copy())
            self.follower_path_history.append(self.follower_pose[:2].copy())
            self._publish_follower_pose(now)
            self._publish_uwb_distances(now)
            self._publish_leader_ground_truth(now)
    
    def _publish_follower_pose(self, stamp):
        msg = PoseStamped(); msg.header.stamp = stamp.to_msg(); msg.header.frame_id = 'world'
        msg.pose.position.x, msg.pose.position.y = self.follower_pose[0], self.follower_pose[1]
        msg.pose.orientation = yaw_to_quaternion(self.follower_pose[2])
        self.follower_pose_pub.publish(msg)

    def _publish_uwb_distances(self, stamp):
        leader_pos = self.leader_pose[:2]
        f_yaw, f_pos = self.follower_pose[2], self.follower_pose[:2]
        rot_mat = np.array([[math.cos(f_yaw), -math.sin(f_yaw)], [math.sin(f_yaw), math.cos(f_yaw)]])
        
        anchor_keys = ['a', 'b', 'c', 'd']
        anchor_locals = [np.array(self.get_parameter(f'anchor_pos.{k}').value) for k in anchor_keys]
        anchors_world = {k: f_pos + rot_mat @ p for k, p in zip(anchor_keys, anchor_locals)}
        
        final_distances = {}
        
        # --- Hard Simulation Mode 분기 ---
        if self.get_parameter('hard_simulation_mode').value:
            base_noise_std = self.get_parameter('uwb_noise_std').value
            
            for key, p_world in anchors_world.items():
                actual_dist = np.linalg.norm(leader_pos - p_world)
                bias, std = 0.0, base_noise_std
                
                # 1. 정적 NLOS (로봇 몸체)
                if self.get_parameter('enable_static_nlos').value and self._is_statically_blocked(leader_pos, p_world):
                    nlos_type = self.get_parameter('static_nlos_type').value
                    model = self.nlos_models[nlos_type]
                    bias += np.random.normal(model['bias_mean'], model['bias_std'])
                    std = max(std, model['noise_std'])

                # 2. 동적 NLOS (사람 등)
                if self.get_parameter('enable_dynamic_nlos').value:
                    self._update_dynamic_nlos_state(key)
                    if self.dynamic_nlos_states[key] == 'NLOS':
                        nlos_type = self.get_parameter('dynamic_nlos_type').value
                        model = self.nlos_models[nlos_type]
                        bias += np.random.normal(model['bias_mean'], model['bias_std'])
                        std = max(std, model['noise_std'])

                total_noise = np.random.normal(bias, std)

                # 3. 다중 경로 에러 (GMM)
                if self.get_parameter('enable_multipath_error').value and random.random() < self.get_parameter('gmm_multipath_probability').value:
                    gmm_bias = self.get_parameter('gmm_multipath_bias').value
                    gmm_std = self.get_parameter('gmm_multipath_std').value
                    total_noise += np.random.normal(gmm_bias, gmm_std)
                
                final_distances[key] = actual_dist + total_noise
        else:
            # --- Ideal Mode (기존 가우시안 노이즈) ---
            noise_std = self.get_parameter('uwb_noise_std').value
            for key, p_world in anchors_world.items():
                actual_dist = np.linalg.norm(leader_pos - p_world)
                final_distances[key] = actual_dist + np.random.normal(0, noise_std)

        msg = PoseStamped(); msg.header.stamp = stamp.to_msg(); msg.header.frame_id = 'follower_uwb_center'
        msg.pose.position.x = max(0.0, final_distances['a']); msg.pose.position.y = max(0.0, final_distances['b'])
        msg.pose.position.z = max(0.0, final_distances['c']); msg.pose.orientation.x = max(0.0, final_distances['d'])
        msg.pose.orientation.w = 1.0
        self.uwb_pub.publish(msg)

    def _is_statically_blocked(self, leader_pos, anchor_pos):
        f_pos, f_yaw = self.follower_pose[:2], self.follower_pose[2]
        dims = self.get_parameter('robot_blocker_dims').value
        
        # 로봇 좌표계로 변환
        rot_mat_inv = np.array([[math.cos(-f_yaw), -math.sin(-f_yaw)], [math.sin(-f_yaw), math.cos(-f_yaw)]])
        leader_local = rot_mat_inv @ (leader_pos - f_pos)
        anchor_local = rot_mat_inv @ (anchor_pos - f_pos)
        
        # 로봇 중심의 사각형과 선분 교차 판정 (간단한 버전)
        # 리더와 앵커가 로봇 중심을 기준으로 서로 다른 사분면에 있는지 확인
        # (예: 리더는 전방, 앵커는 후방에 있으면 차단될 가능성 높음)
        # 이는 정확한 교차 판정은 아니지만, 계산이 빠르고 효과적인 근사치입니다.
        x_multi = leader_local[0] * anchor_local[0]
        y_multi = leader_local[1] * anchor_local[1]
        
        # x좌표 또는 y좌표의 부호가 서로 다르면 교차 가능성이 매우 높음
        if x_multi < 0 or y_multi < 0:
            # 더 정확한 체크: 선분이 실제로 사각형을 가로지르는가?
            # 여기서는 근사치로 충분하다고 판단하여 위 조건만 사용
            return True
        return False
    
    def _update_dynamic_nlos_state(self, anchor_key):
        p_los_to_nlos = self.get_parameter('p_los_to_nlos').value
        p_nlos_to_los = self.get_parameter('p_nlos_to_los').value
        
        if self.dynamic_nlos_states[anchor_key] == 'LOS':
            if random.random() < p_los_to_nlos:
                self.dynamic_nlos_states[anchor_key] = 'NLOS'
        else: # NLOS
            if random.random() < p_nlos_to_los:
                self.dynamic_nlos_states[anchor_key] = 'LOS'

    def _publish_leader_ground_truth(self, stamp):
        msg = PointStamped()
        msg.header.stamp = stamp.to_msg(); msg.header.frame_id = 'world'
        msg.point.x, msg.point.y = self.leader_pose[0], self.leader_pose[1]
        self.gt_leader_pub.publish(msg)

    def publish_ground_truth_paths(self):
        with self.lock:
            now = self.get_clock().now().to_msg()
            epsilon = self.get_parameter('rdp_epsilon').value

            if len(self.gt_path_history) >= 2:
                path_msg = self._create_path_msg(now, self.gt_path_history, epsilon)
                self.gt_path_pub.publish(path_msg)
            
            if len(self.follower_path_history) >= 2:
                path_msg = self._create_path_msg(now, self.follower_path_history, epsilon)
                self.follower_path_pub.publish(path_msg)
    
    def _create_path_msg(self, stamp, history, epsilon):
        simplified_points = self._douglas_peucker(history, epsilon)
        path_msg = Path(); path_msg.header.stamp = stamp; path_msg.header.frame_id = 'world'
        for point in simplified_points:
            pose = PoseStamped(); pose.header = path_msg.header
            pose.pose.position.x, pose.pose.position.y = point[0], point[1]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        return path_msg

    def _douglas_peucker(self, points, epsilon):
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
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()