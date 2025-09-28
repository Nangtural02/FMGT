# 파일명: robot_simulator_node.py
# 경로: fmgt/simulation/robot_simulator_node.py
import rclpy
import numpy as np
import math
import threading
from rclpy.node import Node

# ★★★ PoseStamped 대신 PointStamped를 import ★★★
from geometry_msgs.msg import PoseStamped, Quaternion, Twist, PointStamped 
from scipy.spatial.transform import Rotation

def normalize_angle(angle): return (angle + np.pi) % (2 * np.pi) - np.pi
def yaw_to_quaternion(yaw):
    q = Rotation.from_euler('z', yaw).as_quat()
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

class RobotSimulatorNode(Node):
    def __init__(self):
        super().__init__('robot_simulator_node')

        self.declare_parameter('sim_rate', 5.0)
        self.declare_parameter('anchor_forward_offset', 0.25)
        self.declare_parameter('anchor_width', 0.4)
        self.declare_parameter('init_follower_pose', [0.0, 0.0, 0.0])
        self.declare_parameter('init_leader_pose', [1.5, 0.0, 0.0])
        self.declare_parameter('uwb_noise_std', 0.05)

        self.follower_pose = np.array(self.get_parameter('init_follower_pose').value)
        self.leader_pose = np.array(self.get_parameter('init_leader_pose').value)
        self.follower_cmd_vel = Twist()
        self.leader_cmd_vel = Twist()
        self.lock = threading.Lock()

        self.follower_pose_pub = self.create_publisher(PoseStamped, '/follower/estimated_pose', 10)
        self.uwb_pub = self.create_publisher(PointStamped, 'raw_uwb_distances', 10)
        # ★★★ 발행 타입을 PointStamped로 변경 ★★★
        self.gt_leader_pub = self.create_publisher(PointStamped, '/leader/ground_truth', 10)

        self.create_subscription(Twist, 'cmd_vel', self.follower_cmd_callback, 10)
        self.create_subscription(Twist, 'leader_teleop/cmd_vel', self.leader_cmd_callback, 10)

        sim_rate = self.get_parameter('sim_rate').value
        self.dt = 1.0 / sim_rate
        self.timer = self.create_timer(self.dt, self.simulation_loop)

        self.get_logger().info("Robot Simulator 시작됨 (Leader: Cartesian, Follower: DiffDrive).")

    def follower_cmd_callback(self, msg):
        with self.lock: self.follower_cmd_vel = msg

    def leader_cmd_callback(self, msg):
        with self.lock: self.leader_cmd_vel = msg
        
    def update_diff_drive_pose(self, pose, cmd_vel, dt):
        vx = cmd_vel.linear.x
        wz = cmd_vel.angular.z
        yaw = pose[2]
        pose[0] += vx * math.cos(yaw) * dt
        pose[1] += vx * math.sin(yaw) * dt
        pose[2] = normalize_angle(pose[2] + wz * dt)
        return pose

    def update_cartesian_pose(self, pose, cmd_vel, dt):
        vx = cmd_vel.linear.x
        vy = cmd_vel.linear.y
        pose[0] += vx * dt
        pose[1] += vy * dt
        return pose

    def simulation_loop(self):
        with self.lock:
            now = self.get_clock().now()
            
            self.leader_pose = self.update_cartesian_pose(self.leader_pose, self.leader_cmd_vel, self.dt)
            self.follower_pose = self.update_diff_drive_pose(self.follower_pose, self.follower_cmd_vel, self.dt)

            # 팔로워 위치 발행 (PoseStamped 유지)
            follower_pose_msg = PoseStamped()
            follower_pose_msg.header.stamp = now.to_msg(); follower_pose_msg.header.frame_id = 'world'
            follower_pose_msg.pose.position.x = self.follower_pose[0]
            follower_pose_msg.pose.position.y = self.follower_pose[1]
            follower_pose_msg.pose.orientation = yaw_to_quaternion(self.follower_pose[2])
            self.follower_pose_pub.publish(follower_pose_msg)

            # UWB 거리 계산 및 발행
            anchor_fwd = self.get_parameter('anchor_forward_offset').value
            anchor_width = self.get_parameter('anchor_width').value
            f_yaw = self.follower_pose[2]
            R = np.array([[math.cos(f_yaw), -math.sin(f_yaw)], [math.sin(f_yaw), math.cos(f_yaw)]])
            f_pos = self.follower_pose[:2]
            anchors_local = np.array([[anchor_fwd, anchor_width / 2.0], [anchor_fwd, -anchor_width / 2.0]])
            anchors_world = (R @ anchors_local.T).T + f_pos
            leader_pos = self.leader_pose[:2]
            d_a = np.linalg.norm(leader_pos - anchors_world[0]) + np.random.normal(0, self.get_parameter('uwb_noise_std').value)
            d_b = np.linalg.norm(leader_pos - anchors_world[1]) + np.random.normal(0, self.get_parameter('uwb_noise_std').value)
            uwb_msg = PointStamped()
            uwb_msg.header.stamp = now.to_msg(); uwb_msg.header.frame_id = 'follower_uwb_center'
            uwb_msg.point.x = max(0.0, d_a); uwb_msg.point.y = max(0.0, d_b)
            self.uwb_pub.publish(uwb_msg)
            
            # ★★★ (디버깅용) 리더 실제 위치를 PointStamped로 발행 ★★★
            leader_gt_msg = PointStamped()
            leader_gt_msg.header.stamp = now.to_msg()
            leader_gt_msg.header.frame_id = 'world'
            leader_gt_msg.point.x = self.leader_pose[0]
            leader_gt_msg.point.y = self.leader_pose[1]
            leader_gt_msg.point.z = 0.0 # z축은 0으로 설정
            self.gt_leader_pub.publish(leader_gt_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotSimulatorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()