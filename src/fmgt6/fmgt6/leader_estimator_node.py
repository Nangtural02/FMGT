# 파일명: leader_estimator_node.py
import rclpy
import numpy as np
import math
import threading
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation

# --- 상수 ---
ANCHOR_FORWARD_OFFSET = 0.25; ANCHOR_WIDTH = 0.4
def normalize_angle(angle): return (angle + np.pi) % (2 * np.pi) - np.pi

class LeaderEstimatorNode(Node):
    def __init__(self):
        super().__init__('leader_estimator_node')
        
        self.declare_parameter('leader_process_variance', 0.2**2)
        self.declare_parameter('leader_measurement_variance', 0.3**2)

        self.leader_point_pub = self.create_publisher(PointStamped, '/leader/estimated_point', 10)
        self.lock = threading.Lock()
        
        self.x_l = np.zeros(2)
        self.P_l = np.eye(2) * 1000
        q_var, r_var = self.get_parameter('leader_process_variance').value, self.get_parameter('leader_measurement_variance').value
        self.Q_l = np.diag([q_var, q_var]); self.R_l = np.diag([r_var, r_var])
        self.is_leader_initialized = False
        self.latest_follower_pose = None

        self.follower_pose_sub = self.create_subscription(PoseStamped, '/follower/estimated_pose', self.follower_pose_callback, 10)
        self.uwb_sub = self.create_subscription(PointStamped, 'raw_uwb_distances', self.uwb_update_callback, QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1))
        
        self.get_logger().info("Leader Estimator Node 시작됨.")

    def follower_pose_callback(self, msg: PoseStamped):
        with self.lock:
            if self.is_leader_initialized and self.latest_follower_pose is not None:
                prev_pos = self.latest_follower_pose.pose.position
                q_prev = self.latest_follower_pose.pose.orientation
                yaw_prev = Rotation.from_quat([q_prev.x, q_prev.y, q_prev.z, q_prev.w]).as_euler('zyx')[0]
                q_curr = msg.pose.orientation
                yaw_curr = Rotation.from_quat([q_curr.x, q_curr.y, q_curr.z, q_curr.w]).as_euler('zyx')[0]
                
                p_follower_world = np.array([prev_pos.x, prev_pos.y])
                R_w_f_prev_inv = np.array([[math.cos(-yaw_prev), -math.sin(-yaw_prev)], [math.sin(-yaw_prev), math.cos(-yaw_prev)]])
                p_leader_local = R_w_f_prev_inv @ (self.x_l - p_follower_world)
                
                p_new_follower_world = np.array([msg.pose.position.x, msg.pose.position.y])
                R_w_f_new = np.array([[math.cos(yaw_curr), -math.sin(yaw_curr)], [math.sin(yaw_curr), math.cos(yaw_curr)]])
                self.x_l = R_w_f_new @ p_leader_local + p_new_follower_world

                self.P_l += self.Q_l * 0.01 # dt는 근사치 사용

            self.latest_follower_pose = msg

    def uwb_update_callback(self, uwb_msg):
        with self.lock:
            if self.latest_follower_pose is None: return
            
            pf_x, pf_y = self.latest_follower_pose.pose.position.x, self.latest_follower_pose.pose.position.y
            q = self.latest_follower_pose.pose.orientation
            follower_yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]

            try:
                d_a, d_b = uwb_msg.point.x, uwb_msg.point.y
                if not(d_a > 0.1 and d_b > 0.1): return
                
                Y_off = ANCHOR_WIDTH / 2; py_local = (d_b**2 - d_a**2) / (4 * Y_off)
                px_sq = d_a**2 - (py_local - Y_off)**2
                if px_sq < 0: return
                px_local = ANCHOR_FORWARD_OFFSET + math.sqrt(px_sq)
                
                cos_th, sin_th = math.cos(follower_yaw), math.sin(follower_yaw)
                z = np.array([pf_x + (px_local*cos_th-py_local*sin_th), pf_y + (px_local*sin_th+py_local*cos_th)])
                
                if not self.is_leader_initialized:
                    self.x_l, self.is_leader_initialized = z, True
                    self.get_logger().info("Leader KF 초기화 성공!")

                H_l = np.eye(2); y_err = z - self.x_l
                S = H_l @ self.P_l @ H_l.T + self.R_l
                K = self.P_l @ H_l.T @ np.linalg.inv(S)
                self.x_l += K @ y_err; self.P_l = (np.eye(2) - K @ H_l) @ self.P_l

                header = Header(stamp=uwb_msg.header.stamp, frame_id='world')
                point_msg = PointStamped(header=header, point={'x': self.x_l[0], 'y': self.x_l[1]})
                self.leader_point_pub.publish(point_msg)
                
            except Exception: return

def main(args=None):
    rclpy.init(args=args)
    node = LeaderEstimatorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()