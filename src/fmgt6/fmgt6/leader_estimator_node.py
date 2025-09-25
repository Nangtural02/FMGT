# 파일명: leader_estimator_node.py
"""
Leader Estimator Node (Coordinate Transformer)

이 노드는 팔로워의 위치/자세와 UWB 거리 측정치를 이용해 리더의 절대 위치를 계산하는
단순 좌표 변환기 역할을 합니다. 내부적인 필터링 없이 계산된 '날것의' 위치 측정치를 발행합니다.

- 구독 (Subscriptions):
  - /follower/estimated_pose (geometry_msgs/PoseStamped): 팔로워의 추정된 위치 및 자세
  - raw_uwb_distances (geometry_msgs/PointStamped): UWB 태그로부터의 거리 (x: d_a, y: d_b)

- 발행 (Publications):
  - /leader/raw_point (geometry_msgs/PointStamped): 월드 좌표계에서 계산된 리더의 위치 측정치

- 파라미터 (Parameters):
  - anchor_forward_offset (double): 로봇 중심에서 UWB 앵커 중앙까지의 전방 거리 (m)
  - anchor_width (double): 두 UWB 앵커 사이의 폭 (m)
"""
import rclpy, numpy as np, math, threading
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PointStamped, PoseStamped
from scipy.spatial.transform import Rotation

class LeaderEstimatorNode(Node):
    def __init__(self):
        super().__init__('leader_estimator_node')
        
        self.declare_parameter('anchor_forward_offset', 0.25)
        self.declare_parameter('anchor_width', 0.4)
        
        self.raw_point_pub = self.create_publisher(PointStamped, '/leader/raw_point', 10)
        self.lock = threading.Lock(); self.latest_follower_pose = None

        self.follower_pose_sub = self.create_subscription(PoseStamped, '/follower/estimated_pose', self.follower_pose_callback, 10)
        self.uwb_sub = self.create_subscription(PointStamped, 'raw_uwb_distances', self.uwb_update_callback, QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1))
        self.get_logger().info("Leader Estimator Node (Coordinate Transformer) 시작됨.")

    def follower_pose_callback(self, msg: PoseStamped):
        with self.lock: self.latest_follower_pose = msg

    def uwb_update_callback(self, uwb_msg):
        with self.lock:
            if self.latest_follower_pose is None: return
            
            anchor_forward_offset = self.get_parameter('anchor_forward_offset').value
            anchor_width = self.get_parameter('anchor_width').value
            
            pf_x, pf_y = self.latest_follower_pose.pose.position.x, self.latest_follower_pose.pose.position.y
            q = self.latest_follower_pose.pose.orientation
            follower_yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]

            try:
                d_a, d_b = uwb_msg.point.x, uwb_msg.point.y
                if not(d_a > 0.1 and d_b > 0.1): return
                
                Y_off = anchor_width / 2
                py_local = (d_b**2 - d_a**2) / (4 * Y_off)
                px_sq = d_a**2 - (py_local - Y_off)**2
                if px_sq < 0: return
                px_local = anchor_forward_offset + math.sqrt(px_sq)
                
                cos_th, sin_th = math.cos(follower_yaw), math.sin(follower_yaw)
                z = np.array([pf_x + (px_local*cos_th-py_local*sin_th), pf_y + (px_local*sin_th+py_local*cos_th)])

                raw_point_msg = PointStamped()
                raw_point_msg.header.stamp = uwb_msg.header.stamp; raw_point_msg.header.frame_id = 'world'
                raw_point_msg.point.x, raw_point_msg.point.y = z[0], z[1]
                self.raw_point_pub.publish(raw_point_msg)
            except Exception: return

def main(args=None):
    rclpy.init(args=args); node = LeaderEstimatorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__': main()