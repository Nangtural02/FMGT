import rclpy
import numpy as np
import math
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation
# QoS 관련 클래스 임포트
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy

def yaw_to_quaternion(yaw):
    """Yaw 각도를 ROS Quaternion 메시지로 변환합니다."""
    q = Rotation.from_euler('z', yaw).as_quat()
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

class VirtualPathPublisher(Node):
    def __init__(self):
        super().__init__('virtual_path_publisher')
        self.declare_parameter('path_type', 's_curve')
        self.declare_parameter('num_points', 100)
        self.declare_parameter('path_length', 8.0)
        self.declare_parameter('path_width', 3.0)
        self.declare_parameter('circle_radius', 2.0)

        # ★★★★★★★★★★★★★★★★★★★★★
        # QoS 프로파일 정의: 메시지를 들고 있지 않도록(VOLATILE) 설정
        qos_profile = QoSProfile(
            durability=DurabilityPolicy.VOLATILE,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        # ★★★★★★★★★★★★★★★★★★★★★

        # 퍼블리셔 생성 시 QoS 프로파일 적용
        self.path_pub = self.create_publisher(
            Path, 
            '/leader/control_trajectory', 
            qos_profile) # QoS 적용

        self.published = False 
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info("가상 경로 발행 노드 시작됨.")

    def timer_callback(self):
        if not self.published:
            self.publish_path()
            self.published = True
            self.get_logger().info("경로를 성공적으로 1회 발행했습니다. 이후 추가 발행 없음.")

    # 나머지 코드는 이전과 동일합니다.
    def publish_path(self):
        path_type = self.get_parameter('path_type').value
        num_points = self.get_parameter('num_points').value
        path_msg = Path()
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
        path_msg.header = header
        if path_type == 's_curve':
            poses = self.create_s_curve(num_points, header)
        elif path_type == 'circle':
            poses = self.create_circle(num_points, header)
        elif path_type == 'straight':
            poses = self.create_straight_line(num_points, header)
        else:
            self.get_logger().error(f"알 수 없는 경로 타입: {path_type}"); return
        path_msg.poses = poses
        self.path_pub.publish(path_msg)

    def create_poses_from_points(self, points_x, points_y, header):
        poses = []
        for i in range(len(points_x)):
            pose = PoseStamped(header=header)
            pose.pose.position.x = points_x[i]
            pose.pose.position.y = points_y[i]
            if i < len(points_x) - 1:
                yaw = math.atan2(points_y[i+1] - points_y[i], points_x[i+1] - points_x[i])
            elif len(points_x) > 1:
                yaw = math.atan2(points_y[i] - points_y[i-1], points_x[i] - points_x[i-1])
            else: yaw = 0.0
            pose.pose.orientation = yaw_to_quaternion(yaw)
            poses.append(pose)
        return poses

    def create_s_curve(self, num_points, header):
        length = self.get_parameter('path_length').value
        width = self.get_parameter('path_width').value
        x = np.linspace(0, length, num_points)
        y = (width / 2) * np.sin(x * (np.pi / (length / 2)))
        return self.create_poses_from_points(x, y, header)

    def create_circle(self, num_points, header):
        radius = self.get_parameter('circle_radius').value
        theta = np.linspace(0, 2 * np.pi, num_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return self.create_poses_from_points(x, y, header)
        
    def create_straight_line(self, num_points, header):
        length = self.get_parameter('path_length').value
        x = np.linspace(0, length, num_points)
        y = np.zeros(num_points)
        return self.create_poses_from_points(x, y, header)

def main(args=None):
    rclpy.init(args=args)
    node = VirtualPathPublisher()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()