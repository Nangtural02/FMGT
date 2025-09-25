# ~/dev/ROS2/FMGT/src/fmgt_test/fmgt_test/mock_pose_publisher.py

import rclpy
import math
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped, Quaternion
from tf2_ros import TransformBroadcaster
from scipy.spatial.transform import Rotation

class MockPosePublisher(Node):
    def __init__(self):
        super().__init__('mock_pose_publisher')
        self.declare_parameter('initial_x', 0.0)
        self.declare_parameter('initial_y', 0.0)
        self.declare_parameter('initial_yaw_deg', 0.0)
        self.x = self.get_parameter('initial_x').value
        self.y = self.get_parameter('initial_y').value
        self.yaw = math.radians(self.get_parameter('initial_yaw_deg').value)
        self.last_cmd_vel = Twist()
        self.pose_pub = self.create_publisher(PoseStamped, '/follower/estimated_pose', 10)
        self.cmd_vel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.timer = self.create_timer(0.05, self.update_and_publish_pose)
        self.get_logger().info('가상 로봇 위치 발행 노드 시작됨.')

    def cmd_vel_callback(self, msg):
        self.last_cmd_vel = msg

    def update_and_publish_pose(self):
        dt = 0.05
        self.yaw += self.last_cmd_vel.angular.z * dt
        self.x += self.last_cmd_vel.linear.x * math.cos(self.yaw) * dt
        self.y += self.last_cmd_vel.linear.x * math.sin(self.yaw) * dt

        current_time = self.get_clock().now().to_msg()
        
        pose_msg = PoseStamped()
        pose_msg.header.stamp = current_time
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = self.x
        pose_msg.pose.position.y = self.y
        
        q = Rotation.from_euler('z', self.yaw).as_quat()
        pose_msg.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.pose_pub.publish(pose_msg)
        
        t = TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = 'world'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        t.transform.rotation = pose_msg.pose.orientation
        self.tf_broadcaster.sendTransform(t)

# --- main 함수와 실행 블록 ---
def main(args=None):
    rclpy.init(args=args)
    node = MockPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()