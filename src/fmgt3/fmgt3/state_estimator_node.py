# fmgt/state_estimator_node.py
import rclpy, math, time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Point, PointStamped, TwistStamped, Twist
from std_msgs.msg import Header

ANCHOR_DISTANCE = 0.4
ANCHOR_A_POS = np.array([-ANCHOR_DISTANCE / 2, 0.0])

class StateEstimatorNode(Node):
    def __init__(self):
        super().__init__('state_estimator_node')
        
        # 발행자 2개 선언: 위치용, 속도용
        self.pos_publisher_ = self.create_publisher(PointStamped, 'relative_position', 10)
        self.vel_publisher_ = self.create_publisher(TwistStamped, 'relative_velocity', 10) # ★ 속도 발행자 추가
        
        # EKF 상태 변수
        self.x = np.zeros(4)
        self.P = np.eye(4) * 100.0
        accel_std_dev = 0.5; self.Q = np.diag([0.0, 0.0, accel_std_dev**2, accel_std_dev**2])
        measurement_std_dev = 0.1; self.R = np.eye(2) * (measurement_std_dev**2)
        
        self.last_timestamp = None
        self.is_initialized = False
        
        self.subscription = self.create_subscription(Point, 'raw_uwb_distances', self.listener_callback, 10)
        self.get_logger().info('EKF State Estimator Node 시작됨.')

    def listener_callback(self, msg):
        current_timestamp = time.time()
        
        if not self.is_initialized:
            pos = self.triangulate(msg.x, msg.y)
            if pos:
                self.x[0], self.x[1] = pos[0], pos[1]
                self.last_timestamp = current_timestamp
                self.is_initialized = True
                self.get_logger().info(f'EKF 초기화 성공: ({pos[0]:.2f}, {pos[1]:.2f})')
            return

        dt = current_timestamp - self.last_timestamp
        self.last_timestamp = current_timestamp
        if dt <= 0: return

        # 예측 (Prediction)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q * dt

        # 업데이트 (Update)
        z_pos = self.triangulate(msg.x, msg.y)
        if z_pos is not None:
            z = np.array(z_pos)
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
            y_err = z - H @ self.x
            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y_err
            self.P = (np.eye(4) - K @ H) @ self.P

        # 발행 (Publish)
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='follower_base_link')
        
        # 위치 발행
        pos_msg = PointStamped(header=header, point=Point(x=self.x[0], y=self.x[1], z=0.0))
        self.pos_publisher_.publish(pos_msg)

        # ★ 속도 발행 추가
        vel_msg = TwistStamped(header=header)
        vel_msg.twist.linear.x = self.x[2] # vx
        vel_msg.twist.linear.y = self.x[3] # vy
        self.vel_publisher_.publish(vel_msg)

    def triangulate(self, d_a, d_b):
        L = ANCHOR_DISTANCE
        if not (d_a > 0 and d_b > 0 and d_a + d_b >= L and abs(d_a - d_b) <= L): return None
        try:
            x = (d_a**2 - d_b**2) / (2 * L)
            y_squared = d_a**2 - (x - ANCHOR_A_POS[0])**2
            return (x, math.sqrt(y_squared)) if y_squared >= 0 else None
        except (ValueError, ZeroDivisionError): return None

def main(args=None):
    rclpy.init(args=args)
    node = StateEstimatorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__': main()