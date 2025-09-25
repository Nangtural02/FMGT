# 파일명: trajectory_estimator_node.py
import rclpy
import numpy as np
import math
import threading
import os
import csv
from datetime import datetime
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from collections import deque

from geometry_msgs.msg import Point, PointStamped, PoseStamped, Quaternion, TransformStamped
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
from nav_msgs.msg import Path
import tf2_ros

from scipy.spatial.transform import Rotation
from pykalman import KalmanFilter

# --- 상수 정의 ---
GRAVITY = np.array([0, 0, -9.81]); ANCHOR_FORWARD_OFFSET = 0.25; ANCHOR_WIDTH = 0.4

def normalize_angle(angle): return (angle + np.pi) % (2 * np.pi) - np.pi
def yaw_to_quaternion(yaw):
    q = Rotation.from_euler('z', yaw).as_quat(); return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])

class TrajectoryEstimatorNode(Node):
    def __init__(self):
        super().__init__('trajectory_estimator_node')
        
        # --- 파라미터 선언 ---
        self.declare_parameter('log_raw_data', True)
        self.declare_parameter('log_file_path', os.path.expanduser('~') + '/fmgt_logs')
        self.declare_parameter('smoother_update_period_sec', 1.0)
        self.declare_parameter('leader_process_variance', 0.2**2)
        self.declare_parameter('leader_measurement_variance', 0.3**2)
        self.declare_parameter('motion_variance_threshold', 0.2**2)

        # --- 발행자 ---
        self.follower_pose_pub = self.create_publisher(PoseStamped, '/follower/estimated_pose', 10)
        self.full_trajectory_pub = self.create_publisher(Path, '/leader/full_trajectory', 10)
        self.stable_point_pub = self.create_publisher(PointStamped, '/leader/stable_point', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.lock = threading.Lock()

        # --- 필터 (fmgt4 로직) ---
        self.x_f = np.zeros(9); self.P_f = np.eye(9)*0.1
        self.Q_f = np.diag([1e-8,1e-8,1e-6,0.05**2,0.05**2,0.05**2,0.01**2,0.01**2,(math.radians(0.1))**2])
        odom_pos_var=0.1**2; odom_yaw_var=(math.radians(2.0))**2
        self.R_odom = np.diag([odom_pos_var, odom_pos_var, odom_yaw_var])
        self.is_initialized=False; self.last_imu_timestamp=None
        self.initial_odom_pos=None; self.initial_odom_yaw=0.0; self.last_odom_pose=None
        self.x_l = np.zeros(2); self.P_l = np.eye(2) * 1000
        q_var = self.get_parameter('leader_process_variance').value; self.Q_l = np.diag([q_var, q_var])
        r_var = self.get_parameter('leader_measurement_variance').value; self.R_l = np.diag([r_var, r_var])
        self.is_leader_initialized = False
        
        self.raw_measurements = [] # ★★★ KF 결과가 아닌, 날것의 측정치를 저장
        self.leader_state = 'Stationary'; self.pos_buffer = deque(maxlen=15)

        # --- 로깅 설정 ---
        self.log_file = None; self.csv_writer = None; self.last_imu_msg = None
        if self.get_parameter('log_raw_data').value:
            log_path = self.get_parameter('log_file_path').value
            os.makedirs(log_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_filename = os.path.join(log_path, f"fmgt_log_{timestamp}.csv")
            
            self.log_file = open(log_filename, 'w', newline='')
            self.csv_writer = csv.writer(self.log_file)
            # ★★★ 로깅 대상 확장 ★★★
            self.csv_writer.writerow([
                'timestamp', 'd_a', 'd_b', 
                'odom_x', 'odom_y', 'odom_qw', 'odom_qx', 'odom_qy', 'odom_qz',
                'imu_acc_x', 'imu_acc_y', 'imu_acc_z', 
                'imu_w_x', 'imu_w_y', 'imu_w_z',
                'imu_q_w', 'imu_q_x', 'imu_q_y', 'imu_q_z'
            ])
            self.get_logger().info(f"Raw data logging enabled. Saving to: {log_filename}")

        # --- 구독자 ---
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_predict_callback, sensor_qos)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_update_callback, sensor_qos)
        self.uwb_sub = self.create_subscription(PointStamped, 'raw_uwb_distances', self.uwb_update_callback, sensor_qos)
        
        smoother_period = self.get_parameter('smoother_update_period_sec').value
        self.path_smoother_timer = self.create_timer(smoother_period, self.smooth_and_publish_path)
        
        self.get_logger().info(f"Trajectory Estimator (fmgt5-final-v4) 시작됨.")
        
    def imu_predict_callback(self, imu_msg):
        with self.lock:
            self.last_imu_msg = imu_msg # 로깅을 위해 최신 IMU 메시지 저장
            
            if not self.is_initialized: return
            current_timestamp = rclpy.time.Time.from_msg(imu_msg.header.stamp).nanoseconds / 1e9
            if self.last_imu_timestamp is None: self.last_imu_timestamp = current_timestamp; return
            dt = current_timestamp - self.last_imu_timestamp
            self.last_imu_timestamp = current_timestamp
            if not (0 < dt < 0.5): return
            
            x_f_before = self.x_f.copy()

            q_orientation=np.array([imu_msg.orientation.x,imu_msg.orientation.y,imu_msg.orientation.z,imu_msg.orientation.w])
            accel_original=np.array([imu_msg.linear_acceleration.x,imu_msg.linear_acceleration.y,imu_msg.linear_acceleration.z])
            omega_original=np.array([imu_msg.angular_velocity.x,imu_msg.angular_velocity.y,imu_msg.angular_velocity.z])
            imu_rotation=Rotation.from_quat(q_orientation); gravity_in_imu_frame=imu_rotation.inv().apply(GRAVITY)
            pure_accel = accel_original - gravity_in_imu_frame
            ax_body_ros = pure_accel[0]; ay_body_ros = pure_accel[1]
            b_ax, b_ay, b_wz = self.x_f[6], self.x_f[7], self.x_f[8]
            ax_local = ax_body_ros - b_ax; ay_local = ay_body_ros - b_ay
            omega_z_local = omega_original[2] - b_wz
            theta_f = self.x_f[2]; cos_th, sin_th = math.cos(theta_f), math.sin(theta_f)
            ax_world = ax_local * cos_th - ay_local * sin_th
            ay_world = ax_local * sin_th + ay_local * cos_th
            self.x_f[0]+=self.x_f[3]*dt; self.x_f[1]+=self.x_f[4]*dt; self.x_f[2]=normalize_angle(self.x_f[2]+self.x_f[5]*dt)
            self.x_f[3]+=ax_world*dt; self.x_f[4]+=ay_world*dt; self.x_f[5]=omega_z_local
            F_f=np.eye(9); F_f[0,3]=dt; F_f[1,4]=dt; F_f[2,5]=dt
            F_f[3,2]=(-ax_local*sin_th-ay_local*cos_th)*dt; F_f[4,2]=(ax_local*cos_th-ay_local*sin_th)*dt
            F_f[3,6]=-cos_th*dt; F_f[3,7]=sin_th*dt; F_f[4,6]=-sin_th*dt; F_f[4,7]=-cos_th*dt; F_f[5,8]=-dt
            self.P_f = F_f @ self.P_f @ F_f.T + self.Q_f * dt

            if self.is_leader_initialized:
                rot_mat_inv = np.array([[math.cos(-normalize_angle(self.x_f[2] - x_f_before[2])), -math.sin(-normalize_angle(self.x_f[2] - x_f_before[2]))],
                                        [math.sin(-normalize_angle(self.x_f[2] - x_f_before[2])),  math.cos(-normalize_angle(self.x_f[2] - x_f_before[2]))]])
                self.x_l = rot_mat_inv @ (self.x_l - self.x_f[:2]) + x_f_before[:2]
                self.P_l += self.Q_l * dt

            header = Header(stamp=imu_msg.header.stamp, frame_id='world')
            f_pose = PoseStamped(header=header)
            f_pose.pose.position.x, f_pose.pose.position.y = self.x_f[0], self.x_f[1]
            f_pose.pose.orientation = yaw_to_quaternion(self.x_f[2])
            self.follower_pose_pub.publish(f_pose)

            if self.is_leader_initialized:
                stable_point = PointStamped(header=header)
                stable_point.point.x, stable_point.point.y = self.x_l[0], self.x_l[1]
                self.stable_point_pub.publish(stable_point)

            if self.last_odom_pose:
                T_world_base=np.eye(4); T_world_base[:3,:3]=Rotation.from_euler('z',self.x_f[2]).as_matrix(); T_world_base[:2,3]=self.x_f[:2]
                T_odom_base=np.eye(4); q_o=self.last_odom_pose.orientation
                T_odom_base[:3,:3]=Rotation.from_quat([q_o.x,q_o.y,q_o.z,q_o.w]).as_matrix()
                T_odom_base[0,3]=self.last_odom_pose.position.x; T_odom_base[1,3]=self.last_odom_pose.position.y
                T_world_odom=T_world_base@np.linalg.inv(T_odom_base)
                t=TransformStamped(); t.header.stamp=imu_msg.header.stamp; t.header.frame_id='world'; t.child_frame_id='odom'
                t.transform.translation.x=T_world_odom[0,3]; t.transform.translation.y=T_world_odom[1,3]; t.transform.translation.z=T_world_odom[2,3]
                q_tf=Rotation.from_matrix(T_world_odom[:3,:3]).as_quat()
                t.transform.rotation=Quaternion(x=q_tf[0],y=q_tf[1],z=q_tf[2],w=q_tf[3])
                self.tf_broadcaster.sendTransform(t)

    def odom_update_callback(self, odom_msg):
        with self.lock:
            self.last_odom_pose = odom_msg.pose.pose
            if not self.is_initialized:
                q_odom=odom_msg.pose.pose.orientation
                odom_yaw=Rotation.from_quat([q_odom.x,q_odom.y,q_odom.z,q_odom.w]).as_euler('zyx')[0]
                self.initial_odom_pos=np.array([odom_msg.pose.pose.position.x,odom_msg.pose.pose.position.y])
                self.initial_odom_yaw=odom_yaw
                self.is_initialized=True; self.get_logger().info("Follower EKF Init Complete!")
                return
            
            q_odom=odom_msg.pose.pose.orientation
            odom_yaw=Rotation.from_quat([q_odom.x,q_odom.y,q_odom.z,q_odom.w]).as_euler('zyx')[0]
            odom_pos=np.array([odom_msg.pose.pose.position.x,odom_msg.pose.pose.position.y])
            relative_pos = odom_pos - self.initial_odom_pos
            relative_yaw = normalize_angle(odom_yaw - self.initial_odom_yaw)
            
            z = np.array([relative_pos[0], relative_pos[1], relative_yaw])
            h_x = self.x_f[[0, 1, 2]]
            H_f = np.zeros((3, 9)); H_f[0,0]=1; H_f[1,1]=1; H_f[2,2]=1
            y_err=z-h_x; y_err[2]=normalize_angle(y_err[2])
            S=H_f@self.P_f@H_f.T+self.R_odom; K=self.P_f@H_f.T@np.linalg.inv(S)
            self.x_f+=K@y_err; self.x_f[2]=normalize_angle(self.x_f[2])
            self.P_f=(np.eye(9)-K@H_f)@self.P_f
            
    def uwb_update_callback(self, uwb_msg):
        with self.lock:
            if not self.is_initialized or self.last_imu_msg is None or self.last_odom_pose is None: return
            
            if self.csv_writer:
                ts=rclpy.time.Time.from_msg(uwb_msg.header.stamp).nanoseconds; imu=self.last_imu_msg; odom=self.last_odom_pose
                self.csv_writer.writerow([
                    ts, uwb_msg.point.x, uwb_msg.point.y,
                    odom.position.x, odom.position.y, odom.orientation.w, odom.orientation.x, odom.orientation.y, odom.orientation.z,
                    imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z,
                    imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z,
                    imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z
                ])

            try:
                pf_x,pf_y,follower_yaw=self.x_f[0],self.x_f[1],self.x_f[2]
                d_a,d_b=uwb_msg.point.x,uwb_msg.point.y
                if not(d_a>0.1 and d_b>0.1): return
                Y_off=ANCHOR_WIDTH/2; py_local=(d_b**2-d_a**2)/(4*Y_off)
                px_sq=d_a**2-(py_local-Y_off)**2
                if px_sq<0: return
                px_local=ANCHOR_FORWARD_OFFSET+math.sqrt(px_sq)
                cos_th,sin_th=math.cos(follower_yaw),math.sin(follower_yaw)
                z=np.array([pf_x+(px_local*cos_th-py_local*sin_th),pf_y+(px_local*sin_th+py_local*cos_th)])
                
                self.pos_buffer.append(z) # 상태 판별을 위해 측정치 버퍼에 저장
                if not self.is_leader_initialized:
                    self.x_l=z; self.is_leader_initialized=True
                
                H_l=np.eye(2); y_err=z-self.x_l
                S=H_l@self.P_l@H_l.T+self.R_l; K=self.P_l@H_l.T@np.linalg.inv(S)
                self.x_l+=K@y_err; self.P_l=(np.eye(2)-K@H_l)@self.P_l

                if len(self.pos_buffer) == self.pos_buffer.maxlen:
                    poses = np.array(list(self.pos_buffer))
                    if np.var(poses[:,0]) + np.var(poses[:,1]) > self.get_parameter('motion_variance_threshold').value:
                        self.leader_state = 'Moving'
                    else: self.leader_state = 'Stationary'
                
                if self.leader_state == 'Moving': self.raw_measurements.append(z)
            except Exception: return

    def smooth_and_publish_path(self):
        with self.lock:
            if len(self.raw_measurements) < 5: return

            header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
            
            try:
                kf_smooth = KalmanFilter(transition_matrices=np.eye(2), observation_matrices=np.eye(2),
                                         transition_covariance=0.05*np.eye(2), observation_covariance=0.5*np.eye(2))
                (smoothed_means, _) = kf_smooth.smooth(self.raw_measurements)
                self.full_waypoints = [p for p in smoothed_means]
            except Exception as e:
                self.get_logger().warn(f"Path smoothing failed: {e}", throttle_duration_sec=5.0)
                self.full_waypoints = self.raw_measurements # 스무딩 실패시 원본 사용

            path_msg = self._create_waypoint_path_msg(header, self.full_waypoints)
            self.full_trajectory_pub.publish(path_msg)
    
    def _create_waypoint_path_msg(self, header, waypoints):
        path_msg = Path(header=header)
        for i, wp in enumerate(waypoints):
            pose=PoseStamped(header=header); pose.pose.position.x,pose.pose.position.y=wp[0],wp[1]
            if i<len(waypoints)-1: yaw=math.atan2(waypoints[i+1][1]-wp[1],waypoints[i+1][0]-wp[0])
            elif len(waypoints)>1: yaw=math.atan2(wp[1]-waypoints[i-1][1],wp[0]-waypoints[i-1][0])
            else: yaw=0.0
            pose.pose.orientation=yaw_to_quaternion(yaw)
            path_msg.poses.append(pose)
        return path_msg
    
    def __del__(self):
        if self.log_file:
            self.log_file.close()
            self.get_logger().info("Log file closed.")

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryEstimatorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()