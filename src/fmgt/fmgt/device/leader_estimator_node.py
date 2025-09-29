# 파일명: leader_estimator_node.py
"""
Leader Estimator Node (Active Alignment & Robust Estimation)

이 노드는 UWB 측정의 모호성을 능동적으로 해결하는 시스템의 핵심 두뇌 역할을 합니다.
불확실성이 높다고 판단되면, Controller에 'Align Mode'를 요청하고, 상태가 안정된 후에
정제된 위치 데이터를 발행하여 전체 시스템의 안정성을 확보합니다.

- 핵심 기능:
  1. 해 선택: 과거 위치 기록의 '이동 평균 기반 선형 예측'을 통해 가장 신뢰도 높은 해를 선택합니다.
  2. 불확실성 계산: 두 해와 예측점 간의 거리 비율을 이용해 현재 측정의 불확실성을 0.0(확실)~1.0(불확실)로 정량화합니다.
  3. Align Mode 관리: '불확실성의 이동 평균'이 임계값을 넘으면 Align Mode를 활성화/비활성화합니다.
  4. 버퍼링 및 순차 발행: Align Mode 동안에는 위치 발행을 멈추고 데이터를 버퍼에 저장했다가,
     Align이 완료되면 버퍼의 데이터를 순차적으로 빠르게 발행하여 Path Generator가 깨끗한 경로를
     생성하도록 합니다.

- 구독 (Subscriptions):
  - /follower/estimated_pose (geometry_msgs/PoseStamped): 팔로워의 추정된 위치 및 자세
  - raw_uwb_distances (geometry_msgs/PointStamped): UWB 태그로부터의 거리 (x: d_a, y: d_b)

- 발행 (Publications):
  - /leader/raw_point (geometry_msgs/PointStamped): 월드 좌표계에서 계산된 리더의 위치(x,y)와
                                                  불확실성(z)을 담은 메시지
  - /align_needed (std_msgs/Bool): Path Controller에 Align Mode 진입/해제를 요청하는 신호

- 파라미터 (Parameters):
  - anchor_forward_offset, anchor_width (double): UWB 앵커의 물리적 위치
  - history_size_for_solution (int): 해 선택 및 예측에 사용할 과거 위치 데이터의 최대 개수
  - ma_window_size (int): 이동 평균 계산에 사용할 윈도우 크기
  - uncertainty_history_size (int): Align Mode 결정을 위한 불확실성 값 저장 개수
  - align_entry_threshold (double): Align Mode에 진입하는 불확실성 이동 평균 임계값 (0.0 ~ 1.0)
  - align_exit_threshold (double): Align Mode를 해제하는 불확실성 이동 평균 임계값 (0.0 ~ 1.0)
"""
import rclpy
import numpy as np
import math
import threading
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Bool
from scipy.spatial.transform import Rotation
from collections import deque
import time

class LeaderEstimatorNode(Node):
    def __init__(self):
        super().__init__('leader_estimator_node')
        
        # --- 파라미터 선언 (새로운 불확실성 지표에 맞게 수정) ---
        self.declare_parameter('anchor_forward_offset', 0.25)
        self.declare_parameter('anchor_width', 0.4)
        self.declare_parameter('history_size_for_solution', 10)
        self.declare_parameter('ma_window_size', 5)
        self.declare_parameter('uncertainty_history_size', 10)
        self.declare_parameter('align_entry_threshold', 0.9)  # 값이 1.0에 가까울수록 불확실
        self.declare_parameter('align_exit_threshold', 0.6)   # 값이 0.0에 가까울수록 확실

        # --- 발행자 ---
        self.raw_point_pub = self.create_publisher(PointStamped, '/leader/raw_point', 10)
        self.align_needed_pub = self.create_publisher(Bool, '/align_needed', 10)
        
        # --- 내부 상태 변수 ---
        self.lock = threading.Lock()
        self.latest_follower_pose = None
        
        hist_size = self.get_parameter('history_size_for_solution').value
        self.valid_history = deque(maxlen=hist_size)
        
        uncertainty_hist_size = self.get_parameter('uncertainty_history_size').value
        self.uncertainty_history = deque(maxlen=uncertainty_hist_size)
        
        self.is_align_mode = False
        self.align_buffer = []

        # --- 구독자 ---
        self.follower_pose_sub = self.create_subscription(PoseStamped, '/follower/estimated_pose', self.follower_pose_callback, 10)
        self.uwb_sub = self.create_subscription(PointStamped, 'raw_uwb_distances', self.uwb_update_callback, QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1))
            
        self.get_logger().info("Leader Estimator (Active Alignment, Corrected Logic) 시작됨.")

    def follower_pose_callback(self, msg: PoseStamped):
        with self.lock: self.latest_follower_pose = msg

    def uwb_update_callback(self, uwb_msg):
        with self.lock:
            if self.latest_follower_pose is None: return
            
            solutions = self._calculate_possible_solutions(self.latest_follower_pose, uwb_msg)
            if solutions is None: return
            p_world_sol1, p_world_sol2 = solutions

            reference_point = self._get_reference_point()
            chosen_pos, uncertainty = self._choose_best_solution(p_world_sol1, p_world_sol2, reference_point)
            
            self.uncertainty_history.append(uncertainty)
            avg_uncertainty = np.mean(self.uncertainty_history) if self.uncertainty_history else 0.0
            
            entry_thresh = self.get_parameter('align_entry_threshold').value
            exit_thresh = self.get_parameter('align_exit_threshold').value

            # --- Align Mode 관리 (올바른 로직) ---
            # 불확실성(avg_uncertainty)이 클 때(entry_thresh 초과) Align Mode 진입
            if not self.is_align_mode and len(self.uncertainty_history) == self.uncertainty_history.maxlen and avg_uncertainty > entry_thresh:
                self._enter_align_mode()
            # 불확실성(avg_uncertainty)이 작을 때(exit_thresh 미만) Align Mode 해제
            elif self.is_align_mode and avg_uncertainty < exit_thresh:
                self._exit_align_mode()

            # 데이터 발행 또는 버퍼링
            if self.is_align_mode:
                self.align_buffer.append((chosen_pos, uncertainty))
            else:
                self.valid_history.append(chosen_pos)
                self._publish_raw_point(chosen_pos, uncertainty, uwb_msg.header.stamp)

    def _calculate_possible_solutions(self, pose_msg, uwb_msg):
        try:
            # ... (이 함수 내용은 이전과 동일하여 생략하지 않고 모두 포함) ...
            anchor_forward_offset = self.get_parameter('anchor_forward_offset').value
            anchor_width = self.get_parameter('anchor_width').value
            pf_x, pf_y = pose_msg.pose.position.x, pose_msg.pose.position.y
            q = pose_msg.pose.orientation
            follower_yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('zyx')[0]
            d_a, d_b = uwb_msg.point.x, uwb_msg.point.y
            if not(d_a > 0.1 and d_b > 0.1): return None
            Y_off = anchor_width / 2
            py_local = (d_b**2 - d_a**2) / (4 * Y_off)
            px_sq = d_a**2 - (py_local - Y_off)**2
            if px_sq < 0: return None
            px_sqrt = math.sqrt(px_sq)
            px_local_sol1, px_local_sol2 = anchor_forward_offset + px_sqrt, anchor_forward_offset - px_sqrt
            cos_th, sin_th = math.cos(follower_yaw), math.sin(follower_yaw)
            rot_mat = np.array([[cos_th, -sin_th], [sin_th, cos_th]])
            p_follower = np.array([pf_x, pf_y])
            p_world_sol1 = p_follower + rot_mat @ np.array([px_local_sol1, py_local])
            p_world_sol2 = p_follower + rot_mat @ np.array([px_local_sol2, py_local])
            return p_world_sol1, p_world_sol2
        except Exception: return None

    def _get_reference_point(self):
        ma_window = self.get_parameter('ma_window_size').value
        if len(self.valid_history) < ma_window + 1:
            return self.valid_history[-1] if self.valid_history else None
        history_points = np.array(self.valid_history)
        ma1 = np.mean(history_points[-(ma_window+1):-1], axis=0)
        ma2 = np.mean(history_points[-ma_window:], axis=0)
        velocity_ma = ma2 - ma1
        return ma2 + velocity_ma

    def _choose_best_solution(self, sol1, sol2, reference_point):
        """
        [로직 수정] 두 해 중에서 기준점과 더 가까운 것을 선택하고,
        '정규화된 거리 비율'을 이용해 0.0(확실)~1.0(불확실) 사이의 불확실성을 계산합니다.
        """
        if reference_point is None:
            # 기준점이 없으면(초기 상태), 전방 해를 선택하고 불확실성은 최대로 설정
            return sol1, 1.0

        dist_to_sol1 = np.linalg.norm(reference_point - sol1)
        dist_to_sol2 = np.linalg.norm(reference_point - sol2)
        
        # 불확실성 계산: min/max 비율 사용. 1.0에 가까울수록 불확실.
        if max(dist_to_sol1, dist_to_sol2) < 1e-6: # 분모가 0이 되는 것 방지
            uncertainty = 1.0
        else:
            uncertainty = min(dist_to_sol1, dist_to_sol2) / max(dist_to_sol1, dist_to_sol2)

        return (sol1, uncertainty) if dist_to_sol1 <= dist_to_sol2 else (sol2, uncertainty)

    def _enter_align_mode(self):
        self.get_logger().warn("불확실성 증가! Align Mode를 요청합니다.")
        self.is_align_mode = True
        self.align_buffer.clear()
        self.align_needed_pub.publish(Bool(data=True))

    def _exit_align_mode(self):
        self.get_logger().info("상태 안정화. Align Mode를 종료하고 경로를 발행합니다.")
        self.is_align_mode = False
        self.align_needed_pub.publish(Bool(data=False))
        self.get_logger().info(f"{len(self.align_buffer)}개의 버퍼링된 데이터 순차 발행 시작...")
        
        # 버퍼에 쌓인 데이터 중 올바른 해를 '현재의 안정된' 예측 기준으로 다시 한번 검증하여 발행
        for point, uncertainty in self.align_buffer:
            ref_point = self._get_reference_point()
            # 버퍼의 점은 이미 계산된 해이므로, 이를 sol1, sol2 자리에 모두 넣어줌
            chosen_pos, new_uncertainty = self._choose_best_solution(point, point, ref_point)
            self.valid_history.append(chosen_pos)
            self._publish_raw_point(chosen_pos, new_uncertainty, self.get_clock().now().to_msg())
            time.sleep(0.01) # Path_Generator가 처리할 시간을 주기 위한 약간의 지연
        self.align_buffer.clear()
        self.get_logger().info("버퍼 데이터 발행 완료.")

    def _publish_raw_point(self, point, uncertainty, stamp):
        msg = PointStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = 'world'
        msg.point.x, msg.point.y = point[0], point[1]
        msg.point.z = uncertainty # Z좌표에 0.0~1.0 사이의 불확실성 전달
        self.raw_point_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args); node = LeaderEstimatorNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()