# fmgt5/launch/fmgt.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    3개의 노드(publisher, estimator, follower)를 동시에 실행하는 런치 파일을 생성합니다.
    """
    debug_mode = True
    follow_distance = 1.0
    rdp_epsilon = 0.1 # 경로 단순화 강도. 클수록 더 단순해짐
    leader_measurement_variance = 0.3**2 # UWB 측정값 신뢰도. 클수록 부드러워지고 반응이 느려
    
    # 1. uwb_publisher_node 실행 설정
    # 이 노드는 시리얼 포트와 직접 통신합니다.
    publisher_node = Node(
        package='fmgt5',
        executable='QANI',
        name='qani_publisher_node',
        output='screen'  # 노드의 로그를 터미널에 바로 출력
    )

    # 2. state_estimator_node 실행 설정
    # 이 노드는 UWB 거리값을 받아 위치와 속도를 추정합니다.
    estimator_node = Node(
        package='fmgt5',
        executable='estimator',
        name='trajectory_estimator_node',
        output='screen'
    )
    Node(
            package='fmgt5',
            executable='postprocessor',
            name='path_postprocessor_node',
            output='screen',
            parameters=[
                {'follow_distance': follow_distance},
                {'rdp_epsilon': rdp_epsilon},
            ]
        ),

    # 3. smooth_follower_node 실행 설정
    # 이 노드는 추정된 상태를 바탕으로 로봇을 제어합니다.
    follower_node = Node(
        package='fmgt5',
        executable='follower',
        name='follower_control_node',
        output='screen',
        # 런치 파일에서 직접 파라미터를 설정할 수 있습니다.
        parameters=[
            {'target_distance': 0.8},
            {'kp_pos': 0.7},
            {'kp_vel': 0.5},
            {'max_linear_speed': 0.22},
            {'max_angular_speed': 1.0}
        ]
    )

    # LaunchDescription 객체에 실행할 노드들을 리스트로 담아 반환
    return LaunchDescription([
        publisher_node,
        estimator_node,
        follower_node
    ])