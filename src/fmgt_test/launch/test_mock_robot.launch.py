from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 테스트할 fmgt 버전
    target_pkg = 'fmgt5'

    return LaunchDescription([
        # 1. 테스트 대상인 제어 노드 실행
        Node(
            package=target_pkg,
            executable='follower',
            name='follower_control_node',
            output='screen',
            parameters=[{}]
        ),

        # 2. 가상 경로 발행 노드 실행
        Node(
            package='fmgt_test',
            executable='virtual_path_publisher',
            name='virtual_path_publisher',
            output='screen',
            parameters=[{'path_type': 'straight'}]
        ),
        
        # 3. 가상 로봇 위치 발행 노드 실행 (state_estimator 대체)
        Node(
            package='fmgt_test',
            executable='mock_pose_publisher',
            name='mock_pose_publisher',
            output='screen',
        ),
    ])