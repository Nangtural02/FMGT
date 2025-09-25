from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # 테스트할 fmgt 버전 (로직이 있는 패키지)
    target_logic_pkg = 'fmgt5' 
    
    # 안정적인 위치 추정기 버전 (필요시 'fmgt4' 등으로 변경)
    target_estimator_pkg = 'fmgt5'

    return LaunchDescription([
        # 1. 위치 추정 노드 실행 (실제 로봇의 센서 데이터를 사용)
        #    - 실행 전, 로봇의 IMU, Odometry, UWB Publisher가 켜져 있어야 합니다.
        Node(
            package=target_estimator_pkg,
            # fmgt5의 estimator 노드 이름이 trajectory_estimator_node.py 이므로,
            # setup.py에 등록된 executable 이름으로 변경해야 합니다. 
            # 예시: executable='trajectory_estimator'
            executable='estimator', 
            name='state_estimator'
        ),

        # 2. 테스트 대상인 제어 노드 실행
        Node(
            package=target_logic_pkg,
            executable='follower',
            name='follower_control_node',
            output='screen',
        ),

        # 3. 가상 경로 발행 노드 실행
        Node(
            package='fmgt_test',
            executable='virtual_path_publisher',
            name='virtual_path_publisher',
            output='screen',
            parameters=[{'path_type': 's_curve'}]
        ),
    ])