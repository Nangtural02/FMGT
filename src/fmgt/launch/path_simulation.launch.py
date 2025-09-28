# launch/path_simulation.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """시뮬레이션 환경을 위한 런치 파일"""

    # 1. 시뮬레이터 노드: UWB, 로봇/리더의 위치를 시뮬레이션
    robot_simulator_node = Node(
        package='fmgt', # 실제 패키지 이름으로 수정하세요
        executable='robot_simulator', # setup.py에 등록된 이름
        name='robot_simulator_node',
        output='screen'
    )


    # 2. 개발 중인 알고리즘 노드들
    #    (실행 파일 이름은 setup.py에 정의된 이름을 사용해야 합니다)
    leader_estimator_node = Node(
        package='fmgt',
        executable='leader_estimator', # 예시 실행파일 이름
        name='leader_estimator_node',
        output='screen'
    )
    
    path_generator_node = Node(
            package='fmgt',
            executable='path_generator', # 예시 실행파일 이름
            name='path_generator_node',
            output='screen'
    )

    path_postprocessor_node = Node(
        package= 'fmgt',
        executable='path_postprocessor', # 예시 실행파일 이름
        name='path_postprocessor_node',
        output='screen'
    )
    
    path_controller_node = Node(
        package='fmgt',
        executable='path_controller', # 예시 실행파일 이름
        name='path_controller_node',
        output='screen'
    )

    return LaunchDescription([
        # 시뮬레이션 환경
        robot_simulator_node,
        # 테스트 대상 알고리즘
        leader_estimator_node,
        path_generator_node,
        path_postprocessor_node,
        path_controller_node
    ])