# fmgt5/launch/fmgt.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    controller_node = path_controller_node

    uwb_publisher_node = Node(
        package='fmgt6',
        executable='uwb_publisher',
        name='uwb_publisher_node',
        output='screen',
        parameters=[
            {'firmware_build':'QANI'}
        ]
    )

    # 2. state_estimator_node 실행 설정
    # 이 노드는 UWB 거리값을 받아 위치와 속도를 추정합니다.
    follower_estimator_node = Node(
        package='fmgt6',
        executable='follower_estimator',
        name='follower_estimator_node',
        output='screen',
        parameters=[

        ]
    )
    leader_estimator_node = Node(
        package='fmgt6',
        executable='leader_estimator',
        name='leader_estimator_node',
        output='screen',
        pareameters=[

        ]
    )

    path_generator_node = Node(
            package='fmgt6',
            executable='generator',
            name='path_generator_node',
            output='screen',
            parameters=[

            ]
    )
    path_postprocessor_node = Node(
        package='fmgt6',
        executable='path_postprocessor',
        name='path_postprocessor_node',
        output='screen',
        parameters=[

        ]
    )

    path_controller_node = Node(
        package='fmgt6',
        executable='path_controller',
        name='path_controller_node',
        output='screen',
        parameters=[
            {'target_distance': 0.8},
            {'kp_pos': 0.7},
            {'kp_vel': 0.5},
            {'max_linear_speed': 0.22},
            {'max_angular_speed': 1.0}
        ]
    )

    point_controller_node = Node(
        package='fmgt6',
        executable='point_controller',
        name='point_controller_node',
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
        uwb_publisher_node,
        follower_estimator_node,
        leader_estimator_node,
        path_generator_node,
        path_postprocessor_node,
        controller_node
    ])