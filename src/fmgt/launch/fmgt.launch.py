from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    uwb_publisher_node = Node(
        package='fmgt',
        executable='uwb_publisher',
        name='uwb_publisher_node',
        output='screen',
        parameters=[
            {'firmware_build':'QANI'}
        ]
    )
    follower_estimator_node = Node(
        package='fmgt',
        executable='follower_estimator',
        name='follower_estimator_node',
        output='screen',
        # parameters=[        ]
    )
    leader_estimator_node = Node(
        package='fmgt',
        executable='leader_estimator',
        name='leader_estimator_node',
        output='screen',
        # pareameters=[        ]
    )
    path_generator_node = Node(
            package='fmgt',
            executable='path_generator',
            name='path_generator_node',
            output='screen',
            # parameters=[            ]
    )
    path_postprocessor_node = Node(
        package='fmgt',
        executable='path_postprocessor',
        name='path_postprocessor_node',
        output='screen',
        # parameters=[        ]
    )
    path_controller_node = Node(
        package='fmgt',
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
    # LaunchDescription 객체에 실행할 노드들을 리스트로 담아 반환
    return LaunchDescription([
        uwb_publisher_node,
        follower_estimator_node,
        leader_estimator_node,
        path_generator_node,
        path_postprocessor_node,
        path_controller_node
    ])