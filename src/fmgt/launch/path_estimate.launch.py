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

    return LaunchDescription([
        uwb_publisher_node,
        follower_estimator_node,
        leader_estimator_node,
        path_generator_node,
        path_postprocessor_node,
    ])