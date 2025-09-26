# fmgt5/launch/fmgt.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    uwb_publisher_node = Node(
        package='fmgt6',
        executable='uwb_publisher',
        name='uwb_publisher_node',
        output='screen',
        parameters=[
            {'firmware_build':'QANI'}
        ]
    )

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
    point_postprocessor_node = Node(
        package= 'fmgt6',
        executable='poinst_postprocessor',
        name='point_postprocessor_node',
        output='screen',
        parameters=[

        ]
    )

    point_controller_node = Node(
        package='fmgt6',
        executable='point_controller',
        name='point_controller_node',
        output='screen',
        parameters=[
            
        ]
    )
    # LaunchDescription 객체에 실행할 노드들을 리스트로 담아 반환
    return LaunchDescription([
        uwb_publisher_node,
        follower_estimator_node,
        leader_estimator_node,
        point_postprocessor_node,
        point_controller_node
    ])