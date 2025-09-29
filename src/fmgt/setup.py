from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'fmgt'

setup(
    name=package_name,
    version='6.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jy',
    maintainer_email='nangtural02@gmail.com',
    description='Follow Me, Go There',
    license='Closed License for Hyundai whereb project.',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            #device driver
            'uwb_publisher = fmgt.device.uwb_publisher_node:main',
            'follower_estimator = fmgt.device.follower_esitimator_node:main',
            'leader_estimator = fmgt.device.leader_estimator_node:main',

            #point
            'point_controller = fmgt.point.point_controller_node:main', 
            'point_postprocessor = fmgt.point.point_postprocessor_node:main',

            #path
            'path_generator = fmgt.path.path_generator_node:main',
            'path_postprocessor = fmgt.path.path_postprocessor_node:main',
            'path_controller = fmgt.path.path_controller_node:main', 

            #simulation
            'robot_simulator = fmgt.simulation.robot_simulator_node:main',
            'leader_teleop = fmgt.simulation.leader_teleop_node:main',
        ],
    },
)
