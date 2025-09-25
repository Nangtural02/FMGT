# ~/dev/ROS2/FMGT/src/fmgt_test/setup.py (수정된 버전)

from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'fmgt_test'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), 
         glob(os.path.join('launch', '*.launch.py'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='nangtural02',
    maintainer_email='nangtural02@todo.todo',
    description='Test package for FMGT project',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'virtual_path_publisher = fmgt_test.virtual_path_publisher:main',
            'mock_pose_publisher = fmgt_test.mock_pose_publisher:main'
        ],
    },
)