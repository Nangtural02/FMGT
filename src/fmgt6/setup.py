from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'fmgt5'

setup(
    name=package_name,
    version='0.0.0',
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
    maintainer_email='jy@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'QANI = fmgt5.qani_publisher_node:main',
            'CLI = fmgt5.cli_publisher_node:main',
            'estimator = fmgt5.trajectory_estimator_node:main',
            'postprocessor = fmgt5.path_postprocessor_node:main',
            'follower = fmgt5.follower_control_node:main', 
        ],
    },
)
