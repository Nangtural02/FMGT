from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'fmgt6'

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
            'uwb_publisher = fmgt6.uwb_publisher_node:main',
            'follower_estimator = fmgt6.follower_esitimator_node:main'
            'leader_estimator = fmgt6.leader_estimator_node:main',

            'point_controller = fmgt6.point_controller_node:main', 
            'point_postprocessor = fmgt6.point_postprocessor_node:main',

            'path_generator = fmgt6.path_gererator_node:main'
            'path_postprocessor = fmgt6.path_postprocessor_node:main',
            'path_controller = fmgt6.path_controller_node:main', 
        ],
    },
)
