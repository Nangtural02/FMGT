from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'fmgt2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch.py')))
    ],
    install_requires=['setuptools','numpy'],
    zip_safe=True,
    maintainer='jy',
    maintainer_email='jy@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'publisher = fmgt2.uwb_publisher_node:main',
            'estimator = fmgt2.state_estimator_node:main',
            'follower = fmgt2.smooth_follower_node:main', 
        ],
    },
)
