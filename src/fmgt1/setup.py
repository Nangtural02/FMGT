from setuptools import find_packages, setup

package_name = 'fmgt1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'publisher = fmgt1.uwb_publisher_node:main',
            'calculator = fmgt1.position_calculator_node:main',
            'controller = fmgt1.follower_control_node_stateful:main', 
        ],
    },
)
