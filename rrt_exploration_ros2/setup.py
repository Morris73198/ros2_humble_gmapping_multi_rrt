from setuptools import setup
import os
from glob import glob

package_name = 'rrt_exploration_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 添加launch文件安装
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        # 如果有config文件也添加
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        # 如果有其他资源文件
        ('share/' + package_name + '/saved_models',
            glob('rrt_exploration_ros2/saved_models/*.h5')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='RRT exploration for ROS 2',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'boundary_node = rrt_exploration_ros2.boundary_node:main',
            'global_rrt_detector = rrt_exploration_ros2.global_detector_node:main',
            'local_detector_node = rrt_exploration_ros2.local_detector_node:main',
            'filter_node = rrt_exploration_ros2.filter_node:main',
            'assigner = rrt_exploration_ros2.assigner:main',
            'test_assigner = rrt_exploration_ros2.test_assigner:main',
        ],
    },
)