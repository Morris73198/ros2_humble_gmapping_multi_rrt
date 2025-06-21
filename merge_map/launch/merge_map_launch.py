import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # 修改为读取 mm.rviz 文件
    rviz_file = os.path.join(get_package_share_directory('merge_map'), 'config', 'mm.rviz')
    
    # 检查文件是否存在（可选的调试信息）
    if os.path.exists(rviz_file):
        print(f"✅ Found RViz config file: {rviz_file}")
    else:
        print(f"❌ RViz config file not found: {rviz_file}")
        print("Available files in config directory:")
        config_dir = os.path.join(get_package_share_directory('merge_map'), 'config')
        if os.path.exists(config_dir):
            for file in os.listdir(config_dir):
                print(f"  - {file}")
    
    return LaunchDescription([
        # 启动 RViz2 并加载 mm.rviz 配置
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_file],
            parameters=[{'use_sim_time': True}]
        ),
        
        # 启动地图合并节点
        Node(
            package='merge_map',
            executable='merge_map',
            name='merge_map_node',
            output='screen',
            parameters=[{'use_sim_time': True}]
        ),
    ])