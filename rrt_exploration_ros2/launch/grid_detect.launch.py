#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    """Generate launch description for grid frontier detector node."""
    
    # Declare the launch arguments
    declare_map_topic = DeclareLaunchArgument(
        'map_topic',
        default_value='/merge_map',
        description='Topic name for merged map'
    )
    
    declare_detection_frequency = DeclareLaunchArgument(
        'detection_frequency',
        default_value='2.0',
        description='Frontier detection frequency in Hz'
    )
    
    declare_output_all_frontiers = DeclareLaunchArgument(
        'output_all_frontiers',
        default_value='true',
        description='Whether to output all found frontiers without filtering'
    )
    
    declare_boundary_required = DeclareLaunchArgument(
        'boundary_required',
        default_value='true',
        description='Whether boundary is required before starting detection'
    )
    
    # Create the grid frontier detector node
    grid_frontier_detector_node = Node(
        package='rrt_exploration_ros2',
        executable='grid_frontier_detector',
        name='grid_frontier_detector',
        output='screen',
        parameters=[{
            'map_topic': LaunchConfiguration('map_topic'),
            'detection_frequency': LaunchConfiguration('detection_frequency'),
        }],
        remappings=[
            # Input topics
            ('merge_map', LaunchConfiguration('map_topic')),
            
            # Output topics (keeping default names for compatibility)
            # '/detected_points' -> published by the node
            # '/visualization_marker_array' -> published by the node  
            # '/found' -> published by the node
        ]
    )
    
    return LaunchDescription([
        declare_map_topic,
        declare_detection_frequency,
        declare_output_all_frontiers,
        declare_boundary_required,
        grid_frontier_detector_node
    ])