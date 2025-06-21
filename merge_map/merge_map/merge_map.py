#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point, TransformStamped, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from scipy import ndimage
from std_msgs.msg import ColorRGBA
import threading
import time
from collections import deque

class MergeMapNode(Node):
    def __init__(self):
        super().__init__('merge_map_node')
        
        # Declare parameters
        self.declare_parameter('map_size', 40.0)
        self.declare_parameter('origin_offset', -20.0)
        self.declare_parameter('dilation_size', 2)
        self.declare_parameter('robot_marker_size', 0.3)
        self.declare_parameter('marker_update_rate', 1.0)  # Hz
        self.declare_parameter('debug_mode', True)
        
        # 新增路徑記錄相關參數
        self.declare_parameter('path_recording_enabled', True)
        self.declare_parameter('path_max_points', 1000)  # 最大路徑點數
        self.declare_parameter('path_min_distance', 0.1)  # 記錄點之間的最小距離
        self.declare_parameter('path_line_width', 0.05)  # 路徑線寬
        self.declare_parameter('save_path_to_file', False)  # 是否保存路徑到文件
        
        # Get parameters
        self.map_size = self.get_parameter('map_size').value
        self.origin_offset = self.get_parameter('origin_offset').value
        self.dilation_size = self.get_parameter('dilation_size').value
        self.robot_marker_size = self.get_parameter('robot_marker_size').value
        self.marker_update_rate = self.get_parameter('marker_update_rate').value
        self.debug_mode = self.get_parameter('debug_mode').value
        
        # 路徑記錄參數
        self.path_recording_enabled = self.get_parameter('path_recording_enabled').value
        self.path_max_points = self.get_parameter('path_max_points').value
        self.path_min_distance = self.get_parameter('path_min_distance').value
        self.path_line_width = self.get_parameter('path_line_width').value
        self.save_path_to_file = self.get_parameter('save_path_to_file').value
        
        # Initialize publishers
        self.map_publisher = self.create_publisher(
            OccupancyGrid, 
            '/merge_map', 
            10
        )
        
        self.marker_publisher = self.create_publisher(
            MarkerArray,
            '/robot_markers',
            10
        )
        
        # 新增路徑可視化發布者
        self.path_publisher = self.create_publisher(
            MarkerArray,
            '/robot_paths',
            10
        )
        
        # Initialize subscribers with threading lock
        self.map_lock = threading.Lock()
        self.subscription1 = self.create_subscription(
            OccupancyGrid,
            '/robot1/map',
            lambda msg: self.map_callback(msg, 'robot1'),
            10
        )
        self.subscription2 = self.create_subscription(
            OccupancyGrid,
            '/robot2/map',
            lambda msg: self.map_callback(msg, 'robot2'),
            10
        )
        
        # Initialize map and pose data
        self.maps = {
            'robot1': None,
            'robot2': None
        }
        self.robot_poses = {
            'robot1': None,
            'robot2': None
        }
        self.pose_lock = threading.Lock()
        
        # 新增路徑記錄數據結構
        self.robot_paths = {
            'robot1': deque(maxlen=self.path_max_points),
            'robot2': deque(maxlen=self.path_max_points)
        }
        self.last_recorded_positions = {
            'robot1': None,
            'robot2': None
        }
        self.path_lock = threading.Lock()
        self.start_time = time.time()
        
        # Add pose subscribers
        self.pose_subs = {}
        for robot_name in ['robot1', 'robot2']:
            self.pose_subs[robot_name] = self.create_subscription(
                PoseStamped,
                f'/{robot_name}_pose',
                lambda msg, name=robot_name: self.pose_callback(msg, name),
                10
            )
        
        # Set robot colors
        self.robot_colors = {
            'robot1': ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0),  # Red
            'robot2': ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)   # Green
        }
        
        # 路徑顏色（稍微透明一些）
        self.path_colors = {
            'robot1': ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8),  # Semi-transparent Red
            'robot2': ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)   # Semi-transparent Green
        }
        
        # Create timer for map merging and marker updates
        self.timer = self.create_timer(1.0 / self.marker_update_rate, self.merge_and_publish)
        
        # 創建路徑保存定時器（如果啟用）
        if self.save_path_to_file:
            self.save_timer = self.create_timer(30.0, self.save_paths_to_file)  # 每30秒保存一次
        
        self.get_logger().info('Enhanced Map merge node initialized with path tracking')
        if self.path_recording_enabled:
            self.get_logger().info(f'Path recording enabled: max_points={self.path_max_points}, min_distance={self.path_min_distance}')

    def map_callback(self, msg, robot_name):
        """
        Callback for map messages with thread safety
        """
        with self.map_lock:
            self.maps[robot_name] = msg
            if self.debug_mode:
                self.get_logger().debug(f'Received map from {robot_name}')

    def pose_callback(self, msg, robot_name):
        """
        Callback for pose messages - 現在同時記錄路徑
        """
        with self.pose_lock:
            self.robot_poses[robot_name] = msg
            if self.debug_mode:
                self.get_logger().debug(
                    f'Updated {robot_name} pose: '
                    f'x={msg.pose.position.x:.2f}, '
                    f'y={msg.pose.position.y:.2f}'
                )
        
        # 記錄路徑點
        if self.path_recording_enabled:
            self.record_path_point(robot_name, msg)

    def record_path_point(self, robot_name, pose_msg):
        """
        記錄機器人路徑點
        """
        current_pos = (pose_msg.pose.position.x, pose_msg.pose.position.y)
        current_time = time.time() - self.start_time
        
        with self.path_lock:
            last_pos = self.last_recorded_positions[robot_name]
            
            # 檢查是否需要記錄新點（距離閾值）
            should_record = False
            if last_pos is None:
                should_record = True
            else:
                distance = np.sqrt((current_pos[0] - last_pos[0])**2 + 
                                 (current_pos[1] - last_pos[1])**2)
                if distance >= self.path_min_distance:
                    should_record = True
            
            if should_record:
                # 記錄路徑點：(x, y, timestamp)
                path_point = {
                    'x': current_pos[0],
                    'y': current_pos[1],
                    'timestamp': current_time,
                    'seq': len(self.robot_paths[robot_name])
                }
                
                self.robot_paths[robot_name].append(path_point)
                self.last_recorded_positions[robot_name] = current_pos
                
                if self.debug_mode and len(self.robot_paths[robot_name]) % 50 == 0:
                    self.get_logger().info(
                        f'{robot_name} path length: {len(self.robot_paths[robot_name])} points'
                    )

    def create_robot_marker(self, robot_name, position):
        """
        Create a marker for robot visualization
        """
        marker = Marker()
        marker.header.frame_id = "merge_map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = robot_name
        marker.id = 0 if robot_name == 'robot1' else 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position = position
        
        # Set orientation (upright)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        
        # Set size
        marker.scale.x = self.robot_marker_size
        marker.scale.y = self.robot_marker_size
        marker.scale.z = self.robot_marker_size / 3  # Height
        
        # Set color
        marker.color = self.robot_colors[robot_name]
        
        return marker

    def create_path_marker(self, robot_name):
        """
        創建機器人路徑的可視化標記
        """
        with self.path_lock:
            path_points = list(self.robot_paths[robot_name])
        
        if len(path_points) < 2:
            return None
        
        marker = Marker()
        marker.header.frame_id = "merge_map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"{robot_name}_path"
        marker.id = 10 if robot_name == 'robot1' else 11  # 不同的ID避免衝突
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        
        # Set line properties
        marker.scale.x = self.path_line_width
        marker.color = self.path_colors[robot_name]
        
        # Add all path points
        for point in path_points:
            p = Point()
            p.x = point['x']
            p.y = point['y']
            p.z = 0.05  # 稍微抬高，避免與地圖重疊
            marker.points.append(p)
        
        return marker

    def create_path_info_marker(self, robot_name):
        """
        創建路徑資訊文字標記
        """
        with self.path_lock:
            path_length = len(self.robot_paths[robot_name])
            if path_length == 0:
                return None
                
            # 計算總距離
            total_distance = 0.0
            path_points = list(self.robot_paths[robot_name])
            for i in range(1, len(path_points)):
                dx = path_points[i]['x'] - path_points[i-1]['x']
                dy = path_points[i]['y'] - path_points[i-1]['y']
                total_distance += np.sqrt(dx*dx + dy*dy)
        
        marker = Marker()
        marker.header.frame_id = "merge_map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"{robot_name}_path_info"
        marker.id = 20 if robot_name == 'robot1' else 21
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        # 位置設在地圖的角落
        if robot_name == 'robot1':
            marker.pose.position.x = self.origin_offset + 1.0
            marker.pose.position.y = self.origin_offset + self.map_size - 2.0
        else:
            marker.pose.position.x = self.origin_offset + 1.0
            marker.pose.position.y = self.origin_offset + self.map_size - 4.0
        marker.pose.position.z = 1.0
        
        marker.pose.orientation.w = 1.0
        
        # 文字內容
        marker.text = f"{robot_name}: {path_length} pts, {total_distance:.1f}m"
        
        # 文字大小和顏色
        marker.scale.z = 0.5
        marker.color = self.robot_colors[robot_name]
        
        return marker

    def get_robot_position(self, robot_name):
        """
        Get robot position from pose message
        """
        try:
            with self.pose_lock:
                pose_msg = self.robot_poses[robot_name]
                if pose_msg is None:
                    raise ValueError(f"No pose data available for {robot_name}")
                
                # 直接使用原始位置
                pos = Point(
                    x=pose_msg.pose.position.x,
                    y=pose_msg.pose.position.y,
                    z=0.0
                )
                
                if self.debug_mode:
                    self.get_logger().debug(
                        f'{robot_name} position: x={pos.x:.2f}, y={pos.y:.2f}'
                    )
                
                return pos
                    
        except Exception as e:
            self.get_logger().warning(f'Failed to get position for {robot_name}: {str(e)}')
            return None

    def merge_maps(self):
        """
        Merge maps with thread safety
        """
        with self.map_lock:
            if any(map_data is None for map_data in self.maps.values()):
                return None
            
            # Create merged map
            merged_map = OccupancyGrid()
            first_map = next(iter(self.maps.values()))
            merged_map.header = first_map.header
            merged_map.header.frame_id = 'merge_map'
            
            # Use minimum resolution
            merged_map.info.resolution = min(
                map_data.info.resolution for map_data in self.maps.values()
            )
            
            # Set origin and size
            merged_map.info.origin.position.x = self.origin_offset
            merged_map.info.origin.position.y = self.origin_offset
            merged_map.info.width = int(self.map_size / merged_map.info.resolution)
            merged_map.info.height = int(self.map_size / merged_map.info.resolution)
            
            # Initialize as unknown
            merged_data = np.full(
                (merged_map.info.height, merged_map.info.width),
                -1,
                dtype=np.int8
            )
            
            # Merge maps
            for map_data in self.maps.values():
                map_array = np.array(map_data.data).reshape(
                    map_data.info.height,
                    map_data.info.width
                )
                
                # Calculate coordinates in merged map
                for y in range(map_data.info.height):
                    for x in range(map_data.info.width):
                        merged_x = int((map_data.info.origin.position.x +
                                      x * map_data.info.resolution -
                                      self.origin_offset) /
                                     merged_map.info.resolution)
                        merged_y = int((map_data.info.origin.position.y +
                                      y * map_data.info.resolution -
                                      self.origin_offset) /
                                     merged_map.info.resolution)
                        
                        if (0 <= merged_x < merged_map.info.width and
                            0 <= merged_y < merged_map.info.height):
                            current_value = map_array[y, x]
                            existing_value = merged_data[merged_y, merged_x]
                            
                            # Merge rules
                            if existing_value == -1:
                                merged_data[merged_y, merged_x] = current_value
                            elif current_value == 100 or existing_value == 100:
                                merged_data[merged_y, merged_x] = 100
                            elif current_value >= 0:
                                merged_data[merged_y, merged_x] = current_value
            
            # Apply dilation to obstacles
            obstacle_mask = (merged_data == 100)
            kernel = np.ones((self.dilation_size, self.dilation_size), np.uint8)
            dilated_obstacles = ndimage.binary_dilation(obstacle_mask, kernel)
            merged_data[dilated_obstacles] = 100
            
            # Convert back to 1D list
            merged_map.data = merged_data.flatten().tolist()
            
            return merged_map
        
    def merge_and_publish(self):
        """
        Periodically merge maps and publish robot markers and paths
        """
        # Merge maps
        merged_map = self.merge_maps()
        if merged_map is None:
            if self.debug_mode:
                self.get_logger().debug('Not all maps available for merging')
            return
            
        # Publish merged map
        merged_map.header.stamp = self.get_clock().now().to_msg()
        self.map_publisher.publish(merged_map)
        
        # Create and publish robot markers
        marker_array = MarkerArray()
        for robot_name in self.maps.keys():
            position = self.get_robot_position(robot_name)
            if position is not None:
                marker = self.create_robot_marker(robot_name, position)
                marker_array.markers.append(marker)
        
        if marker_array.markers:
            if self.debug_mode:
                self.get_logger().debug(
                    f'Publishing {len(marker_array.markers)} robot markers'
                )
            self.marker_publisher.publish(marker_array)
        else:
            self.get_logger().warning('No robot positions available for visualization')
        
        # 發布路徑可視化
        if self.path_recording_enabled:
            self.publish_paths()

    def publish_paths(self):
        """
        發布機器人路徑可視化
        """
        path_marker_array = MarkerArray()
        
        for robot_name in ['robot1', 'robot2']:
            # 添加路徑線條
            path_marker = self.create_path_marker(robot_name)
            if path_marker is not None:
                path_marker_array.markers.append(path_marker)
            
            # 添加路徑資訊文字
            info_marker = self.create_path_info_marker(robot_name)
            if info_marker is not None:
                path_marker_array.markers.append(info_marker)
        
        if path_marker_array.markers:
            self.path_publisher.publish(path_marker_array)

    def save_paths_to_file(self):
        """
        保存路徑數據到文件
        """
        if not self.save_path_to_file:
            return
            
        try:
            import json
            import os
            
            # 創建保存目錄
            save_dir = "/tmp/robot_paths"
            os.makedirs(save_dir, exist_ok=True)
            
            timestamp = int(time.time())
            
            for robot_name in ['robot1', 'robot2']:
                with self.path_lock:
                    path_data = {
                        'robot_name': robot_name,
                        'timestamp': timestamp,
                        'total_points': len(self.robot_paths[robot_name]),
                        'path_points': list(self.robot_paths[robot_name])
                    }
                
                filename = f"{save_dir}/{robot_name}_path_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(path_data, f, indent=2)
                
                self.get_logger().info(f"Saved {robot_name} path to {filename}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to save paths: {str(e)}")

    def get_path_statistics(self):
        """
        獲取路徑統計資訊
        """
        stats = {}
        
        with self.path_lock:
            for robot_name in ['robot1', 'robot2']:
                path_points = list(self.robot_paths[robot_name])
                
                if len(path_points) == 0:
                    stats[robot_name] = {
                        'total_points': 0,
                        'total_distance': 0.0,
                        'duration': 0.0
                    }
                    continue
                
                # 計算總距離
                total_distance = 0.0
                for i in range(1, len(path_points)):
                    dx = path_points[i]['x'] - path_points[i-1]['x']
                    dy = path_points[i]['y'] - path_points[i-1]['y']
                    total_distance += np.sqrt(dx*dx + dy*dy)
                
                # 計算持續時間
                duration = path_points[-1]['timestamp'] - path_points[0]['timestamp'] if len(path_points) > 1 else 0.0
                
                stats[robot_name] = {
                    'total_points': len(path_points),
                    'total_distance': total_distance,
                    'duration': duration,
                    'average_speed': total_distance / duration if duration > 0 else 0.0
                }
        
        return stats

def main(args=None):
    rclpy.init(args=args)
    merge_map_node = MergeMapNode()
    
    try:
        rclpy.spin(merge_map_node)
    except KeyboardInterrupt:
        # 在退出前打印路徑統計
        if merge_map_node.path_recording_enabled:
            stats = merge_map_node.get_path_statistics()
            merge_map_node.get_logger().info("=== Final Path Statistics ===")
            for robot_name, robot_stats in stats.items():
                merge_map_node.get_logger().info(
                    f"{robot_name}: {robot_stats['total_points']} points, "
                    f"{robot_stats['total_distance']:.2f}m, "
                    f"{robot_stats['duration']:.1f}s, "
                    f"avg speed: {robot_stats['average_speed']:.2f}m/s"
                )
    except Exception as e:
        merge_map_node.get_logger().error(f'Unexpected error: {str(e)}')
    finally:
        # 最後保存路徑（如果啟用）
        if merge_map_node.save_path_to_file:
            merge_map_node.save_paths_to_file()
        
        merge_map_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()