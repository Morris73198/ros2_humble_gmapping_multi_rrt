#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Twist
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String, ColorRGBA
import numpy as np
import cv2
import socket
import json

def send_state_and_get_target(state, host='127.0.0.1', port=9000):
    """Send state and receive target with enhanced error handling"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((host, port))
        
        data_to_send = json.dumps(state, ensure_ascii=False).encode('utf-8')
        s.sendall(data_to_send)
        
        all_data = b''
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            all_data += chunk
            try:
                target = json.loads(all_data.decode('utf-8'))
                break
            except json.JSONDecodeError:
                continue
                
        s.close()
        return target
    except Exception as e:
        print(f"Socket communication error: {e}")
        if 's' in locals():
            s.close()
        return {"target_point": None, "error": str(e)}

class TestSocketAssigner(Node):
    def __init__(self):
        super().__init__('test_socket_assigner')

        # Basic state
        self.robot1_pose = None
        self.robot2_pose = None
        self.available_points = []
        self.assigned_targets = {'robot1': None, 'robot2': None}
        
        # Motion detection
        self.robot_last_pose = {'robot1': None, 'robot2': None}
        self.robot_cmd_vel = {'robot1': None, 'robot2': None}
        self.robot_last_cmd_time = {'robot1': self.get_clock().now(), 'robot2': self.get_clock().now()}
        self.robot_last_move_time = {'robot1': self.get_clock().now(), 'robot2': self.get_clock().now()}
        self.robot_static_time = {'robot1': 0.0, 'robot2': 0.0}
        
        # Target locking mechanism
        self.target_locked = {'robot1': False, 'robot2': False}
        self.target_assignment_time = {'robot1': None, 'robot2': None}
        
        # Parameters
        self.static_threshold = 15.0
        self.movement_threshold = 0.1
        self.target_threshold = 0.8
        self.exclusion_radius = 2.0
        self.min_target_distance = 1.5
        self.cmd_vel_threshold = 0.05
        self.no_cmd_timeout = 5.0
        
        # Map related
        self.map_data = None
        self.map_resolution = None
        self.map_width = None
        self.map_height = None
        self.map_origin = None
        self.processed_map = None
        self.max_frontiers = 50

        # ROS2 communication
        self.setup_subscribers()
        self.setup_publishers()

        # Timers
        self.create_timer(8.0, self.assign_targets)
        self.create_timer(2.0, self.check_robot_status)
        self.create_timer(0.2, self.publish_visualization)
        self.create_timer(5.0, self.publish_debug_info)

        self.get_logger().info('Test Socket Assigner started')
        self.get_logger().info(f'Static threshold: {self.static_threshold} seconds')

    def setup_subscribers(self):
        # Basic subscriptions
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/merge_map', self.map_callback, 10)
        self.robot1_pose_sub = self.create_subscription(
            PoseStamped, '/robot1_pose', self.robot1_pose_callback, 10)
        self.robot2_pose_sub = self.create_subscription(
            PoseStamped, '/robot2_pose', self.robot2_pose_callback, 10)
        self.filtered_points_sub = self.create_subscription(
            MarkerArray, '/filtered_points', self.filtered_points_callback, 10)

        # Subscribe to command velocities
        self.robot1_cmd_vel_sub = self.create_subscription(
            Twist, '/robot1/cmd_vel', 
            lambda msg: self.cmd_vel_callback(msg, 'robot1'), 10)
        self.robot2_cmd_vel_sub = self.create_subscription(
            Twist, '/robot2/cmd_vel', 
            lambda msg: self.cmd_vel_callback(msg, 'robot2'), 10)

    def setup_publishers(self):
        self.robot1_target_pub = self.create_publisher(
            PoseStamped, '/robot1/goal_pose', 10)
        self.robot2_target_pub = self.create_publisher(
            PoseStamped, '/robot2/goal_pose', 10)
        self.target_viz_pub = self.create_publisher(
            MarkerArray, '/assigned_targets_viz', 10)
        self.debug_pub = self.create_publisher(
            String, '/assigner/debug', 10)

    def cmd_vel_callback(self, msg, robot_name):
        """Record robot command velocity"""
        total_cmd_vel = abs(msg.linear.x) + abs(msg.linear.y) + abs(msg.angular.z)
        self.robot_cmd_vel[robot_name] = total_cmd_vel
        self.robot_last_cmd_time[robot_name] = self.get_clock().now()
        
        # Reset static time if command velocity is present
        if total_cmd_vel > self.cmd_vel_threshold:
            self.robot_static_time[robot_name] = 0.0
            self.robot_last_move_time[robot_name] = self.get_clock().now()

    def map_callback(self, msg):
        """Map callback"""
        try:
            self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
            self.map_resolution = msg.info.resolution
            self.map_width = msg.info.width
            self.map_height = msg.info.height
            self.map_origin = msg.info.origin

            map_array = self.map_data.copy()
            map_binary = np.zeros_like(map_array, dtype=np.uint8)
            map_binary[map_array == 0] = 255
            map_binary[map_array == 100] = 0
            map_binary[map_array == -1] = 127

            resized_map = cv2.resize(map_binary, (84, 84), interpolation=cv2.INTER_LINEAR)
            normalized_map = resized_map.astype(np.float32) / 255.0
            self.processed_map = np.expand_dims(normalized_map, axis=-1)
            
        except Exception as e:
            self.get_logger().error(f'Map processing error: {e}')
            self.processed_map = None

    def robot1_pose_callback(self, msg):
        self.robot1_pose = msg.pose

    def robot2_pose_callback(self, msg):
        self.robot2_pose = msg.pose

    def filtered_points_callback(self, msg):
        old_count = len(self.available_points)
        self.available_points = []
        if msg.markers:
            for marker in msg.markers:
                self.available_points.extend([(p.x, p.y) for p in marker.points])
        
        if len(self.available_points) != old_count:
            self.get_logger().info(f'Updated frontier points: {old_count} -> {len(self.available_points)}')

    def is_robot_actually_moving(self, robot_name):
        """Improved motion detection logic"""
        current_time = self.get_clock().now()
        
        # Check if there is command velocity
        if self.robot_cmd_vel[robot_name] is not None:
            # Check if command velocity is sufficient
            if self.robot_cmd_vel[robot_name] > self.cmd_vel_threshold:
                return True, "Sufficient command velocity"
            
            # Check if no command velocity for too long
            time_since_cmd = (current_time - self.robot_last_cmd_time[robot_name]).nanoseconds / 1e9
            if time_since_cmd > self.no_cmd_timeout:
                return False, f"No command velocity for {self.no_cmd_timeout} seconds"
        
        # Check position change
        current_pose = getattr(self, f'{robot_name}_pose')
        if current_pose and self.robot_last_pose[robot_name]:
            current_pos = [current_pose.position.x, current_pose.position.y]
            last_pos = [
                self.robot_last_pose[robot_name].position.x,
                self.robot_last_pose[robot_name].position.y
            ]
            
            movement_distance = np.sqrt(
                (current_pos[0] - last_pos[0])**2 + 
                (current_pos[1] - last_pos[1])**2
            )
            
            if movement_distance > self.movement_threshold:
                return True, f"Position change {movement_distance:.3f}m"
            else:
                return False, f"Position change too small {movement_distance:.3f}m"
        
        return False, "Cannot determine"

    def check_robot_status(self):
        """Improved robot status check"""
        current_time = self.get_clock().now()
        
        robots = {
            'robot1': self.robot1_pose,
            'robot2': self.robot2_pose
        }
        
        for robot_name, current_pose in robots.items():
            if current_pose is None:
                continue
                
            current_pos = [current_pose.position.x, current_pose.position.y]
            
            # Check if reached target
            if self.assigned_targets[robot_name] is not None and self.target_locked[robot_name]:
                target_pos = self.assigned_targets[robot_name]
                distance_to_target = np.sqrt(
                    (current_pos[0] - target_pos[0])**2 + 
                    (current_pos[1] - target_pos[1])**2
                )
                
                if distance_to_target < self.target_threshold:
                    self.get_logger().info(f'{robot_name} reached target, unlocking and allowing reassignment')
                    self._clear_robot_target(robot_name, current_time)
                    continue
            
            # Check motion
            is_moving, reason = self.is_robot_actually_moving(robot_name)
            
            if is_moving:
                # Robot is moving
                self.robot_static_time[robot_name] = 0.0
                self.robot_last_move_time[robot_name] = current_time
                self.get_logger().debug(f'{robot_name} is moving: {reason}')
            else:
                # Robot is not moving, accumulate static time
                time_diff = (current_time - self.robot_last_move_time[robot_name]).nanoseconds / 1e9
                self.robot_static_time[robot_name] = time_diff
                
                # Force unlock only if static for too long
                if (self.robot_static_time[robot_name] > self.static_threshold and 
                    self.assigned_targets[robot_name] is not None and 
                    self.target_locked[robot_name]):
                    
                    self.get_logger().warn(
                        f'{robot_name} static for {self.robot_static_time[robot_name]:.1f}s '
                        f'(reason: {reason}), force unlock'
                    )
                    self._clear_robot_target(robot_name, current_time)
            
            # Update last position
            self.robot_last_pose[robot_name] = current_pose

    def _clear_robot_target(self, robot_name, current_time):
        """Clear robot target"""
        self.assigned_targets[robot_name] = None
        self.target_locked[robot_name] = False
        self.target_assignment_time[robot_name] = None
        self.robot_static_time[robot_name] = 0.0
        self.robot_last_move_time[robot_name] = current_time

    def is_point_too_close_to_other_target(self, point, robot_name):
        """Check if point is too close to other robot's target"""
        other_robot = 'robot2' if robot_name == 'robot1' else 'robot1'
        other_target = self.assigned_targets[other_robot]
        
        if other_target is None:
            return False
            
        distance = np.sqrt(
            (point[0] - other_target[0])**2 + 
            (point[1] - other_target[1])**2
        )
        
        return distance < self.min_target_distance

    def filter_excluded_points(self, points, robot_name):
        """Filter out points excluded by other robots"""
        other_robot = 'robot2' if robot_name == 'robot1' else 'robot1'
        other_target = self.assigned_targets[other_robot]
        
        if other_target is None:
            return points
        
        filtered_points = []
        for point in points:
            distance_to_other_target = np.sqrt(
                (point[0] - other_target[0])**2 + 
                (point[1] - other_target[1])**2
            )
            
            if distance_to_other_target >= self.exclusion_radius:
                filtered_points.append(point)
        
        return filtered_points

    def assign_targets(self):
        """Smart target assignment"""
        if not self.available_points:
            self.get_logger().debug('No available frontier points')
            return
            
        if self.robot1_pose is None or self.robot2_pose is None:
            self.get_logger().debug('Robot position information incomplete')
            return
            
        if self.processed_map is None:
            self.get_logger().debug('Map data not processed')
            return

        # Check which robots need new targets
        need_assignment = []
        current_time = self.get_clock().now()
        
        for robot_name in ['robot1', 'robot2']:
            # Absolute lock logic: if target is locked, never reassign
            if self.target_locked[robot_name]:
                target = self.assigned_targets[robot_name]
                time_locked = (current_time - self.target_assignment_time[robot_name]).nanoseconds / 1e9
                self.get_logger().debug(
                    f'{robot_name} target locked {target} (locked for {time_locked:.1f}s), never reassign'
                )
                continue
            
            if self.assigned_targets[robot_name] is None and not self.target_locked[robot_name]:
                need_assignment.append(robot_name)
                self.get_logger().info(f'{robot_name} has no locked target, can assign')

        if not need_assignment:
            return

        self.get_logger().info(f'Need assignment: {need_assignment}, available frontier: {len(self.available_points)}')

        # Process each robot needing assignment
        for robot_name in need_assignment:
            filtered_points = self.filter_excluded_points(self.available_points, robot_name)
            
            if not filtered_points:
                self.get_logger().warning(f'{robot_name} has no available frontier points (all excluded by other robots)')
                continue

            # Compose state dictionary - Fixed: add robot_target
            state = {
                "map": self.processed_map.tolist(),
                "frontiers": filtered_points,
                "robot1_pose": [self.robot1_pose.position.x, self.robot1_pose.position.y],
                "robot2_pose": [self.robot2_pose.position.x, self.robot2_pose.position.y],
                # Add current robot targets
                "robot1_target": self.assigned_targets['robot1'] if self.assigned_targets['robot1'] else [0.0, 0.0],
                "robot2_target": self.assigned_targets['robot2'] if self.assigned_targets['robot2'] else [0.0, 0.0],
                "request_robot": robot_name
            }

            try:
                self.get_logger().info(f'Requesting target assignment for {robot_name} from RL server...')
                target_result = send_state_and_get_target(state)
                
                if "error" in target_result:
                    self.get_logger().error(f'RL server error: {target_result["error"]}')
                    continue
                
                target_point = target_result.get('target_point')
                if target_point is None:
                    self.get_logger().warning(f'RL server did not return target point for {robot_name}')
                    continue
                
                if self.is_point_too_close_to_other_target(target_point, robot_name):
                    self.get_logger().warning(f'{robot_name} target point too close to other robot target, finding alternative')
                    alternative_target = self.find_alternative_target(filtered_points, robot_name)
                    if alternative_target:
                        target_point = alternative_target
                        self.get_logger().info(f'Found alternative target for {robot_name}: {alternative_target}')
                    else:
                        self.get_logger().warning(f'Cannot find suitable alternative target for {robot_name}')
                        continue
                
                # Assign target and immediately enable absolute lock
                self.publish_target_to_robot(robot_name, target_point)
                
            except Exception as e:
                self.get_logger().error(f'Error assigning target for {robot_name}: {e}')

    def find_alternative_target(self, available_points, robot_name):
        """Find alternative target point for robot"""
        robot_pose = getattr(self, f'{robot_name}_pose')
        robot_pos = [robot_pose.position.x, robot_pose.position.y]
        
        distances = []
        for point in available_points:
            if not self.is_point_too_close_to_other_target(point, robot_name):
                dist = np.sqrt(
                    (robot_pos[0] - point[0])**2 + 
                    (robot_pos[1] - point[1])**2
                )
                distances.append((point, dist))
        
        if not distances:
            return None
        
        distances.sort(key=lambda x: x[1])
        return distances[0][0]

    def publish_target_to_robot(self, robot_name, target):
        """Publish target point to robot and immediately enable absolute lock"""
        # Immediately lock target
        self.assigned_targets[robot_name] = target
        self.target_locked[robot_name] = True
        self.target_assignment_time[robot_name] = self.get_clock().now()
        
        # Reset motion state
        self.robot_static_time[robot_name] = 0.0
        self.robot_last_move_time[robot_name] = self.get_clock().now()
        
        # Create target message
        target_pose = PoseStamped()
        target_pose.header.frame_id = 'merge_map'
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.pose.position.x = target[0]
        target_pose.pose.position.y = target[1]
        target_pose.pose.orientation.w = 1.0

        # Publish to corresponding topic
        if robot_name == 'robot1':
            self.robot1_target_pub.publish(target_pose)
        else:
            self.robot2_target_pub.publish(target_pose)

        # Publish debug message
        debug_msg = String()
        debug_msg.data = f'Target locked: {robot_name} -> {target} (locked, never switch)'
        self.debug_pub.publish(debug_msg)
        self.get_logger().info(debug_msg.data)

    def publish_debug_info(self):
        """Publish detailed debug information"""
        debug_msg = String()
        debug_info = {
            "robot1_pose": "OK" if self.robot1_pose else "MISSING",
            "robot2_pose": "OK" if self.robot2_pose else "MISSING", 
            "map_data": "OK" if self.map_data is not None else "MISSING",
            "processed_map": "OK" if self.processed_map is not None else "MISSING",
            "available_points": len(self.available_points),
            "robot1_target": self.assigned_targets['robot1'],
            "robot2_target": self.assigned_targets['robot2'],
            "robot1_locked": self.target_locked['robot1'],
            "robot2_locked": self.target_locked['robot2'],
            "robot1_static_time": f"{self.robot_static_time['robot1']:.1f}s",
            "robot2_static_time": f"{self.robot_static_time['robot2']:.1f}s",
            "robot1_cmd_vel": f"{self.robot_cmd_vel['robot1']:.3f}" if self.robot_cmd_vel['robot1'] else "None",
            "robot2_cmd_vel": f"{self.robot_cmd_vel['robot2']:.3f}" if self.robot_cmd_vel['robot2'] else "None"
        }
        debug_msg.data = f"Test Socket Assigner status: {json.dumps(debug_info, ensure_ascii=False)}"
        self.debug_pub.publish(debug_msg)

    def create_target_marker(self, point, robot_name, marker_id):
        marker = Marker()
        marker.header.frame_id = "merge_map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"{robot_name}_target"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.6
        
        is_locked = self.target_locked[robot_name]
        if robot_name == 'robot1':
            if is_locked:
                marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Bright red: locked
            else:
                marker.color = ColorRGBA(r=0.8, g=0.4, b=0.4, a=0.8)  # Dark red: unlocked
        else:
            if is_locked:
                marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Bright green: locked
            else:
                marker.color = ColorRGBA(r=0.4, g=0.8, b=0.4, a=0.8)  # Dark green: unlocked
                
        return marker

    def publish_visualization(self):
        marker_array = MarkerArray()
        for robot_name in ['robot1', 'robot2']:
            if self.assigned_targets[robot_name]:
                marker_array.markers.append(
                    self.create_target_marker(
                        self.assigned_targets[robot_name],
                        robot_name,
                        len(marker_array.markers)
                    )
                )
        if marker_array.markers:
            self.target_viz_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    try:
        node = TestSocketAssigner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Received interrupt signal, shutting down...")
    except Exception as e:
        print(f'Error: {str(e)}')
        import traceback
        traceback.print_exc()
    finally:
        if 'node' in locals():
            try:
                node.destroy_node()
            except:
                pass
        try:
            rclpy.shutdown()
        except:
            pass

if __name__ == '__main__':
    main()