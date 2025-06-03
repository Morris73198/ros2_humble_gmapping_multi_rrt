import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped, Twist, Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import qos_profile_sensor_data
import threading
import math, time
import numpy as np
import tf_transformations

class PID:
    def __init__(self,kp,ki,kd,target):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.integral = 0
        self.diff = 0
        self.preerror = 0
        
    def PIDoutput(self,position):
        error = position - self.target
        self.integral = self.integral + error
        self.diff = error - self.preerror
        self.output = self.kp * error + self.ki * self.integral + self.kd * self.diff
        self.preerror = error

class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')
        
        # 获取节点的namespace来确定机器人身份
        self.robot_namespace = self.get_namespace()
        if 'robot1' in self.robot_namespace:
            self.robot_name = 'robot1'
            self.other_robot = 'robot2'
        elif 'robot2' in self.robot_namespace:
            self.robot_name = 'robot2'
            self.other_robot = 'robot1'
        else:
            self.robot_name = 'robot1'  # 默认
            self.other_robot = 'robot2'
        
        # Publishers (保持原有的topic)
        self.publisher_visual_path = self.create_publisher(Path, 'visual_path', 10)
        self.publisher_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.publisher_markers = self.create_publisher(MarkerArray, 'segment_markers', 10)
        self.publisher_current_goal = self.create_publisher(Marker, 'current_goal_marker', 10)
        self.publisher_apf_forces = self.create_publisher(Marker, 'apf_forces', 10)
        
        # Subscribers (保持原有的topic)
        self.subscription_path = self.create_subscription(
            Float32MultiArray,
            'path',
            self.path_callback,
            10)
            
        self.subscription_odom = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
            
        self.subscription_scan = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile_sensor_data)
        
        # 新增：订阅另一台机器人的位置 (新增功能)
        self.other_robot_pose = None
        self.subscription_other_robot = self.create_subscription(
            PoseStamped,
            f'/{self.other_robot}_pose',
            self.other_robot_callback,
            10)
            
        # APF parameters (保持原有参数，添加新的机器人相关参数)
        self.Eta_att = 110  
        self.Eta_rep_ob = 3  
        self.Eta_rep_robot = 15  # 新增：机器人间斥力系数
        self.n = 3  
        self.obstacle_distance_threshold = 0.5
        self.robot_distance_threshold = 1.2  # 新增：机器人间安全距离
        self.front_angle_range = 2 * math.pi
        self.segment_distance = 0.5
        
        # Speed limits (保持原有)
        self.speedTH = 0.15
        self.thetaTH = 0.65
        
        # Initialize states (保持原有)
        self.position = [0, 0, 0]
        self.euler = [0, 0, 0]
        self.path = None
        self.current_goal = None
        self.current_goal_index = 0
        self.following_path = False
        
        # Path following thread control (保持原有)
        self.follow_thread = None
        self.path_follow_lock = threading.Lock()
        
        # Initialize LiDAR (保持原有)
        self.angle_min = 0
        self.angle_max = 0
        self.angle_increment = 0
        self.ranges = []
        
        self.front_min_angle = 2*math.pi - self.front_angle_range/2
        self.front_max_angle = self.front_angle_range/2
        
        self.get_logger().info(f"Path follower node has been started for {self.robot_name}")

    def other_robot_callback(self, msg):
        """新增：接收另一台机器人的位置信息"""
        self.other_robot_pose = msg.pose

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.publisher_cmd_vel.publish(twist)

    def create_current_goal_marker(self):
        """Create marker for current goal point"""
        marker = Marker()
        marker.header.frame_id = "merge_map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "current_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = self.current_goal[0]
        marker.pose.position.y = self.current_goal[1]
        marker.pose.position.z = 0.2
        
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.4
        marker.scale.y = 0.4
        marker.scale.z = 0.4
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        return marker

    def create_force_vector_marker(self, fx, fy):
        """Create marker for APF force vector"""
        marker = Marker()
        marker.header.frame_id = "merge_map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "force_vector"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        
        # Start point at robot position
        marker.points.append(Point(x=self.position[0], y=self.position[1], z=0.1))
        
        # End point shows force direction
        scale = 0.5  # Scale factor for vector visualization
        marker.points.append(Point(
            x=self.position[0] + fx * scale,
            y=self.position[1] + fy * scale,
            z=0.1
        ))
        
        # Arrow properties
        marker.scale.x = 0.1  # shaft diameter
        marker.scale.y = 0.2  # head diameter
        marker.scale.z = 0.2  # head length
        
        # Green color for force vector
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        return marker

    def segment_path(self, path):
        """Split path into fixed distance segments"""
        segment_points = []  # Always include start point
        
        for i in range(1, len(path)):
            if not segment_points:  # 如果 segment_points 是空的
                prev_point = path[0]
            else:
                prev_point = segment_points[-1]
            current_point = path[i]
            
            distance = math.hypot(
                current_point[0] - prev_point[0],
                current_point[1] - prev_point[1]
            )
            
            if distance >= self.segment_distance:
                num_segments = int(distance / self.segment_distance)
                for j in range(1, num_segments + 1):
                    ratio = j * self.segment_distance / distance
                    new_x = prev_point[0] + (current_point[0] - prev_point[0]) * ratio
                    new_y = prev_point[1] + (current_point[1] - prev_point[1]) * ratio
                    segment_points.append((new_x, new_y))
                
        segment_points.append(path[-1])  # Always include end point
        return segment_points

    def create_segment_markers(self, path):
        """Create hollow circle markers for segment points"""
        marker_array = MarkerArray()
        segment_points = self.segment_path(path)
        
        for i, point in enumerate(segment_points):
            marker = Marker()
            marker.header.frame_id = "merge_map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "segment_markers"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = point[0]
            marker.pose.position.y = point[1]
            marker.pose.position.z = 0.05
            
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.3
            marker.scale.y = 0.3
            marker.scale.z = 0.01
            
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 0.8
            
            marker_array.markers.append(marker)
            
        return marker_array

    def path_callback(self, msg):
        self.get_logger().info("New path received")
        
        with self.path_follow_lock:
            if self.following_path:
                self.following_path = False
                if self.follow_thread:
                    self.follow_thread.join()
                self.stop_robot()
            
            data_list = list(msg.data)
            self.path = [(data_list[i], data_list[i+1]) 
                        for i in range(0, len(data_list), 2)]
            
            # Create segment markers
            if len(self.path) > 0:
                marker_array = self.create_segment_markers(self.path)
                self.publisher_markers.publish(marker_array)
                
            self.path = self.segment_path(self.path)    
            self.current_goal_index = 0
            self.current_goal = self.path[0]
            
            self.following_path = True
            self.follow_thread = threading.Thread(target=self.follow_path)
            self.follow_thread.start()

    def odom_callback(self, msg):
        self.position[0] = msg.pose.pose.position.x
        self.position[1] = msg.pose.pose.position.y
        self.position[2] = msg.pose.pose.position.z
        
        self.euler = tf_transformations.euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])

    def scan_callback(self, msg):
        self.angle_min = msg.angle_min
        self.angle_max = msg.angle_max
        self.angle_increment = msg.angle_increment
        self.ranges = msg.ranges

    def calculate_robot_repulsive_force(self):
        """新增：计算来自另一台机器人的斥力"""
        if not self.other_robot_pose:
            return 0, 0
        
        # 计算与另一台机器人的距离
        other_pos = [self.other_robot_pose.position.x, self.other_robot_pose.position.y]
        dx = self.position[0] - other_pos[0]
        dy = self.position[1] - other_pos[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance < self.robot_distance_threshold and distance > 0:
            # 计算斥力大小
            force_magnitude = self.Eta_rep_robot * (1/distance - 1/self.robot_distance_threshold) / (distance**2)
            
            # 归一化方向向量
            unit_vector_x = dx / distance
            unit_vector_y = dy / distance
            
            # 返回斥力分量
            F_rep_x = force_magnitude * unit_vector_x
            F_rep_y = force_magnitude * unit_vector_y
            
            return F_rep_x, F_rep_y
        
        return 0, 0

    def calculate_apf_forces(self):
        if not self.current_goal:
            return 0, 0
            
        # Attractive force toward goal (保持原有逻辑)
        Ptogoal = [
            self.current_goal[0] - self.position[0],
            self.current_goal[1] - self.position[1]
        ]
        distance_to_goal = np.linalg.norm(Ptogoal)
        F_att = [
            self.Eta_att * distance_to_goal * Ptogoal[0]/distance_to_goal,
            self.Eta_att * distance_to_goal * Ptogoal[1]/distance_to_goal
        ]
        
        # Repulsive force from obstacles (保持原有逻辑)
        F_rep_obstacles = [0, 0]
        if len(self.ranges) > 0:
            angles = self.angle_min + np.arange(len(self.ranges)) * self.angle_increment
            front_indices = np.where(
                (angles > self.front_min_angle) | 
                (angles < self.front_max_angle)
            )[0]
            
            for idx in front_indices:
                if self.ranges[idx] < self.obstacle_distance_threshold:
                    obs_angle = angles[idx] + self.euler[2]
                    
                    F_rep_ob1_abs = (
                        self.Eta_rep_ob * 
                        (1/self.ranges[idx] - 1/self.obstacle_distance_threshold) * 
                        distance_to_goal**self.n / 
                        self.ranges[idx]**2
                    )
                    
                    F_rep_obstacles[0] += F_rep_ob1_abs * (-math.cos(obs_angle))
                    F_rep_obstacles[1] += F_rep_ob1_abs * (-math.sin(obs_angle))
        
        # 新增：计算来自另一台机器人的斥力
        F_rep_robot_x, F_rep_robot_y = self.calculate_robot_repulsive_force()
        
        # Sum all forces and normalize (修改：包含机器人斥力)
        F_sum = [
            F_att[0] + F_rep_obstacles[0] + F_rep_robot_x, 
            F_att[1] + F_rep_obstacles[1] + F_rep_robot_y
        ]
        F_magnitude = np.linalg.norm(F_sum)
        
        if F_magnitude > 0:
            UniVec_Fsum = [f / F_magnitude for f in F_sum]
            return UniVec_Fsum[0], UniVec_Fsum[1]
    def follow_path(self):
        """Main path following logic"""
        twist = Twist()
        path_msg = Path()
        path_msg.header.frame_id = "merge_map"
        
        # Initialize PID controllers
        z = PID(0.4, 0.0, 0.0, 0)  # Angular PID
        x = PID(1, 0.01, 0.05, 0)   # Linear PID
        
        # 根據路徑段動態調整閾值
        def get_distance_threshold(current_index):
            if current_index >= len(self.path) - 1:  # 最後一個點
                return 0.5  # 終點使用更寬鬆的閾值
            else:
                return 0.3  # 中間點保持原閾值
        
        while self.position[0] == 0 and self.position[1] == 0:
            time.sleep(0.1)
            
        # Create visualization path message
        for px, py in self.path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = px
            pose.pose.position.y = py
            path_msg.poses.append(pose)
            
        while self.following_path and self.current_goal_index < len(self.path):
            path_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Calculate and visualize APF forces
            fx, fy = self.calculate_apf_forces()
            force_marker = self.create_force_vector_marker(fx, fy)
            self.publisher_apf_forces.publish(force_marker)
            
            # Update and publish current goal marker
            current_goal_marker = self.create_current_goal_marker()
            self.publisher_current_goal.publish(current_goal_marker)
            
            theta = math.atan2(fy, fx)
            rotate = self.euler[2] - theta

            distance_to_goal = math.hypot(
                self.position[0] - self.current_goal[0],
                self.position[1] - self.current_goal[1]
            )
            
            # 使用動態閾值
            current_threshold = get_distance_threshold(self.current_goal_index)
            
            min_obstacle_distance = self.obstacle_distance_threshold
            if len(self.ranges) > 0:
                min_obstacle_distance = min(self.obstacle_distance_threshold, min(self.ranges))
            
            # Calculate linear velocity based on angular error
            twist.linear.x = abs(self.speedTH * (1 - abs(rotate)/math.pi))
            
            # Calculate angular velocity using PID
            if rotate > 0:
                if rotate <= math.pi:
                    z.PIDoutput(rotate)
                    twist.angular.z = max(-self.thetaTH, -z.output)
                else:
                    z.PIDoutput(2*math.pi - rotate)
                    twist.angular.z = min(self.thetaTH, z.output)
            else:
                if abs(rotate) <= math.pi:
                    z.PIDoutput(rotate)
                    twist.angular.z = min(self.thetaTH, -z.output)
                else:
                    z.PIDoutput(-2*math.pi - rotate)
                    twist.angular.z = max(-self.thetaTH, z.output)
            
            # 檢查是否到達當前目標點，使用動態閾值
            if distance_to_goal < current_threshold:
                self.current_goal_index += 1
                if self.current_goal_index < len(self.path):
                    self.current_goal = self.path[self.current_goal_index]
                    self.get_logger().info(f"Reached waypoint {self.current_goal_index}")
                    x.integral = 0
                    z.integral = 0
                else:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.publisher_cmd_vel.publish(twist)
                    self.get_logger().info("Goal reached")
                    break
            
            # Publish commands if still following path
            if self.following_path:
                self.publisher_visual_path.publish(path_msg)
                self.publisher_cmd_vel.publish(twist)
            else:
                self.stop_robot()
                break
            
            time.sleep(0.1)
        
        self.following_path = False
        self.stop_robot()

def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()