#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Twist
from visualization_msgs.msg import MarkerArray, Marker
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String, ColorRGBA, Float32MultiArray
import numpy as np
import heapq
from typing import List, Tuple, Set
import cv2
import os
import tensorflow as tf

# 導入DRL模型相關模塊
try:
    from rrt_exploration_ros2.multi_robot_network import MultiRobotNetworkModel
except ImportError:
    try:
        # 嘗試其他可能的導入路徑
        from two_robot_dueling_dqn_attention.models.multi_robot_network import MultiRobotNetworkModel
    except ImportError:
        MultiRobotNetworkModel = None
        print("警告: 無法導入 MultiRobotNetworkModel，將使用距離基礎分配")

class DRLAssigner(Node):
    def __init__(self):
        super().__init__('drl_assigner')
        
        # 首先設置所有基本參數（在載入模型之前）
        # DRL模型相關參數
        self.map_size_for_model = (84, 84)  # 模型輸入的地圖尺寸
        self.max_frontiers = 50  # 最大frontier點數量
        
        # 聲明模型路徑參數 - 設置默認路徑到您的模型目錄
        default_model_path = '/home/airlab/rrt_ws/src/ros2_humble_gmapping_multi_rrt/rrt_exploration_ros2/rrt_exploration_ros2/saved_models/robot_rl_model.h5'
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('use_drl', True)  # 默認啟用DRL
        
        self.model_path = self.get_parameter('model_path').value
        self.use_drl = self.get_parameter('use_drl').value
        
        # 初始化變量
        self.robot1_pose = None
        self.robot2_pose = None
        self.available_points = []
        self.assigned_targets = {'robot1': None, 'robot2': None}
        self.robot_status = {'robot1': True, 'robot2': True}  # True 表示機器人可以接收新目標
        
        # 機器人速度相關變量
        self.robot_velocities = {'robot1': None, 'robot2': None}
        self.velocity_check_threshold = 0.01  # 速度閾值
        self.static_duration = {'robot1': 0.0, 'robot2': 0.0}  # 記錄機器人靜止的持續時間
        self.static_threshold = 2.0  # 靜止超過此時間（秒）就重新分配目標
        
        # 地圖相關變量
        self.map_data = None
        self.map_resolution = None
        self.map_width = None
        self.map_height = None
        self.map_origin = None
        self.processed_map = None  # 處理後的地圖，用於DRL模型
        
        # 目標到達閾值
        self.target_threshold = 0.3  # 機器人距離目標點小於此值視為到達
        
        # 初始化DRL模型（在所有參數設置完成後）
        self.drl_model = None
        if self.use_drl:
            self.load_drl_model()
        
        # 設置訂閱者和發布者
        self.setup_subscribers()
        self.setup_publishers()
        
        # 創建定時器
        self.create_timer(1.0, self.assign_targets)
        self.create_timer(0.1, self.publish_visualization)
        self.create_timer(0.1, self.check_target_reached)
        self.create_timer(0.1, self.check_robot_motion)
        
        mode = "DRL增強" if self.use_drl and self.drl_model else "距離基礎"
        self.get_logger().info(f'{mode}的分配節點已啟動')
        if self.use_drl:
            self.get_logger().info(f'模型路徑: {self.model_path}')

    def load_drl_model(self):
        """載入預訓練的DRL模型"""
        try:
            # 檢查模型類是否可用
            if MultiRobotNetworkModel is None:
                self.get_logger().error('MultiRobotNetworkModel 類不可用，請檢查導入路徑')
                self.use_drl = False
                return
                
            # 檢查模型文件是否存在
            if not os.path.exists(self.model_path):
                # 嘗試尋找模型目錄中的其他模型文件
                model_dir = '/home/airlab/rrt_ws/src/ros2_humble_gmapping_multi_rrt/rrt_exploration_ros2/rrt_exploration_ros2/saved_models/'
                self.get_logger().info(f'模型文件不存在: {self.model_path}')
                self.get_logger().info(f'正在搜索模型目錄: {model_dir}')
                
                if os.path.exists(model_dir):
                    # 列出目錄中的所有 .h5 文件
                    h5_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
                    if h5_files:
                        self.get_logger().info(f'找到的模型文件: {h5_files}')
                        # 使用第一個找到的 .h5 文件
                        self.model_path = os.path.join(model_dir, h5_files[0])
                        self.get_logger().info(f'使用模型文件: {self.model_path}')
                    else:
                        self.get_logger().warn(f'在 {model_dir} 中未找到 .h5 模型文件')
                        self.get_logger().info('將使用距離基礎分配方法')
                        self.use_drl = False
                        return
                else:
                    self.get_logger().error(f'模型目錄不存在: {model_dir}')
                    self.use_drl = False
                    return
            
            # 檢查 TensorFlow 版本是否過新，dueling.h5 可能是舊格式
            if 'dueling.h5' in self.model_path:
                self.get_logger().warn('檢測到 dueling.h5 文件，這可能不是正確的 MultiRobotNetworkModel 格式')
                self.get_logger().warn('如果載入失敗，將自動切換到距離基礎分配方法')
            
            # 初始化模型
            self.drl_model = MultiRobotNetworkModel(
                input_shape=(84, 84, 1),
                max_frontiers=self.max_frontiers
            )
            
            # 載入預訓練權重
            self.drl_model.load(self.model_path)
            
            self.get_logger().info(f'成功載入DRL模型: {self.model_path}')
            
        except Exception as e:
            self.get_logger().error(f'載入DRL模型失敗: {str(e)}')
            self.get_logger().info('將使用距離基礎分配方法')
            self.drl_model = None
            self.use_drl = False

    def setup_subscribers(self):
        """設置所有訂閱者"""
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/merge_map',
            self.map_callback,
            10
        )
        
        self.robot1_pose_sub = self.create_subscription(
            PoseStamped,
            '/robot1_pose',
            self.robot1_pose_callback,
            10
        )
        
        self.robot2_pose_sub = self.create_subscription(
            PoseStamped,
            '/robot2_pose',
            self.robot2_pose_callback,
            10
        )
        
        self.filtered_points_sub = self.create_subscription(
            MarkerArray,
            '/filtered_points',
            self.filtered_points_callback,
            10
        )
        
        # 訂閱各機器人的速度命令
        self.robot1_cmd_vel_sub = self.create_subscription(
            Twist,
            '/robot1/cmd_vel',
            lambda msg: self.cmd_vel_callback(msg, 'robot1'),
            10
        )
        
        self.robot2_cmd_vel_sub = self.create_subscription(
            Twist,
            '/robot2/cmd_vel',
            lambda msg: self.cmd_vel_callback(msg, 'robot2'),
            10
        )

    def setup_publishers(self):
        """設置所有發布者"""
        # 目標點發布
        self.robot1_target_pub = self.create_publisher(
            PoseStamped,
            '/robot1/goal_pose',
            10
        )
        
        self.robot2_target_pub = self.create_publisher(
            PoseStamped,
            '/robot2/goal_pose',
            10
        )

        # 可視化發布
        self.target_viz_pub = self.create_publisher(
            MarkerArray,
            '/assigned_targets_viz',
            10
        )

        # 調試信息發布
        self.debug_pub = self.create_publisher(
            String,
            '/assigner/debug',
            10
        )

    def process_map_for_model(self, occupancy_grid):
        """處理地圖數據為DRL模型輸入格式"""
        if occupancy_grid is None:
            return None
            
        try:
            # 將佔用格網數據轉換為二進制圖像
            map_array = np.array(occupancy_grid)
            map_binary = np.zeros_like(map_array, dtype=np.uint8)
            
            # 地圖值映射：0=已知自由空間, 100=障礙物, -1=未知空間
            map_binary[map_array == 0] = 255    # 已知自由空間 -> 白色
            map_binary[map_array == 100] = 0    # 障礙物 -> 黑色  
            map_binary[map_array == -1] = 127   # 未知空間 -> 灰色
            
            # 調整地圖尺寸到模型所需尺寸
            resized_map = cv2.resize(
                map_binary, 
                self.map_size_for_model, 
                interpolation=cv2.INTER_LINEAR
            )
            
            # 正規化到[0,1]範圍並添加通道維度
            normalized_map = resized_map.astype(np.float32) / 255.0
            processed_map = np.expand_dims(normalized_map, axis=-1)
            
            return processed_map
            
        except Exception as e:
            self.get_logger().error(f'地圖處理錯誤: {str(e)}')
            return None

    def pad_frontiers(self, frontiers):
        """填充frontier點到固定長度並進行標準化"""
        padded = np.zeros((self.max_frontiers, 2))
        
        if len(frontiers) > 0:
            frontiers = np.array(frontiers)
            
            # 標準化座標到[0,1]範圍
            if self.map_width and self.map_height and self.map_resolution:
                map_width_m = self.map_width * self.map_resolution
                map_height_m = self.map_height * self.map_resolution
                
                normalized_frontiers = frontiers.copy()
                normalized_frontiers[:, 0] = (frontiers[:, 0] - self.map_origin.position.x) / map_width_m
                normalized_frontiers[:, 1] = (frontiers[:, 1] - self.map_origin.position.y) / map_height_m
                
                # 確保座標在[0,1]範圍內
                normalized_frontiers = np.clip(normalized_frontiers, 0.0, 1.0)
            else:
                normalized_frontiers = frontiers
            
            n_frontiers = min(len(frontiers), self.max_frontiers)
            padded[:n_frontiers] = normalized_frontiers[:n_frontiers]
        
        return padded

    def get_normalized_position(self, pose):
        """獲取正規化後的機器人位置"""
        if not pose or not self.map_width or not self.map_height or not self.map_resolution:
            return np.array([0.0, 0.0])
            
        map_width_m = self.map_width * self.map_resolution
        map_height_m = self.map_height * self.map_resolution
        
        normalized_x = (pose.position.x - self.map_origin.position.x) / map_width_m
        normalized_y = (pose.position.y - self.map_origin.position.y) / map_height_m
        
        # 確保座標在[0,1]範圍內
        normalized_x = np.clip(normalized_x, 0.0, 1.0)
        normalized_y = np.clip(normalized_y, 0.0, 1.0)
        
        return np.array([normalized_x, normalized_y])

    def get_normalized_target(self, target):
        """標準化目標位置"""
        if target is None or not self.map_width or not self.map_height or not self.map_resolution:
            return np.array([0.0, 0.0])
            
        map_width_m = self.map_width * self.map_resolution
        map_height_m = self.map_height * self.map_resolution
        
        normalized_x = (target[0] - self.map_origin.position.x) / map_width_m
        normalized_y = (target[1] - self.map_origin.position.y) / map_height_m
        
        # 確保座標在[0,1]範圍內
        normalized_x = np.clip(normalized_x, 0.0, 1.0)
        normalized_y = np.clip(normalized_y, 0.0, 1.0)
        
        return np.array([normalized_x, normalized_y])

    def map_callback(self, msg):
        """處理地圖數據"""
        self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_resolution = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin = msg.info.origin
        
        # 處理地圖供DRL模型使用
        if self.use_drl:
            self.processed_map = self.process_map_for_model(self.map_data)
        
        self.get_logger().debug('收到地圖更新並處理完成')

    def robot1_pose_callback(self, msg):
        """處理機器人1的位置更新"""
        self.robot1_pose = msg.pose
        self.get_logger().debug('收到機器人1位置更新')

    def robot2_pose_callback(self, msg):
        """處理機器人2的位置更新"""
        self.robot2_pose = msg.pose
        self.get_logger().debug('收到機器人2位置更新')

    def filtered_points_callback(self, msg):
        """處理過濾後的點"""
        self.available_points = []
        if msg.markers:
            for marker in msg.markers:
                self.available_points.extend([(p.x, p.y) for p in marker.points])
        self.get_logger().debug(f'收到 {len(self.available_points)} 個過濾後的點')

    def cmd_vel_callback(self, msg: Twist, robot_name: str):
        """處理速度命令消息，更新機器人速度狀態"""
        total_velocity = abs(msg.linear.x) + abs(msg.linear.y) + abs(msg.angular.z)
        self.robot_velocities[robot_name] = total_velocity

    def check_robot_motion(self):
        """檢查機器人是否靜止"""
        for robot_name in ['robot1', 'robot2']:
            if self.robot_velocities[robot_name] is None:
                continue
                
            if self.robot_velocities[robot_name] < self.velocity_check_threshold:
                self.static_duration[robot_name] += 0.1
                
                if (self.static_duration[robot_name] >= self.static_threshold and 
                    not self.robot_status[robot_name]):
                    self.get_logger().info(f'{robot_name} 已靜止 {self.static_threshold} 秒，標記為可用狀態')
                    self.robot_status[robot_name] = True
                    self.assigned_targets[robot_name] = None
            else:
                self.static_duration[robot_name] = 0.0

    def check_target_reached(self):
        """檢查機器人是否到達目標點"""
        robots = {
            'robot1': self.robot1_pose,
            'robot2': self.robot2_pose
        }

        for robot_name, robot_pose in robots.items():
            if not robot_pose or not self.assigned_targets[robot_name]:
                continue

            target = self.assigned_targets[robot_name]
            current_pos = (robot_pose.position.x, robot_pose.position.y)
            target_pos = target

            distance = np.sqrt(
                (current_pos[0] - target_pos[0])**2 + 
                (current_pos[1] - target_pos[1])**2
            )

            if distance < self.target_threshold:
                if not self.robot_status[robot_name]:
                    self.get_logger().info(f'{robot_name} 已到達目標點 {target_pos}')
                self.robot_status[robot_name] = True
                self.assigned_targets[robot_name] = None
            else:
                self.robot_status[robot_name] = False

    def calculate_distance(self, point1, point2):
        """計算兩點之間的歐幾里德距離"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def calculate_utility_score(self, robot_pose, target_point, other_robot_pose=None, other_target=None):
        """計算目標點的效用分數（距離基礎的智能分配）"""
        if not robot_pose:
            return float('inf')
        
        robot_pos = (robot_pose.position.x, robot_pose.position.y)
        
        # 基礎距離分數（距離越近分數越高）
        distance = self.calculate_distance(robot_pos, target_point)
        distance_score = 1.0 / (1.0 + distance)
        
        # 避免衝突分數
        conflict_penalty = 0.0
        if other_robot_pose and other_target:
            other_pos = (other_robot_pose.position.x, other_robot_pose.position.y)
            
            # 如果另一個機器人更接近這個目標，給予懲罰
            other_distance = self.calculate_distance(other_pos, target_point)
            if other_distance < distance:
                conflict_penalty = 0.5
            
            # 如果目標點太接近另一個機器人的目標，給予懲罰
            if other_target:
                target_distance = self.calculate_distance(target_point, other_target)
                if target_distance < 2.0:  # 2米內視為太接近
                    conflict_penalty += 0.3
        
        # 綜合分數
        final_score = distance_score - conflict_penalty
        return final_score

    def distance_based_assignment(self, robot_name, available_points, assigned_points):
        """基於距離和效用的智能分配方法"""
        robot_pose = self.robot1_pose if robot_name == 'robot1' else self.robot2_pose
        other_robot_pose = self.robot2_pose if robot_name == 'robot1' else self.robot1_pose
        other_robot_name = 'robot2' if robot_name == 'robot1' else 'robot1'
        other_target = self.assigned_targets.get(other_robot_name)
        
        if not robot_pose:
            return None
            
        valid_targets = []
        MIN_DISTANCE = 0.5  # 最小距離要求
        
        for point in available_points:
            if tuple(point) in assigned_points:
                continue
                
            distance = self.calculate_distance(
                (robot_pose.position.x, robot_pose.position.y), 
                point
            )
            
            if distance >= MIN_DISTANCE:
                # 計算效用分數
                utility_score = self.calculate_utility_score(
                    robot_pose, point, other_robot_pose, other_target
                )
                valid_targets.append((point, utility_score, distance))
        
        if valid_targets:
            # 選擇效用分數最高的點
            best_target = max(valid_targets, key=lambda x: x[1])
            self.get_logger().debug(
                f'{robot_name} 選擇目標: {best_target[0]}, '
                f'效用分數: {best_target[1]:.3f}, 距離: {best_target[2]:.2f}m'
            )
            return best_target[0]
            
        return None

    def predict_best_frontier_with_drl(self, robot_name):
        """使用DRL模型預測最佳frontier點"""
        if (not self.use_drl or 
            self.drl_model is None or 
            self.processed_map is None or 
            len(self.available_points) == 0):
            return None
            
        try:
            # 準備輸入數據
            state = np.expand_dims(self.processed_map, 0)  # 添加batch維度
            frontiers = np.expand_dims(self.pad_frontiers(self.available_points), 0)
            
            # 獲取機器人位置
            robot1_pos = self.get_normalized_position(self.robot1_pose)
            robot2_pos = self.get_normalized_position(self.robot2_pose)
            robot1_pos_batch = np.expand_dims(robot1_pos, 0)
            robot2_pos_batch = np.expand_dims(robot2_pos, 0)
            
            # 獲取當前目標位置
            robot1_target = self.get_normalized_target(self.assigned_targets.get('robot1'))
            robot2_target = self.get_normalized_target(self.assigned_targets.get('robot2'))
            robot1_target_batch = np.expand_dims(robot1_target, 0)
            robot2_target_batch = np.expand_dims(robot2_target, 0)
            
            # 使用DRL模型進行預測
            predictions = self.drl_model.predict(
                state, frontiers,
                robot1_pos_batch, robot2_pos_batch,
                robot1_target_batch, robot2_target_batch
            )
            
            # 提取對應機器人的Q值
            valid_frontiers = min(self.max_frontiers, len(self.available_points))
            if robot_name == 'robot1':
                q_values = predictions['robot1'][0, :valid_frontiers]
            else:
                q_values = predictions['robot2'][0, :valid_frontiers]
            
            # 選擇Q值最高的動作
            best_action = np.argmax(q_values)
            best_frontier = self.available_points[best_action]
            
            self.get_logger().debug(f'DRL模型為 {robot_name} 選擇的frontier: {best_frontier}')
            return best_frontier
            
        except Exception as e:
            self.get_logger().error(f'DRL預測錯誤: {str(e)}')
            return None

    def create_target_marker(self, point: Tuple[float, float], robot_name: str, marker_id: int) -> Marker:
        """創建目標點標記"""
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
        
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        
        if robot_name == 'robot1':
            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
        else:
            marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.8)
            
        return marker

    def publish_visualization(self):
        """發布目標點的可視化"""
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
        
        self.target_viz_pub.publish(marker_array)

    def assign_targets(self):
        """分配目標給機器人 - 主要邏輯"""
        if (not self.available_points or 
            self.robot1_pose is None or 
            self.robot2_pose is None):
            return

        # 記錄已分配的點
        assigned_points = set()
        for robot, target in self.assigned_targets.items():
            if target is not None:
                assigned_points.add(tuple(target))

        for robot_name in ['robot1', 'robot2']:
            # 只在機器人可用且沒有當前目標時分配新目標
            if not self.robot_status[robot_name] or self.assigned_targets[robot_name] is not None:
                continue

            # 過濾掉已分配的點
            available_points = [
                point for point in self.available_points 
                if tuple(point) not in assigned_points
            ]
            
            if not available_points:
                continue

            # 首先嘗試使用DRL模型（如果啟用且可用）
            best_point = None
            method_used = "距離基礎"
            
            if self.use_drl and self.drl_model:
                best_point = self.predict_best_frontier_with_drl(robot_name)
                if best_point:
                    method_used = "DRL模型"
            
            # 如果DRL模型失敗或未啟用，使用智能距離基礎方法
            if best_point is None:
                best_point = self.distance_based_assignment(robot_name, available_points, assigned_points)
            
            if best_point is not None:
                # 確保選擇的點還沒有被分配
                if tuple(best_point) not in assigned_points:
                    assigned_points.add(tuple(best_point))
                    self.assigned_targets[robot_name] = best_point

                    # 創建並發布目標點消息
                    target_pose = PoseStamped()
                    target_pose.header.frame_id = 'merge_map'
                    target_pose.header.stamp = self.get_clock().now().to_msg()
                    target_pose.pose.position.x = best_point[0]
                    target_pose.pose.position.y = best_point[1]
                    target_pose.pose.orientation.w = 1.0

                    # 根據機器人選擇對應的發布者
                    if robot_name == 'robot1':
                        self.robot1_target_pub.publish(target_pose)
                    else:
                        self.robot2_target_pub.publish(target_pose)

                    # 發布調試信息
                    debug_msg = String()
                    debug_msg.data = f'已將目標點 {best_point} 分配給 {robot_name} (使用{method_used})'
                    self.debug_pub.publish(debug_msg)
                    self.get_logger().info(debug_msg.data)
            else:
                self.get_logger().warn(f'未找到 {robot_name} 的有效目標')

def main(args=None):
    """主函數"""
    rclpy.init(args=args)
    
    # 配置TensorFlow GPU設置（如果有GPU）
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f'找到 {len(gpus)} 個 GPU 設備，已啟用記憶體增長')
            except RuntimeError as e:
                print(f'GPU配置錯誤: {e}')
        else:
            print('未找到 GPU 設備，使用 CPU 運行')
    except Exception as e:
        print(f'TensorFlow 配置錯誤: {e}')
    
    try:
        node = DRLAssigner()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("收到中斷信號，正在關閉...")
    except Exception as e:
        print(f'錯誤: {str(e)}')
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