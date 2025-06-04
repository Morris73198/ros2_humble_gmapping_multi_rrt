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
import sys

# 安全導入 TensorFlow
try:
    import tensorflow as tf
    print(f"TensorFlow {tf.__version__} 已載入")
    
    # 處理不同版本的 TensorFlow API
    if hasattr(tf, 'config'):
        # TensorFlow 2.x
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f'找到 {len(gpus)} 個 GPU 設備')
            else:
                print('未找到 GPU 設備，使用 CPU 運行')
        except Exception as e:
            print(f'GPU 配置警告: {e}')
    else:
        # 舊版本 TensorFlow
        print('使用舊版本 TensorFlow API')
        
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow 未安裝，將僅使用距離基礎分配")
    TENSORFLOW_AVAILABLE = False
    tf = None

class SimpleNetworkModel:
    """簡化的神經網絡模型類，兼容不同的 TensorFlow 版本"""
    
    def __init__(self, input_shape=(84, 84, 1), max_frontiers=50):
        self.input_shape = input_shape
        self.max_frontiers = max_frontiers
        self.model = None
        self.is_loaded = False
        
        if TENSORFLOW_AVAILABLE:
            try:
                self._build_simple_model()
            except Exception as e:
                print(f"模型建構失敗: {e}")
                self.model = None
    
    def _build_simple_model(self):
        """建構簡化的模型架構"""
        if not TENSORFLOW_AVAILABLE:
            return
            
        try:
            # 地圖輸入
            map_input = tf.keras.layers.Input(shape=self.input_shape, name='map_input')
            
            # 機器人位置輸入
            robot1_pos = tf.keras.layers.Input(shape=(2,), name='robot1_pos_input')
            robot2_pos = tf.keras.layers.Input(shape=(2,), name='robot2_pos_input')
            
            # Frontier 輸入
            frontier_input = tf.keras.layers.Input(
                shape=(self.max_frontiers, 2), 
                name='frontier_input'
            )
            
            # 簡化的 CNN 處理地圖
            x = tf.keras.layers.Conv2D(32, 3, activation='relu')(map_input)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
            x = tf.keras.layers.MaxPooling2D(2)(x)
            x = tf.keras.layers.Flatten()(x)
            map_features = tf.keras.layers.Dense(128, activation='relu')(x)
            
            # 處理機器人位置
            robot1_features = tf.keras.layers.Dense(32, activation='relu')(robot1_pos)
            robot2_features = tf.keras.layers.Dense(32, activation='relu')(robot2_pos)
            
            # 處理 frontier
            frontier_flat = tf.keras.layers.Flatten()(frontier_input)
            frontier_features = tf.keras.layers.Dense(64, activation='relu')(frontier_flat)
            
            # 組合特徵
            combined_features = tf.keras.layers.Concatenate()([
                map_features, robot1_features, robot2_features, frontier_features
            ])
            
            # 共享特徵層
            shared = tf.keras.layers.Dense(256, activation='relu')(combined_features)
            shared = tf.keras.layers.Dropout(0.3)(shared)
            shared = tf.keras.layers.Dense(128, activation='relu')(shared)
            
            # 為每個機器人創建輸出
            robot1_output = tf.keras.layers.Dense(
                self.max_frontiers, 
                activation='linear',
                name='robot1'
            )(shared)
            
            robot2_output = tf.keras.layers.Dense(
                self.max_frontiers, 
                activation='linear',
                name='robot2'
            )(shared)
            
            # 建構模型
            self.model = tf.keras.Model(
                inputs={
                    'map_input': map_input,
                    'frontier_input': frontier_input,
                    'robot1_pos_input': robot1_pos,
                    'robot2_pos_input': robot2_pos
                },
                outputs={
                    'robot1': robot1_output,
                    'robot2': robot2_output
                }
            )
            
            # 編譯模型
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            self.model.compile(
                optimizer=optimizer,
                loss={'robot1': 'mse', 'robot2': 'mse'}
            )
            
            print("簡化模型建構成功")
            
        except Exception as e:
            print(f"模型建構錯誤: {e}")
            self.model = None
    
    def load(self, filepath):
        """載入模型權重"""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            return False
            
        try:
            if os.path.exists(filepath):
                # 嘗試載入完整模型
                try:
                    self.model = tf.keras.models.load_model(filepath)
                    self.is_loaded = True
                    print(f"成功載入完整模型: {filepath}")
                    return True
                except Exception as e:
                    print(f"載入完整模型失敗: {e}")
                    
                # 嘗試只載入權重
                try:
                    self.model.load_weights(filepath)
                    self.is_loaded = True
                    print(f"成功載入模型權重: {filepath}")
                    return True
                except Exception as e:
                    print(f"載入權重失敗: {e}")
                    
            return False
            
        except Exception as e:
            print(f"載入模型時發生錯誤: {e}")
            return False
    
    def predict(self, map_input, frontier_input, robot1_pos, robot2_pos):
        """進行預測"""
        if not TENSORFLOW_AVAILABLE or self.model is None or not self.is_loaded:
            return None
            
        try:
            # 確保輸入形狀正確
            if len(map_input.shape) == 3:
                map_input = np.expand_dims(map_input, 0)
            if len(frontier_input.shape) == 2:
                frontier_input = np.expand_dims(frontier_input, 0)
            if len(robot1_pos.shape) == 1:
                robot1_pos = np.expand_dims(robot1_pos, 0)
            if len(robot2_pos.shape) == 1:
                robot2_pos = np.expand_dims(robot2_pos, 0)
            
            predictions = self.model.predict({
                'map_input': map_input,
                'frontier_input': frontier_input,
                'robot1_pos_input': robot1_pos,
                'robot2_pos_input': robot2_pos
            }, verbose=0)
            
            return predictions
            
        except Exception as e:
            print(f"預測錯誤: {e}")
            return None

class DRLAssigner(Node):
    def __init__(self):
        super().__init__('drl_assigner')
        
        # 基本參數設置
        self.map_size_for_model = (84, 84)
        self.max_frontiers = 50
        
        # 聲明參數
        default_model_path = os.path.join(
            os.path.expanduser('~'),
            'rrt_ws/src/ros2_humble_gmapping_multi_rrt/rrt_exploration_ros2/rrt_exploration_ros2/saved_models/',
            'robot_rl_model.h5'
        )
        
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('use_drl', TENSORFLOW_AVAILABLE)  # 基於 TensorFlow 可用性設置默認值
        self.declare_parameter('fallback_method', 'smart_distance')  # 新增參數
        
        self.model_path = self.get_parameter('model_path').value
        self.use_drl = self.get_parameter('use_drl').value and TENSORFLOW_AVAILABLE
        self.fallback_method = self.get_parameter('fallback_method').value
        
        # 初始化變量
        self.robot1_pose = None
        self.robot2_pose = None
        self.available_points = []
        self.assigned_targets = {'robot1': None, 'robot2': None}
        self.robot_status = {'robot1': True, 'robot2': True}
        
        # 機器人速度相關變量
        self.robot_velocities = {'robot1': None, 'robot2': None}
        self.velocity_check_threshold = 0.01
        self.static_duration = {'robot1': 0.0, 'robot2': 0.0}
        self.static_threshold = 2.0
        
        # 地圖相關變量
        self.map_data = None
        self.map_resolution = None
        self.map_width = None
        self.map_height = None
        self.map_origin = None
        self.processed_map = None
        
        # 目標到達閾值
        self.target_threshold = 0.3
        
        # 初始化DRL模型
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
        
        # 報告狀態
        if self.use_drl and self.drl_model and hasattr(self.drl_model, 'is_loaded') and self.drl_model.is_loaded:
            method = "DRL增強智能分配"
        else:
            method = f"智能{self.fallback_method}分配"
            
        self.get_logger().info(f'{method}節點已啟動')
        if self.use_drl:
            self.get_logger().info(f'模型路徑: {self.model_path}')

    def load_drl_model(self):
        """載入 DRL 模型"""
        if not TENSORFLOW_AVAILABLE:
            self.get_logger().warn('TensorFlow 不可用，無法載入 DRL 模型')
            self.use_drl = False
            return
            
        try:
            self.drl_model = SimpleNetworkModel(
                input_shape=(84, 84, 1),
                max_frontiers=self.max_frontiers
            )
            
            # 尋找模型文件
            model_paths_to_try = [
                self.model_path,
                os.path.join(os.path.dirname(self.model_path), 'multi_robot_model.h5'),
                os.path.join(os.path.dirname(self.model_path), 'frontier_model.h5'),
                os.path.join(os.path.dirname(self.model_path), 'best_model.h5')
            ]
            
            model_loaded = False
            for path in model_paths_to_try:
                if os.path.exists(path):
                    self.get_logger().info(f'嘗試載入模型: {path}')
                    if self.drl_model.load(path):
                        self.model_path = path
                        model_loaded = True
                        break
                        
            if not model_loaded:
                # 嘗試在模型目錄中尋找任何 .h5 文件
                model_dir = os.path.dirname(self.model_path)
                if os.path.exists(model_dir):
                    h5_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
                    if h5_files:
                        random_model = os.path.join(model_dir, h5_files[0])
                        self.get_logger().info(f'嘗試載入找到的模型: {random_model}')
                        if self.drl_model.load(random_model):
                            self.model_path = random_model
                            model_loaded = True
                            
            if not model_loaded:
                self.get_logger().warn('無法載入任何 DRL 模型，將使用智能距離基礎分配')
                self.use_drl = False
                self.drl_model = None
            else:
                self.get_logger().info(f'成功載入 DRL 模型: {self.model_path}')
                
        except Exception as e:
            self.get_logger().error(f'載入 DRL 模型時發生錯誤: {str(e)}')
            self.use_drl = False
            self.drl_model = None

    def setup_subscribers(self):
        """設置所有訂閱者"""
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/merge_map', self.map_callback, 10
        )
        self.robot1_pose_sub = self.create_subscription(
            PoseStamped, '/robot1_pose', self.robot1_pose_callback, 10
        )
        self.robot2_pose_sub = self.create_subscription(
            PoseStamped, '/robot2_pose', self.robot2_pose_callback, 10
        )
        self.filtered_points_sub = self.create_subscription(
            MarkerArray, '/filtered_points', self.filtered_points_callback, 10
        )
        self.robot1_cmd_vel_sub = self.create_subscription(
            Twist, '/robot1/cmd_vel', lambda msg: self.cmd_vel_callback(msg, 'robot1'), 10
        )
        self.robot2_cmd_vel_sub = self.create_subscription(
            Twist, '/robot2/cmd_vel', lambda msg: self.cmd_vel_callback(msg, 'robot2'), 10
        )

    def setup_publishers(self):
        """設置所有發布者"""
        self.robot1_target_pub = self.create_publisher(
            PoseStamped, '/robot1/goal_pose', 10
        )
        self.robot2_target_pub = self.create_publisher(
            PoseStamped, '/robot2/goal_pose', 10
        )
        self.target_viz_pub = self.create_publisher(
            MarkerArray, '/assigned_targets_viz', 10
        )
        self.debug_pub = self.create_publisher(
            String, '/assigner/debug', 10
        )

    def process_map_for_model(self, occupancy_grid):
        """處理地圖數據為模型輸入格式"""
        if occupancy_grid is None:
            return None
            
        try:
            map_array = np.array(occupancy_grid)
            map_binary = np.zeros_like(map_array, dtype=np.uint8)
            
            map_binary[map_array == 0] = 255    # 自由空間 -> 白色
            map_binary[map_array == 100] = 0    # 障礙物 -> 黑色
            map_binary[map_array == -1] = 127   # 未知空間 -> 灰色
            
            resized_map = cv2.resize(
                map_binary, self.map_size_for_model, interpolation=cv2.INTER_LINEAR
            )
            
            normalized_map = resized_map.astype(np.float32) / 255.0
            processed_map = np.expand_dims(normalized_map, axis=-1)
            
            return processed_map
            
        except Exception as e:
            self.get_logger().error(f'地圖處理錯誤: {str(e)}')
            return None

    def pad_frontiers(self, frontiers):
        """填充 frontier 點到固定長度並標準化"""
        padded = np.zeros((self.max_frontiers, 2))
        
        if len(frontiers) > 0:
            frontiers = np.array(frontiers)
            
            if self.map_width and self.map_height and self.map_resolution:
                map_width_m = self.map_width * self.map_resolution
                map_height_m = self.map_height * self.map_resolution
                
                normalized_frontiers = frontiers.copy()
                normalized_frontiers[:, 0] = (frontiers[:, 0] - self.map_origin.position.x) / map_width_m
                normalized_frontiers[:, 1] = (frontiers[:, 1] - self.map_origin.position.y) / map_height_m
                normalized_frontiers = np.clip(normalized_frontiers, 0.0, 1.0)
            else:
                normalized_frontiers = frontiers
            
            n_frontiers = min(len(frontiers), self.max_frontiers)
            padded[:n_frontiers] = normalized_frontiers[:n_frontiers]
        
        return padded

    def get_normalized_position(self, pose):
        """獲取標準化的機器人位置"""
        if not pose or not self.map_width or not self.map_height or not self.map_resolution:
            return np.array([0.0, 0.0])
            
        map_width_m = self.map_width * self.map_resolution
        map_height_m = self.map_height * self.map_resolution
        
        normalized_x = (pose.position.x - self.map_origin.position.x) / map_width_m
        normalized_y = (pose.position.y - self.map_origin.position.y) / map_height_m
        
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
        
        if self.use_drl:
            self.processed_map = self.process_map_for_model(self.map_data)

    def robot1_pose_callback(self, msg):
        self.robot1_pose = msg.pose

    def robot2_pose_callback(self, msg):
        self.robot2_pose = msg.pose

    def filtered_points_callback(self, msg):
        self.available_points = []
        if msg.markers:
            for marker in msg.markers:
                self.available_points.extend([(p.x, p.y) for p in marker.points])

    def cmd_vel_callback(self, msg: Twist, robot_name: str):
        total_velocity = abs(msg.linear.x) + abs(msg.linear.y) + abs(msg.angular.z)
        self.robot_velocities[robot_name] = total_velocity

    def check_robot_motion(self):
        """檢查機器人運動狀態"""
        for robot_name in ['robot1', 'robot2']:
            if self.robot_velocities[robot_name] is None:
                continue
                
            if self.robot_velocities[robot_name] < self.velocity_check_threshold:
                self.static_duration[robot_name] += 0.1
                
                if (self.static_duration[robot_name] >= self.static_threshold and 
                    not self.robot_status[robot_name]):
                    self.get_logger().info(f'{robot_name} 已靜止 {self.static_threshold} 秒，標記為可用')
                    self.robot_status[robot_name] = True
                    self.assigned_targets[robot_name] = None
            else:
                self.static_duration[robot_name] = 0.0

    def check_target_reached(self):
        """檢查機器人是否到達目標"""
        robots = {'robot1': self.robot1_pose, 'robot2': self.robot2_pose}

        for robot_name, robot_pose in robots.items():
            if not robot_pose or not self.assigned_targets[robot_name]:
                continue

            target = self.assigned_targets[robot_name]
            current_pos = (robot_pose.position.x, robot_pose.position.y)
            
            distance = np.sqrt(
                (current_pos[0] - target[0])**2 + (current_pos[1] - target[1])**2
            )

            if distance < self.target_threshold:
                if not self.robot_status[robot_name]:
                    self.get_logger().info(f'{robot_name} 已到達目標點')
                self.robot_status[robot_name] = True
                self.assigned_targets[robot_name] = None
            else:
                self.robot_status[robot_name] = False

    def calculate_utility_score(self, robot_pose, target_point, other_robot_pose=None, other_target=None):
        """計算效用分數"""
        if not robot_pose:
            return float('-inf')
        
        robot_pos = (robot_pose.position.x, robot_pose.position.y)
        
        # 距離分數（距離越近分數越高）
        distance = np.sqrt((robot_pos[0] - target_point[0])**2 + (robot_pos[1] - target_point[1])**2)
        distance_score = 1.0 / (1.0 + distance)
        
        # 避免衝突懲罰
        conflict_penalty = 0.0
        if other_robot_pose and other_target:
            other_pos = (other_robot_pose.position.x, other_robot_pose.position.y)
            
            # 如果另一個機器人更接近這個目標
            other_distance = np.sqrt((other_pos[0] - target_point[0])**2 + (other_pos[1] - target_point[1])**2)
            if other_distance < distance:
                conflict_penalty += 0.5
            
            # 如果目標太接近另一個機器人的目標
            if other_target:
                target_distance = np.sqrt((target_point[0] - other_target[0])**2 + (target_point[1] - other_target[1])**2)
                if target_distance < 2.0:
                    conflict_penalty += 0.3
        
        return distance_score - conflict_penalty

    def smart_distance_assignment(self, robot_name, available_points, assigned_points):
        """智能距離基礎分配"""
        robot_pose = self.robot1_pose if robot_name == 'robot1' else self.robot2_pose
        other_robot_pose = self.robot2_pose if robot_name == 'robot1' else self.robot1_pose
        other_robot_name = 'robot2' if robot_name == 'robot1' else 'robot1'
        other_target = self.assigned_targets.get(other_robot_name)
        
        if not robot_pose:
            return None
            
        valid_targets = []
        MIN_DISTANCE = 0.5
        
        for point in available_points:
            if tuple(point) in assigned_points:
                continue
                
            distance = np.sqrt(
                (robot_pose.position.x - point[0])**2 + (robot_pose.position.y - point[1])**2
            )
            
            if distance >= MIN_DISTANCE:
                utility_score = self.calculate_utility_score(
                    robot_pose, point, other_robot_pose, other_target
                )
                valid_targets.append((point, utility_score, distance))
        
        if valid_targets:
            best_target = max(valid_targets, key=lambda x: x[1])
            return best_target[0]
            
        return None

    def drl_predict_frontier(self, robot_name):
        """使用 DRL 模型預測最佳 frontier"""
        if (not self.use_drl or not self.drl_model or 
            not hasattr(self.drl_model, 'is_loaded') or not self.drl_model.is_loaded or
            self.processed_map is None or len(self.available_points) == 0):
            return None
            
        try:
            frontiers = self.pad_frontiers(self.available_points)
            robot1_pos = self.get_normalized_position(self.robot1_pose)
            robot2_pos = self.get_normalized_position(self.robot2_pose)
            
            predictions = self.drl_model.predict(
                self.processed_map, frontiers, robot1_pos, robot2_pos
            )
            
            if predictions is None:
                return None
                
            valid_frontiers = min(self.max_frontiers, len(self.available_points))
            if robot_name == 'robot1':
                q_values = predictions['robot1'][0, :valid_frontiers]
            else:
                q_values = predictions['robot2'][0, :valid_frontiers]
            
            best_action = np.argmax(q_values)
            return self.available_points[best_action]
            
        except Exception as e:
            self.get_logger().error(f'DRL 預測錯誤: {str(e)}')
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
        """發布可視化標記"""
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
        """主要的目標分配邏輯"""
        if (not self.available_points or 
            self.robot1_pose is None or 
            self.robot2_pose is None):
            return

        assigned_points = set()
        for robot, target in self.assigned_targets.items():
            if target is not None:
                assigned_points.add(tuple(target))

        for robot_name in ['robot1', 'robot2']:
            if not self.robot_status[robot_name] or self.assigned_targets[robot_name] is not None:
                continue

            available_points = [
                point for point in self.available_points 
                if tuple(point) not in assigned_points
            ]
            
            if not available_points:
                continue

            # 嘗試使用 DRL 模型
            best_point = None
            method_used = "智能距離基礎"
            
            if self.use_drl and self.drl_model and hasattr(self.drl_model, 'is_loaded') and self.drl_model.is_loaded:
                best_point = self.drl_predict_frontier(robot_name)
                if best_point:
                    method_used = "DRL模型"
            
            # 後備方法
            if best_point is None:
                if self.fallback_method == 'smart_distance':
                    best_point = self.smart_distance_assignment(robot_name, available_points, assigned_points)
                else:
                    # 簡單的最近距離分配
                    robot_pose = self.robot1_pose if robot_name == 'robot1' else self.robot2_pose
                    if robot_pose:
                        distances = [
                            np.sqrt((robot_pose.position.x - point[0])**2 + (robot_pose.position.y - point[1])**2)
                            for point in available_points
                        ]
                        if distances:
                            min_idx = np.argmin(distances)
                            best_point = available_points[min_idx]

            if best_point is not None:
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
    
    try:
        node = DRLAssigner()
        
        # 打印啟動信息
        node.get_logger().info("=" * 50)
        node.get_logger().info("DRL Assigner 節點狀態:")
        node.get_logger().info(f"TensorFlow 可用: {TENSORFLOW_AVAILABLE}")
        
        if TENSORFLOW_AVAILABLE:
            node.get_logger().info(f"TensorFlow 版本: {tf.__version__}")
        
        node.get_logger().info(f"使用 DRL: {node.use_drl}")
        
        if node.use_drl and node.drl_model:
            if hasattr(node.drl_model, 'is_loaded'):
                node.get_logger().info(f"DRL 模型已載入: {node.drl_model.is_loaded}")
            else:
                node.get_logger().info("DRL 模型狀態: 未知")
        
        node.get_logger().info(f"後備方法: {node.fallback_method}")
        node.get_logger().info("=" * 50)
        
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        print("收到中斷信號，正在關閉...")
    except Exception as e:
        print(f'錯誤: {str(e)}')
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