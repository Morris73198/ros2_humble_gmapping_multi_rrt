// slam_gmapping.h
#ifndef SLAM_GMAPPING_SLAM_GMAPPING_H_
#define SLAM_GMAPPING_SLAM_GMAPPING_H_

#include <mutex>
#include <thread>
#include <memory>
#include <deque>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "std_msgs/msg/float64.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "nav_msgs/msg/map_meta_data.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2/utils.h"
#include "message_filters/subscriber.h"
#include "tf2_ros/message_filter.h"

#include "gmapping/gridfastslam/gridslamprocessor.h"
#include "gmapping/sensor/sensor_base/sensor.h"
#include "gmapping/sensor/sensor_range/rangesensor.h"
#include "gmapping/sensor/sensor_odometry/odometrysensor.h"

class SlamGmapping : public rclcpp::Node {
public:
    SlamGmapping();
    ~SlamGmapping() override;

    void init();
    void startLiveSlam();
    void publishTransform();

private:
    rclcpp::Node::SharedPtr node_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr entropy_publisher_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr sst_;
    rclcpp::Publisher<nav_msgs::msg::MapMetaData>::SharedPtr sstm_;

    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr robot1_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr robot2_pose_sub_;
    
    std::deque<geometry_msgs::msg::PoseStamped> robot1_history_;
    std::deque<geometry_msgs::msg::PoseStamped> robot2_history_;
    std::mutex pose_mutex_;
    static const size_t MAX_HISTORY_SIZE = 100000;

    std::shared_ptr<tf2_ros::Buffer> buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tfl_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::LaserScan>> scan_filter_sub_;
    std::shared_ptr<tf2_ros::MessageFilter<sensor_msgs::msg::LaserScan>> scan_filter_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tfB_;

    GMapping::GridSlamProcessor* gsp_;
    GMapping::RangeSensor* gsp_laser_;
    std::vector<double> laser_angles_;
    geometry_msgs::msg::PoseStamped centered_laser_pose_;
    bool do_reverse_range_;
    unsigned int gsp_laser_beam_count_;
    GMapping::OdometrySensor* gsp_odom_;

    bool got_first_scan_;
    bool got_map_;
    nav_msgs::msg::OccupancyGrid map_;

    tf2::Duration map_update_interval_;
    tf2::Transform map_to_odom_;
    std::mutex map_to_odom_mutex_;
    std::mutex map_mutex_;

    int laser_count_;
    int throttle_scans_;
    int clear_radius_;

    std::shared_ptr<std::thread> transform_thread_;

    std::string base_frame_;
    std::string laser_frame_;
    std::string map_frame_;
    std::string odom_frame_;

    void updateMap(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan);
    bool getOdomPose(GMapping::OrientedPoint& gmap_pose, const rclcpp::Time& t);
    bool initMapper(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan);
    bool addScan(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan, GMapping::OrientedPoint& gmap_pose);
    double computePoseEntropy();
    void clearRobotTraces(nav_msgs::msg::OccupancyGrid& map);
    void robot1PoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void robot2PoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
    void publishLoop(double transform_publish_period);
    void laserCallback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan);

    double maxRange_;
    double maxUrange_;
    double minimum_score_;
    double sigma_;
    int kernelSize_;
    double lstep_;
    double astep_;
    int iterations_;
    double lsigma_;
    double ogain_;
    int lskip_;
    double srr_;
    double srt_;
    double str_;
    double stt_;
    double linearUpdate_;
    double angularUpdate_;
    double temporalUpdate_;
    double resampleThreshold_;
    int particles_;
    double xmin_;
    double ymin_;
    double xmax_;
    double ymax_;
    double delta_;
    double occ_thresh_;
    double llsamplerange_;
    double llsamplestep_;
    double lasamplerange_;
    double lasamplestep_;
    double transform_publish_period_;
    double tf_delay_;
    unsigned long int seed_;
};

#endif
