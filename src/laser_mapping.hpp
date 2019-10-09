#ifndef LASER_MAPPING_HPP
#define LASER_MAPPING_HPP

#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <math.h>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <string>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <thread>
#include <vector>
#include <future>

#include "ceres_icp.hpp"
#include "tools/common.h"
#include "tools/logger.hpp"
#include "tools/pcl_tools.hpp"
#include "tools/timer.hpp"
#include "point_cloud_registration.hpp"
#include "scene_alignment.hpp"
#include "ceres_pose_graph_3d.hpp"
#include "cell_map_hash.hpp"

#define PUB_SURROUND_PTS 1
#define PCD_SAVE_RAW 1
#define PUB_DEBUG_INFO 0

int g_if_undistore = 0;
int MOTION_DEBLUR = 0;
double history_add_t_step = 0.00;
double history_add_angle_step = 0.00;

using namespace PCL_TOOLS;
using namespace COMMON_TOOLS;

struct DataPair
{
    sensor_msgs::PointCloud2ConstPtr pc_corner;
    sensor_msgs::PointCloud2ConstPtr pc_full;
    sensor_msgs::PointCloud2ConstPtr pc_plane;
    bool has_pc_corner = 0;
    bool has_pc_full = 0;
    bool has_pc_plane = 0;

    void add_pc_corner(sensor_msgs::PointCloud2ConstPtr ros_pc)
    {
        pc_corner = ros_pc;
        has_pc_corner = true;
    }

    void add_pc_plane(sensor_msgs::PointCloud2ConstPtr ros_pc)
    {
        pc_plane = ros_pc;
        has_pc_plane = true;
    }

    void add_pc_full(sensor_msgs::PointCloud2ConstPtr ros_pc)
    {
        pc_full = ros_pc;
        has_pc_full = true;
    }

    bool is_completed()
    {
        return (has_pc_corner & has_pc_full & has_pc_plane);
    }
};

class PointCloudRegistration;

class LaserMapping
{
public:
    int current_frame_index = 0;
    int m_para_min_match_blur = 0.0;
    int m_para_max_match_blur = 0.3;
    int max_buffer_size = 50000000;

    int m_mapping_init_accumulate_frames = 100;
    int m_kmean_filter_count = 3;
    int m_kmean_filter_threshold = 2.0;

    double time_pc_corner_past = 0;
    double m_time_pc_surface_past = 0;
    double m_time_pc_full = 0;
    double time_odom = 0;
    double last_time_stamp = 0;
    double minimum_pt_time_stamp = 0;
    double maximum_pt_time_stamp = 1.0;
    float m_last_max_blur = 0.0;

    int m_odom_mode;
    int matching_mode = 0;
    int if_input_downsample_mode = 1;
    int maximum_parallel_thread;
    int maximum_mapping_buff_thread = 1; // Maximum number of thead for matching buffer update
    int maximum_history_size = 100;
    int m_maximum_pt_in_cell = 1e5;
    int m_maximum_cell_life_time = 10;

    float m_para_max_angular_rate = 200.0 / 50.0; // max angular rate = 90.0 /50.0 deg/s
    float m_para_max_speed = 100.0 / 50.0;        // max speed = 10 m/s
    float m_max_final_cost = 100.0;
    int m_para_icp_max_iterations = 20;
    int m_para_cere_max_iterations = 100;
    double m_minimum_icp_R_diff = 0.01;
    double m_minimum_icp_T_diff = 0.01;

    string pcd_save_dir_name, log_save_dir_name, loop_save_dir_name;

    std::list<pcl::PointCloud<PointType>> pc_corner_history;
    std::list<pcl::PointCloud<PointType>> pc_surface_history;
    std::list<pcl::PointCloud<PointType>> pc_full_history;
    std::list<double> his_reg_error;
    Eigen::Quaterniond last_his_add_q;
    Eigen::Vector3d last_his_add_t;

    std::map<int, float> m_map_life_time_corner;
    std::map<int, float> m_map_life_time_surface;

    // ouput: all visualble cube points
    pcl::PointCloud<PointType>::Ptr laser_cloud_surround;

    // surround points in map to build tree
    int if_mapping_updated_corner = true;
    int if_mapping_updated_surface = true;

    pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map;
    pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map;

    pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map_latest;
    pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map_latest;

    //input & output: points in one frame. local --> global
    pcl::PointCloud<PointType>::Ptr pc_full;

    // input: from odom
    pcl::PointCloud<PointType>::Ptr pc_corner_latest;
    pcl::PointCloud<PointType>::Ptr pc_surf_latest;

    //kd-tree
    pcl::KdTreeFLANN<PointType> m_kdtree_corner_from_map_;
    pcl::KdTreeFLANN<PointType> m_kdtree_surf_from_map;

    pcl::KdTreeFLANN<PointType> kdtree_corner_from_map_last;
    pcl::KdTreeFLANN<PointType> kdtree_surf_from_map_last;

    int m_laser_cloud_valid_Idx[1024];
    int laser_cloud_surround_Idx[1024];

    const Eigen::Quaterniond m_q_I = Eigen::Quaterniond(1, 0, 0, 0);

    double para_buffer_RT[7] = {0, 0, 0, 1, 0, 0, 0};
    double para_buffer_RT_last[7] = {0, 0, 0, 1, 0, 0, 0};

    Eigen::Map<Eigen::Quaterniond> q_w_curr = Eigen::Map<Eigen::Quaterniond>(para_buffer_RT);
    Eigen::Map<Eigen::Vector3d> t_w_curr = Eigen::Map<Eigen::Vector3d>(para_buffer_RT + 4);

    Eigen::Map<Eigen::Quaterniond> q_w_last = Eigen::Map<Eigen::Quaterniond>(para_buffer_RT_last);
    Eigen::Map<Eigen::Vector3d> t_w_last = Eigen::Map<Eigen::Vector3d>(para_buffer_RT_last + 4);

    std::map<double, DataPair *> map_data_pair;
    std::queue<DataPair *> queue_avail_data;

    std::queue<nav_msgs::Odometry::ConstPtr> m_odom_que;
    std::mutex mutex_buf;


    float line_resolution = 0;
    float plane_resolution = 0;
    pcl::VoxelGrid<PointType> down_sample_filter_corner_;
    pcl::VoxelGrid<PointType> down_sample_filter_surface_;
    pcl::StatisticalOutlierRemoval<PointType> m_filter_k_means;

    std::vector<int> m_point_search_Idx;
    std::vector<float> m_point_search_sq_dis;

    nav_msgs::Path laser_after_mapped_path;

    int m_if_save_to_pcd_files = 1;
    PCL_tools m_pcl_tools_aftmap;
    PCL_tools m_pcl_tools_raw;

    COMMON_TOOLS::File_logger logger_common;
    COMMON_TOOLS::File_logger logger_pcd;
    COMMON_TOOLS::File_logger m_logger_timer;
    COMMON_TOOLS::File_logger m_logger_matching_buff;
    SceneAlignment<float> m_sceene_align;
    COMMON_TOOLS::Timer m_timer;

    ros::Publisher pub_laser_cloud_surround;
    ros::Publisher pub_laser_cloud_map;
    ros::Publisher pub_pc_full;
    ros::Publisher pub_odom_aft_mapped;
    ros::Publisher pub_odom_aft_mapped_hight_frec;
    ros::Publisher pub_laser_aft_mapped_path;

    ros::NodeHandle ros_node_handle;

    ros::Subscriber sub_pc_corner_latest;
    ros::Subscriber sub_pc_surf_latest;
    ros::Subscriber sub_laser_odom;
    ros::Subscriber sub_pc_full;

    ceres::Solver::Summary m_final_opt_summary;

    int if_loop_closure;
    std::list<std::future<int> *> thread_pool;
    std::list<std::future<void> *> thread_match_buff_refresh;

    double m_maximum_in_fov_angle ;
    double m_maximum_pointcloud_delay_time;
    double m_maximum_search_range_corner;
    double m_maximum_search_range_surface;
    double surround_pointcloud_resolution;
    double last_pc_reg_time = -3e8;
    double latest_pc_matching_refresh_time = -3e8;
    double last_pc_income_time = -3e8;

    std::mutex mutex_mapping;
    std::mutex mutex_querypointcloud;
    std::mutex mutex_buff_for_matching_corner;
    std::mutex mutex_buff_for_matching_surface;
    std::mutex m_mutex_thread_pool;
    std::mutex m_mutex_ros_pub;
    std::mutex mutex_dump_full_history;

    float pt_cell_resolution = 1.0;
    PointCloudMap<float> pt_cell_map_full;
    PointCloudMap<float> pt_cell_map_corners;
    PointCloudMap<float> pt_cell_map_planes;

    int down_sample_replace = 1;
    ros::Publisher pub_last_corner_pts, pub_last_surface_pts;
    ros::Publisher pub_match_corner_pts, pub_match_surface_pts, pub_debug_pts, pub_pc_aft_loop;
    std::future<void> *m_mapping_refresh_service_corner , *m_mapping_refresh_service_surface, *mapping_refresh_service; // Thread for mapping update
    std::future<void> *service_pub_surround_pts_, *service_loop_detection_; // Thread for loop detection and publish surrounding pts

    int if_pt_in_fov(const Eigen::Matrix<double,3,1> &pt)
    {
        auto pt_affine = q_w_curr.inverse() * (pt - t_w_curr);

        if (pt_affine(0) < 0)
            return 0;

        float angle = Eigen_math::vector_angle(pt_affine, Eigen::Matrix<double, 3, 1>(1, 0, 0), 1);

        if (angle * 57.3 < m_maximum_in_fov_angle)
            return 1;
        else
            return 0;
    }


    void update_cude_life_time(std::map<int, float> & map_life_time, int index)
    {
        std::map<int, float>::iterator it = map_life_time.find(index);
        if (it == map_life_time.end())
            map_life_time.insert(std::make_pair(index, last_time_stamp));
        else
            it->second = last_time_stamp;
    }

    void update_buff_for_matching()
    {
        if (latest_pc_matching_refresh_time == last_pc_reg_time)
            return;
        m_timer.tic("Update buff for matching");
        pcl::VoxelGrid<PointType> down_sample_filter_corner = down_sample_filter_corner_;
        pcl::VoxelGrid<PointType> down_sample_filter_surface = down_sample_filter_surface_;
        down_sample_filter_corner.setLeafSize(line_resolution, line_resolution, line_resolution);
        down_sample_filter_surface.setLeafSize(plane_resolution, plane_resolution, plane_resolution);
        pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map(new pcl::PointCloud<PointType>());

        if (matching_mode) // 0
        {
            pcl::VoxelGrid<PointType> down_sample_filter_corner = down_sample_filter_corner_;
            pcl::VoxelGrid<PointType> down_sample_filter_surface = down_sample_filter_surface_;
            std::vector<PointCloudMap<float>::PC_CELL *> corner_cell_vec = pt_cell_map_corners.find_cells_in_radius(t_w_curr, m_maximum_search_range_corner);
            std::vector<PointCloudMap<float>::PC_CELL *> plane_cell_vec = pt_cell_map_planes.find_cells_in_radius(t_w_curr, m_maximum_search_range_surface);
            int corner_cell_numbers_in_fov = 0;
            int surface_cell_numbers_in_fov = 0;
            pcl::PointCloud<PointType> pc_temp;

            for (size_t i = 0; i < corner_cell_vec.size(); i++)
            {
                int if_in_fov = if_pt_in_fov(corner_cell_vec[i]->center.cast<double>());
                if (if_in_fov == 0)
                    continue;
                corner_cell_numbers_in_fov++;
                down_sample_filter_corner.setInputCloud(corner_cell_vec[i]->get_pointcloud().makeShared());
                down_sample_filter_corner.filter(pc_temp);
                if (down_sample_replace)
                    corner_cell_vec[i]->set_pointcloud(pc_temp);
                *laser_cloud_corner_from_map += pc_temp;
            }

            for (size_t i = 0; i < plane_cell_vec.size(); i++)
            {
                int if_in_fov = if_pt_in_fov(plane_cell_vec[i]->center.cast<double>());
                if (if_in_fov == 0)
                    continue;
                surface_cell_numbers_in_fov++;

                down_sample_filter_surface.setInputCloud(plane_cell_vec[i]->get_pointcloud().makeShared());
                down_sample_filter_surface.filter(pc_temp);
                if (down_sample_replace)
                    plane_cell_vec[i]->set_pointcloud(pc_temp);
                *laser_cloud_surf_from_map += pc_temp;
            }
        }
        else
        {
            mutex_mapping.lock();
            for (auto it = pc_corner_history.begin(); it != pc_corner_history.end(); it++)
                *laser_cloud_corner_from_map += (*it);
            for (auto it = pc_surface_history.begin(); it != pc_surface_history.end(); it++)
                *laser_cloud_surf_from_map += (*it);
            mutex_mapping.unlock();
        }

        down_sample_filter_corner.setInputCloud(laser_cloud_corner_from_map);
        down_sample_filter_corner.filter(*laser_cloud_corner_from_map);

        down_sample_filter_surface.setInputCloud(laser_cloud_surf_from_map);
        down_sample_filter_surface.filter(*laser_cloud_surf_from_map);

        pcl::KdTreeFLANN<PointType> kdtree_corner_from_map;
        pcl::KdTreeFLANN<PointType> kdtree_surf_from_map;

        if (laser_cloud_corner_from_map->points.size() && laser_cloud_surf_from_map->points.size())
        {
            kdtree_corner_from_map.setInputCloud(laser_cloud_corner_from_map);
            kdtree_surf_from_map.setInputCloud(laser_cloud_surf_from_map);
        }

        if_mapping_updated_corner = false;
        if_mapping_updated_surface = false;

        mutex_buff_for_matching_corner.lock();
        *laser_cloud_corner_from_map_latest = *laser_cloud_corner_from_map;
        kdtree_corner_from_map_last = kdtree_corner_from_map;
        mutex_buff_for_matching_surface.unlock();

        mutex_buff_for_matching_surface.lock();
        *laser_cloud_surf_from_map_latest = *laser_cloud_surf_from_map;
        kdtree_surf_from_map_last = kdtree_surf_from_map;
        mutex_buff_for_matching_corner.unlock();

        if (last_pc_reg_time > latest_pc_matching_refresh_time || last_pc_reg_time < 10)
            latest_pc_matching_refresh_time = last_pc_reg_time;

        *m_logger_matching_buff.get_ostream() << m_timer.toc_string("Update buff for matching") << std::endl;
    }

    void service_update_buff_for_matching()
    {
        while (1)
        {
            std::this_thread::sleep_for(std::chrono::nanoseconds(100));
            update_buff_for_matching();
        }
    }

    void get_pts_from_mapping(pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map,
                              pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map)
    {
        if (if_mapping_updated_corner == false)
        {
            cout << "=== Mapping is old, return lastest mapping ===" << endl;
            *laser_cloud_corner_from_map = *laser_cloud_corner_from_map_latest;
            *laser_cloud_surf_from_map = *laser_cloud_surf_from_map_latest;
            return;
        }

        laser_cloud_corner_from_map->clear();
        laser_cloud_surf_from_map->clear();

        if (matching_mode)
        {
            pcl::VoxelGrid<PointType> down_sample_filter_corner = down_sample_filter_corner_;
            pcl::VoxelGrid<PointType> down_sample_filter_surface = down_sample_filter_surface_;
            std::vector<PointCloudMap<float>::PC_CELL *> corner_cell_vec = pt_cell_map_corners.find_cells_in_radius(t_w_curr, m_maximum_search_range_corner);
            std::vector<PointCloudMap<float>::PC_CELL *> plane_cell_vec = pt_cell_map_planes.find_cells_in_radius(t_w_curr, m_maximum_search_range_surface);
            int corner_cell_numbers_full = corner_cell_vec.size();
            int corner_cell_numbers_in_fov = 0;
            int surface_cell_numbers_full = plane_cell_vec.size();
            int surface_cell_numbers_in_fov = 0;
            pcl::PointCloud<PointType> pc_temp;
            for (size_t i = 0; i < corner_cell_vec.size(); i++)
            {
                int if_in_fov = if_pt_in_fov(corner_cell_vec[i]->center.cast<double>());
                if (if_in_fov == 0)
                    continue;
                corner_cell_numbers_in_fov++;
                down_sample_filter_corner.setInputCloud(corner_cell_vec[i]->pointcloud.makeShared());
                down_sample_filter_corner.filter(pc_temp);
                *laser_cloud_corner_from_map += pc_temp;
            }

            for (size_t i = 0; i < plane_cell_vec.size(); i++)
            {
                int if_in_fov = if_pt_in_fov(plane_cell_vec[i]->center.cast<double>());
                if (if_in_fov == 0)
                    continue;
                surface_cell_numbers_in_fov++;
                down_sample_filter_surface.setInputCloud(plane_cell_vec[i]->pointcloud.makeShared());
                down_sample_filter_surface.filter(pc_temp);
                *laser_cloud_surf_from_map += pc_temp;
            }
            printf("==== Ratio of corners in fovs %.2f, surface %.2f ====\r\n",
                    (float) corner_cell_numbers_in_fov / corner_cell_numbers_full,
                    (float) surface_cell_numbers_in_fov / surface_cell_numbers_full);
        }
        else
        {
            for (auto it = pc_corner_history.begin(); it != pc_corner_history.end(); it++)
                *laser_cloud_corner_from_map += (*it);

            for (auto it = pc_surface_history.begin(); it != pc_surface_history.end(); it++)
                *laser_cloud_surf_from_map += (*it);
        }

        if_mapping_updated_corner = false;
        *laser_cloud_corner_from_map_latest = *laser_cloud_corner_from_map;
        *laser_cloud_surf_from_map_latest = *laser_cloud_surf_from_map;
    }

    LaserMapping()
    {
        pc_corner_latest = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
        pc_surf_latest = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
        laser_cloud_surround = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());

        laser_cloud_corner_from_map = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
        laser_cloud_surf_from_map = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());

        laser_cloud_corner_from_map_latest = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
        laser_cloud_surf_from_map_latest = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());

        pc_full = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());

        init_parameters(ros_node_handle);

        sub_pc_corner_latest =
            ros_node_handle.subscribe<sensor_msgs::PointCloud2>("/pc2_corners", 10000, &LaserMapping::laserCloudCornerHandler, this);
        sub_pc_surf_latest =
            ros_node_handle.subscribe<sensor_msgs::PointCloud2>("/pc2_surface", 10000, &LaserMapping::laserCloudSurfHandler, this);
        sub_pc_full =
            ros_node_handle.subscribe<sensor_msgs::PointCloud2>("/pc2_full", 10000, &LaserMapping::laserCloudFullHandler, this);

        pub_laser_cloud_surround = ros_node_handle.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 10000);
        pub_last_corner_pts = ros_node_handle.advertise<sensor_msgs::PointCloud2>("/features_corners", 10000);
        pub_last_surface_pts = ros_node_handle.advertise<sensor_msgs::PointCloud2>("/features_surface", 10000);
        pub_match_corner_pts = ros_node_handle.advertise<sensor_msgs::PointCloud2>("/match_pc_corners", 10000);
        pub_match_surface_pts = ros_node_handle.advertise<sensor_msgs::PointCloud2>("/match_pc_surface", 10000);
        pub_pc_aft_loop = ros_node_handle.advertise<sensor_msgs::PointCloud2>("/pc_aft_loop_closure", 10000);
        pub_debug_pts =  ros_node_handle.advertise<sensor_msgs::PointCloud2>("/pc_debug", 10000);
        pub_laser_cloud_map = ros_node_handle.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 10000);
        pub_pc_full = ros_node_handle.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 10000);
        pub_odom_aft_mapped = ros_node_handle.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10000);
        pub_odom_aft_mapped_hight_frec = ros_node_handle.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 10000);
        pub_laser_aft_mapped_path = ros_node_handle.advertise<nav_msgs::Path>("/aft_mapped_path", 10000);

        pt_cell_map_full.set_resolution(pt_cell_resolution);
        pt_cell_map_corners.set_resolution(pt_cell_resolution);
        pt_cell_map_planes.set_resolution(pt_cell_resolution);

        ROS_INFO("LaserMapping Initialization OK");
    };

    ~LaserMapping(){};

    DataPair *get_data_pair(const double &time_stamp)
    {
        std::map<double, DataPair *>::iterator it = map_data_pair.find(time_stamp);
        if (it == map_data_pair.end())
        {
            DataPair *date_pair_ptr = new DataPair();
            map_data_pair.insert(std::make_pair(time_stamp, date_pair_ptr));
            return date_pair_ptr;
        }
        else
            return it->second;
    }

    void init_parameters(ros::NodeHandle &nh)
    {
        nh.param<float>("mapping_line_resolution", line_resolution, 0.4);
        nh.param<float>("mapping_plane_resolution", plane_resolution, 0.8);
        nh.param<int>("icp_maximum_iteration", m_para_icp_max_iterations, 20);
        nh.param<int>("ceres_maximum_iteration", m_para_cere_max_iterations, 20);
        nh.param<int>("if_motion_deblur", MOTION_DEBLUR, 1);

        nh.param<float>("max_allow_incre_R", m_para_max_angular_rate, 200.0 / 50.0);
        nh.param<float>("max_allow_incre_T", m_para_max_speed, 100.0 / 50.0);
        nh.param<float>("max_allow_final_cost", m_max_final_cost, 1.0);
        nh.param<int>("maximum_mapping_buffer", max_buffer_size, 5);
        nh.param<int>("mapping_init_accumulate_frames", m_mapping_init_accumulate_frames, 50);

        nh.param<int>("odom_mode", m_odom_mode, 0);
        nh.param<int>("matching_mode", matching_mode, 1);
        nh.param<int>("input_downsample_mode", if_input_downsample_mode, 1);

        nh.param<int>("maximum_parallel_thread", maximum_parallel_thread, 4);
        nh.param<int>("maximum_histroy_buffer",  maximum_history_size , 100);
        nh.param<int>("maximum_pt_in_cell", m_maximum_pt_in_cell, int(1e5));
        nh.param<int>("maximum_cell_life_time", m_maximum_cell_life_time, 10);
        nh.param<double>("minimum_icp_R_diff", m_minimum_icp_R_diff, 0.01);
        nh.param<double>("minimum_icp_T_diff", m_minimum_icp_T_diff, 0.01);

        nh.param<double>("maximum_in_fov_angle", m_maximum_in_fov_angle, 30);
        nh.param<double>("maximum_pointcloud_delay_time", m_maximum_pointcloud_delay_time, 0.1);
        nh.param<double>("maximum_in_fov_angle", m_maximum_in_fov_angle, 30);
        nh.param<double>("maximum_search_range_corner", m_maximum_search_range_corner, 100);
        nh.param<double>("maximum_search_range_surface", m_maximum_search_range_surface, 100);
        nh.param<double>("surround_pointcloud_resolution", surround_pointcloud_resolution, 0.5);

        nh.param<int>("if_save_to_pcd_files", m_if_save_to_pcd_files, 0);
        nh.param<int>("if_loop_closure", if_loop_closure, 0);

        nh.param<std::string>("log_save_dir", log_save_dir_name, "../");
        logger_common.set_log_dir(log_save_dir_name);
        logger_common.init("mapping.log");
        m_logger_timer.set_log_dir(log_save_dir_name);
        m_logger_timer.init("timer.log");
        m_logger_matching_buff.set_log_dir(log_save_dir_name);
        m_logger_matching_buff.init("match_buff.log");

        nh.param<std::string>("pcd_save_dir", pcd_save_dir_name, std::string("./"));
        nh.param<std::string>("loop_save_dir", loop_save_dir_name, pcd_save_dir_name.append("_loop"));
        m_sceene_align.init(loop_save_dir_name);

        if (m_if_save_to_pcd_files)
        {
            m_pcl_tools_aftmap.set_save_dir_name(pcd_save_dir_name);
            m_pcl_tools_raw.set_save_dir_name(pcd_save_dir_name);
        }

        logger_pcd.set_log_dir(log_save_dir_name);
        logger_pcd.init("poses.log");

        LOG_FILE_LINE(logger_common);
        *logger_common.get_ostream() << logger_common.version();

        printf("line resolution: %f, plane resolution: %f \n", line_resolution, plane_resolution);
        logger_common.printf("line resolution %f plane resolution %f \n", line_resolution, plane_resolution);
        down_sample_filter_corner_.setLeafSize(line_resolution, line_resolution, line_resolution);
        down_sample_filter_surface_.setLeafSize(plane_resolution, plane_resolution, plane_resolution);

        m_filter_k_means.setMeanK(m_kmean_filter_count);
        m_filter_k_means.setStddevMulThresh(m_kmean_filter_threshold);
    }

    void laserCloudCornerHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
    {
        std::unique_lock<std::mutex> lock(mutex_buf);
        DataPair *data_pair = get_data_pair(laserCloudCornerLast2->header.stamp.toSec());
        data_pair->add_pc_corner(laserCloudCornerLast2);
        if (data_pair->is_completed())
            queue_avail_data.push(data_pair);
    }

    void laserCloudSurfHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
    {
        std::unique_lock<std::mutex> lock(mutex_buf);
        DataPair *data_pair = get_data_pair(laserCloudSurfLast2->header.stamp.toSec());
        data_pair->add_pc_plane(laserCloudSurfLast2);
        if (data_pair->is_completed())
            queue_avail_data.push(data_pair);
    }

    void laserCloudFullHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
    {
        std::unique_lock<std::mutex> lock(mutex_buf);
        DataPair *data_pair = get_data_pair(laserCloudFullRes2->header.stamp.toSec());
        data_pair->add_pc_full(laserCloudFullRes2);
        if (data_pair->is_completed())
            queue_avail_data.push(data_pair);
    }

    template <typename T1, typename T2>
    static void save_mat_to_json_writter(T1 &writer, const std::string &name, const T2 &eigen_mat)
    {
        writer.Key(name.c_str()); // output a key,
        writer.StartArray(); // Between StartArray()/EndArray(),
        for (size_t i = 0; i < (size_t)(eigen_mat.cols() * eigen_mat.rows()); i++)
            writer.Double(eigen_mat(i));
        writer.EndArray();
    }

    template <typename T1, typename T2>
    static void save_quaternion_to_json_writter(T1 &writer, const std::string &name, const Eigen::Quaternion<T2> &q_curr)
    {
        writer.Key(name.c_str());
        writer.StartArray();
        writer.Double(q_curr.w());
        writer.Double(q_curr.x());
        writer.Double(q_curr.y());
        writer.Double(q_curr.z());
        writer.EndArray();
    }

    template <typename T1, typename T2>
    static void save_data_vec_to_json_writter(T1 &writer, const std::string &name, T2 &data_vec)
    {
        writer.Key(name.c_str());
        writer.StartArray();
        for (auto it = data_vec.begin(); it!=data_vec.end(); it++)
            writer.Double(*it);
        writer.EndArray();
    }

    void dump_pose_and_regerror(std::string file_name,
                                Eigen::Quaterniond &q_curr,
                                Eigen::Vector3d &t_curr,
                                std::list<double> &reg_err_vec)
    {
        rapidjson::Document document;
        rapidjson::StringBuffer sb;
        rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
        writer.StartObject();
        writer.SetMaxDecimalPlaces(1000); // like set_precision
        save_quaternion_to_json_writter(writer, "Q", q_curr);
        save_mat_to_json_writter(writer, "T", t_curr);
        save_data_vec_to_json_writter(writer, "Reg_err", reg_err_vec);
        writer.EndObject();
        std::fstream ofs;
        ofs.open(file_name.c_str(), std::ios_base::out);
        if (ofs.is_open())
        {
            ofs << std::string(sb.GetString()).c_str();
            ofs.close();
        }
    }

    void service_loop_detection()
    {
        int last_update_index = 0;

        sensor_msgs::PointCloud2 ros_laser_cloud_surround;
        pcl::PointCloud<PointType> pt_full;
        Eigen::Quaterniond q_curr;
        Eigen::Vector3d t_curr;
        std::list<double> reg_error_his;
        std::string json_file_name;
        int curren_frame_idx;
        std::vector<PointCloudMap<float> *> pt_map_vec;
        SceneAlignment<float> scene_align;
        MappingRefine<PointType> map_rfn;
        std::vector<std::string> filename_vec;

        std::map<int, std::string> map_file_name;
        Ceres_pose_graph_3d::MapOfPoses pose3d_map, pose3d_map_ori;
        Ceres_pose_graph_3d::VectorOfPose pose3d_vec;
        Ceres_pose_graph_3d::VectorOfConstraints constrain_vec;

        float avail_ratio_plane = 0.05; // 0.05 for 300 scans, 0.15 for 1000 scans
        float avail_ratio_line = 0.05;
        scene_align.init(loop_save_dir_name);
        map_rfn.set_save_dir(std::string(loop_save_dir_name).append("/mapping_refined"));
        map_rfn.set_down_sample_resolution(0.2);
        FILE *fp = fopen(std::string(loop_save_dir_name).append("/loop.log").c_str(), "w+");
        std::map<int, pcl::PointCloud<PointType>> map_id_pc;
        int if_end = 0;

        while (1)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

            if ((current_frame_index - last_update_index < 100) || (current_frame_index < 100))
                continue;

            if (pc_full_history.size() < 0.95 * maximum_history_size)
                continue;

            mutex_dump_full_history.lock();
            q_curr = q_w_curr;
            t_curr = t_w_curr;
            reg_error_his = his_reg_error;
            curren_frame_idx = current_frame_index;
            PointCloudMap<float> *pt_cell_map_temp = new PointCloudMap<float>();
            pt_cell_map_temp->set_resolution(1.0);
            pt_full.clear();

            for (auto it = pc_full_history.begin(); it != pc_full_history.end(); it++)
                pt_full += (*it);

            pt_cell_map_temp->set_point_cloud(PCL_TOOLS::pcl_pts_to_eigen_pts<float, PointType>(pt_full.makeShared()));
            mutex_dump_full_history.unlock();

            map_id_pc.insert(std::make_pair(map_id_pc.size(), pt_full));
            pose3d_vec.push_back(Ceres_pose_graph_3d::Pose3D(q_curr, t_curr));
            pose3d_map.insert(std::make_pair(pose3d_map.size(), Ceres_pose_graph_3d::Pose3D(q_curr, t_curr)));

            if (pose3d_vec.size() >= 2)
            {
                Ceres_pose_graph_3d::Constraint3D temp_csn;
                Eigen::Vector3d relative_T = pose3d_vec[pose3d_vec.size() - 2].q.inverse() * (t_curr - pose3d_vec[pose3d_vec.size() - 2].p);
                Eigen::Quaterniond relative_Q = pose3d_vec[pose3d_vec.size() - 2].q.inverse() * q_curr;
                temp_csn = Ceres_pose_graph_3d::Constraint3D(pose3d_vec.size() - 2, pose3d_vec.size() - 1, relative_Q, relative_T);
                constrain_vec.push_back(temp_csn);
            }

            // Save pose
            json_file_name = std::string(loop_save_dir_name).append("/pose_").append(std::to_string(curren_frame_idx)).append(".json");
            dump_pose_and_regerror(json_file_name, q_curr, t_curr, reg_error_his);
            last_update_index = current_frame_index;
            m_timer.tic("Find loop");
            pt_cell_map_temp->analyze_mapping(1);
            float ratio_non_zero_plane = pt_cell_map_temp->ratio_nonzero_plane;
            float ratio_non_zero_line = pt_cell_map_temp->ratio_nonzero_line;

            // Save mapping
            json_file_name = std::string("mapping_").append(std::to_string(curren_frame_idx)).append(".json");
            pt_cell_map_temp->save_to_file(std::string(loop_save_dir_name), json_file_name);
            pt_map_vec.push_back(pt_cell_map_temp);

            map_file_name.insert(std::make_pair(map_file_name.size(), std::string(loop_save_dir_name).append("/").append(json_file_name)));
            filename_vec.push_back(std::string(loop_save_dir_name).append("/").append(json_file_name)) ;
            float sim_plane_res_cv = 0, sim_plane_res = 0;
            float sim_line_res_cv = 0, sim_line_res = 0;
            float sim_plane_res_roi = 0, sim_line_res_roi = 0;
            float non_zero_ratio_plane = 0;

            for (size_t his = 0; his < pt_map_vec.size(); his++)
            {
                float ratio_non_zero_plane_his = pt_map_vec[his]->ratio_nonzero_plane;
                float ratio_non_zero_line_his = pt_map_vec[his]->ratio_nonzero_line;

                if (ratio_non_zero_plane_his < avail_ratio_plane && ratio_non_zero_line_his < avail_ratio_line)
                    continue;

                if (abs(pt_map_vec[his]->roi_range - pt_cell_map_temp->roi_range) > 5.0)
                    continue;

                sim_plane_res = pt_cell_map_temp->max_similiarity_of_two_image(pt_cell_map_temp->feature_img_plane,
                                                                               pt_map_vec[his]->feature_img_plane);
                sim_line_res = pt_cell_map_temp->max_similiarity_of_two_image(pt_cell_map_temp->feature_img_line,
                                                                              pt_map_vec[his]->feature_img_line);

                if ((pt_map_vec.size() - his > 200) && ((sim_line_res > 0.80 && sim_plane_res > 0.90) || (sim_plane_res > 0.95)))
                {
                    sim_plane_res_roi = pt_cell_map_temp->max_similiarity_of_two_image(pt_cell_map_temp->feature_img_plane_roi,
                                                                                       pt_map_vec[his]->feature_img_plane_roi);
                    sim_line_res_roi = pt_cell_map_temp->max_similiarity_of_two_image(pt_cell_map_temp->feature_img_line_roi,
                                                                                      pt_map_vec[his]->feature_img_line_roi);

                    if ((sim_line_res_roi > 0.80 && sim_plane_res_roi > 0.90) || (sim_plane_res_roi > 0.95))
                    {
                        printf("Inlier loop detection wait ICP to find the transfrom\r\n");
                        fprintf(fp, "Inlier loop detection wait ICP to find the transfrom\r\n");
                        fflush(fp);
                    }
                    else
                        continue;

                    printf("----------------------------\r\n");
                    printf("%s -- %s\r\n", filename_vec[pt_map_vec.size() - 1].c_str(), filename_vec[his].c_str());
                    printf("Nonzero_ratio %.3f , %.3f, %.3f, %.3f \r\n ", ratio_non_zero_plane, ratio_non_zero_line, ratio_non_zero_plane_his, ratio_non_zero_line_his);
                    printf("Similarity = %.3f , %.3f, %.3f, %.3f\r\n", sim_line_res, sim_plane_res, sim_line_res_cv, sim_plane_res_cv);
                    printf(" Roi similarity [%.2f, %.2f] = %.3f , %.3f \r\n", pt_cell_map_temp->roi_range, pt_map_vec[his]->roi_range, sim_line_res_roi, sim_plane_res_roi);

                    fprintf(fp, "----------------------------\r\n");
                    fprintf(fp, "%s -- %s\r\n", filename_vec[pt_map_vec.size() - 1].c_str(), filename_vec[his].c_str());
                    fprintf(fp, "Nonzero_ratio %.3f , %.3f, %.3f, %.3f \r\n ", ratio_non_zero_plane, ratio_non_zero_line, ratio_non_zero_plane_his, ratio_non_zero_line_his);
                    fprintf(fp, "Similarity = %.3f , %.3f, %.3f, %.3f\r\n", sim_line_res, sim_plane_res, sim_line_res_cv, sim_plane_res_cv);
                    fprintf(fp, "Roi similarity [%.2f, %.2f] = %.3f , %.3f \r\n", pt_cell_map_temp->roi_range, pt_map_vec[his]->roi_range, sim_line_res_roi, sim_plane_res_roi);
                    fflush(fp);

                    PointCloudMap<float> *pt_cell_map_his = new PointCloudMap<float>();
                    pt_cell_map_his->set_resolution(1.0);
                    pt_cell_map_his->set_point_cloud(pt_cell_map_his->load_pts_from_file(filename_vec[his]));
                    pt_cell_map_his->analyze_mapping(1);
                    scene_align.set_downsample_resolution(0.1, 0.1);
                    double ICP_SCORE = scene_align.find_tranfrom_of_two_mappings(pt_cell_map_his, pt_cell_map_temp, 1);
                    pt_cell_map_his->clear_data();
                    delete pt_cell_map_his;
                    if (scene_align.pc_reg.inlier_final_threshold > 1.0)
                        his += 10;

                    printf("ICP inlier threshold = %lf, %lf\r\n", ICP_SCORE, scene_align.pc_reg.inlier_final_threshold);
                    printf("%s\r\n", scene_align.pc_reg.m_final_opt_summary.BriefReport().c_str());
                    fprintf(fp, "ICP inlier threshold = %lf, %lf\r\n", ICP_SCORE, scene_align.pc_reg.inlier_final_threshold);
                    fprintf(fp, "%s\r\n", scene_align.pc_reg.m_final_opt_summary.BriefReport().c_str());
                    
                    if (scene_align.pc_reg.inlier_final_threshold < 0.35)
                    {
                        printf("I believe this is true loop.\r\n");
                        fprintf(fp, "I believe this is true loop.\r\n");
                        auto Q_a = pose3d_vec[his].q;
                        auto Q_b = pose3d_vec[pose3d_vec.size() - 1].q;
                        auto T_a = pose3d_vec[his].p;
                        auto T_b = pose3d_vec[pose3d_vec.size() - 1].p;
                        auto ICP_q = scene_align.pc_reg.q_w_curr;
                        auto ICP_t = scene_align.pc_reg.t_w_curr;
                        for (size_t i = 0; i < 10; i++)
                        {
                            std::cout << "-------------------------------------" << std::endl;
                            std::cout << ICP_q.coeffs().transpose() << std::endl;
                            std::cout << ICP_t.transpose() << std::endl;
                        }
                        constrain_vec.push_back(SceneAlignment<float>::add_constrain_of_loop(pose3d_vec.size() - 1, his, Q_a, T_a, Q_b, T_b, ICP_q, ICP_t));
                        std::string path_name = loop_save_dir_name;
                        std::string g2o_filename = std::string(path_name).append("/loop.g2o");
                        pose3d_map_ori = pose3d_map;
                        auto temp_pose_3d_map = pose3d_map;
                        SceneAlignment<float>::save_edge_and_vertex_to_g2o(g2o_filename.c_str(), temp_pose_3d_map, constrain_vec);
                        Ceres_pose_graph_3d::pose_graph_optimization(temp_pose_3d_map, constrain_vec);
                        Ceres_pose_graph_3d::OutputPoses(std::string(path_name).append("/poses_ori.txt"), pose3d_map_ori);
                        Ceres_pose_graph_3d::OutputPoses(std::string(path_name).append("/poses_opm.txt"), temp_pose_3d_map);

                        scene_align.dump_file_name(std::string(path_name).append("/file_name.txt"), map_file_name);
                        map_rfn.refine_mapping(map_id_pc, pose3d_map_ori, temp_pose_3d_map, 1);
                        pcl::toROSMsg(map_rfn.m_pts_aft_refind, ros_laser_cloud_surround);
                        ros_laser_cloud_surround.header.stamp = ros::Time::now();
                        ros_laser_cloud_surround.header.frame_id = "/camera_init";
                        pub_pc_aft_loop.publish(ros_laser_cloud_surround);
                        if_end = 1;
                        // TODO, add constrain.
                    }
                    else
                        his += 5;
                    
                    if (if_end)
                        break;
                }
                if (if_end)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                    break;
                }
            }
            scene_align.dump_file_name(std::string(loop_save_dir_name).append("/file_name.txt"), map_file_name);
            pt_cell_map_temp->clear_data();
            fflush(fp);

            pcl::toROSMsg(pt_full, ros_laser_cloud_surround);
            ros_laser_cloud_surround.header.stamp = ros::Time::now();
            ros_laser_cloud_surround.header.frame_id = "/camera_init";
            pub_debug_pts.publish(ros_laser_cloud_surround);

            if (if_end)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                break;
            }
        }
    }

    void service_pub_surround_pts()
    {
        pcl::VoxelGrid<PointType> down_sample_filter_surface;
        down_sample_filter_surface.setLeafSize(surround_pointcloud_resolution, surround_pointcloud_resolution, surround_pointcloud_resolution);
        pcl::PointCloud<PointType> pc_temp;
        sensor_msgs::PointCloud2 ros_laser_cloud_surround;
        std::this_thread::sleep_for(std::chrono::nanoseconds(10));
        pcl::PointCloud<PointType>::Ptr laser_cloud_surround(new pcl::PointCloud<PointType>());
        laser_cloud_surround->reserve(1e8);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        int last_update_index = 0;
        while (1)
        {
            while (current_frame_index - last_update_index < 100)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            last_update_index = current_frame_index;
            pcl::PointCloud<PointType> pc_temp;
            laser_cloud_surround->clear();
            if (pt_cell_map_full.get_cells_size() == 0)
                continue;
            std::vector<PointCloudMap<float>::PC_CELL *> cell_vec = pt_cell_map_full.find_cells_in_radius(t_w_curr, 1000.0);
            for (size_t i = 0; i < cell_vec.size(); i++)
            {
                if (down_sample_replace)
                {
                    down_sample_filter_surface.setInputCloud(cell_vec[i]->get_pointcloud().makeShared());
                    down_sample_filter_surface.filter(pc_temp);
                    cell_vec[i]->set_pointcloud(pc_temp);
                    *laser_cloud_surround += pc_temp;
                }
                else
                    *laser_cloud_surround += cell_vec[i]->get_pointcloud();
            }
            if (laser_cloud_surround->points.size())
            {
                down_sample_filter_surface.setInputCloud(laser_cloud_surround);
                down_sample_filter_surface.filter(*laser_cloud_surround);
                pcl::toROSMsg(*laser_cloud_surround, ros_laser_cloud_surround);
                ros_laser_cloud_surround.header.stamp = ros::Time::now();
                ros_laser_cloud_surround.header.frame_id = "/camera_init";
                pub_laser_cloud_surround.publish(ros_laser_cloud_surround);
            }
        }
    }

    Eigen::Matrix<double, 3, 1> pcl_pt_to_eigend(PointType &pt)
    {
        return Eigen::Matrix<double, 3, 1>(pt.x, pt.y, pt.z);
    }

    //receive odomtry
    void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
    {
        mutex_buf.lock();
        m_odom_que.push(laserOdometry);
        mutex_buf.unlock();

        // high frequence publish
        Eigen::Quaterniond q_wodom_curr;
        Eigen::Vector3d t_wodom_curr;
        q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
        q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
        q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
        q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
        t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
        t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
        t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

        Eigen::Quaterniond q_w_curr = Eigen::Quaterniond(1, 0, 0, 0);
        Eigen::Vector3d t_w_curr = Eigen::Vector3d::Zero();

        nav_msgs::Odometry odomAftMapped;
        odomAftMapped.header.frame_id = "/camera_init";
        odomAftMapped.child_frame_id = "/aft_mapped";
        odomAftMapped.header.stamp = laserOdometry->header.stamp;
        odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
        odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
        odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
        odomAftMapped.pose.pose.orientation.w = q_w_curr.w();
        odomAftMapped.pose.pose.position.x = t_w_curr.x();
        odomAftMapped.pose.pose.position.y = t_w_curr.y();
        odomAftMapped.pose.pose.position.z = t_w_curr.z();
        pub_odom_aft_mapped_hight_frec.publish(odomAftMapped);
    }

    void find_min_max_intensity(const pcl::PointCloud<PointType>::Ptr pc_ptr, float &min_I, float &max_I)
    {
        size_t pt_size = pc_ptr->size();
        min_I = 10000;
        max_I = -min_I;
        for (size_t i = 0; i < pt_size; i++)
        {
            min_I = std::min(pc_ptr->points[i].intensity, min_I);
            max_I = std::max(pc_ptr->points[i].intensity, max_I);
        }
    }

    float refine_blur(float in_blur, const float &min_blur, const float &max_blur)
    {
        return (in_blur - min_blur) / (max_blur - min_blur);
    }

    float compute_fov_angle(const PointType &pt)
    {
        float sq_xy = sqrt(std::pow(pt.y / pt.x, 2) + std::pow(pt.z / pt.x, 2));
        return atan(sq_xy) * 57.3;
    }

    void init_pointcloud_registration(PointCloudRegistration &pc_reg)
    {
        pc_reg.logger_common = &logger_common;
        pc_reg.logger_pcd = &logger_pcd;
        pc_reg.m_logger_timer = &m_logger_timer;
        pc_reg.m_timer = &m_timer;
        pc_reg.if_motion_deblur = MOTION_DEBLUR; // 0
        pc_reg.current_frame_index = current_frame_index;
        pc_reg.mapping_init_accumulate_frames = m_mapping_init_accumulate_frames; // 100

        pc_reg.last_time_stamp = last_time_stamp;
        pc_reg.para_max_angular_rate = m_para_max_angular_rate;
        pc_reg.para_max_speed = m_para_max_speed;
        pc_reg.m_max_final_cost = m_max_final_cost;
        pc_reg.para_icp_max_iterations = m_para_icp_max_iterations;
        pc_reg.para_cere_max_iterations = m_para_cere_max_iterations;
        pc_reg.minimum_pt_time_stamp = minimum_pt_time_stamp;
        pc_reg.maximum_pt_time_stamp = maximum_pt_time_stamp;
        pc_reg.m_minimum_icp_R_diff = m_minimum_icp_R_diff;
        pc_reg.m_minimum_icp_T_diff = m_minimum_icp_T_diff;

        pc_reg.q_w_last = q_w_curr; // (0, 0, 0, 1)
        pc_reg.t_w_last = t_w_curr; // (0, 0, 0)

        pc_reg.q_w_curr = q_w_curr; // (0, 0, 0, 1)
        pc_reg.t_w_curr = t_w_curr; // (0, 0, 0)
    }

    int if_matchbuff_and_pc_sync(float point_cloud_current_timestamp)
    {
        if (latest_pc_matching_refresh_time < 0)
            return 1;
        if (point_cloud_current_timestamp - latest_pc_matching_refresh_time < m_maximum_pointcloud_delay_time)
            return 1;
        if (last_pc_reg_time == latest_pc_matching_refresh_time)  // All is processed
            return 1;
        printf("****** Current pc timestamp = %.3f, lastest buff timestamp = %.3f, lastest_pc_reg_time = %.3f ******\r\n",
                point_cloud_current_timestamp,
                latest_pc_matching_refresh_time,
                last_pc_reg_time);

        return 0;
    }

    int process_new_scan()
    {
        m_timer.tic("Frame process");
        m_timer.tic("Query points for match");
        
        pcl::PointCloud<PointType> current_pc_full, current_pc_corner_latest, current_pc_surf_latest;
        pcl::VoxelGrid<PointType> down_sample_filter_corner = down_sample_filter_corner_;
        pcl::VoxelGrid<PointType> down_sample_filter_surface = down_sample_filter_surface_;
        pcl::KdTreeFLANN<PointType> kdtree_corner_from_map;
        pcl::KdTreeFLANN<PointType> kdtree_surf_from_map;

        mutex_querypointcloud.lock();
        current_pc_full = *pc_full;
        current_pc_corner_latest = *pc_corner_latest;
        current_pc_surf_latest = *pc_surf_latest;

        float min_t, max_t;
        find_min_max_intensity(current_pc_full.makeShared(), min_t, max_t);

        double point_cloud_current_timestamp = min_t;
        if (point_cloud_current_timestamp > last_pc_income_time)
            last_pc_income_time = point_cloud_current_timestamp;
        point_cloud_current_timestamp = last_pc_income_time;

        time_odom = last_time_stamp;
        minimum_pt_time_stamp = last_time_stamp;
        maximum_pt_time_stamp = max_t;
        last_time_stamp = max_t;
        PointCloudRegistration pc_reg;
        init_pointcloud_registration(pc_reg);
        current_frame_index++;
        double time_odom = ros::Time::now().toSec();
        mutex_querypointcloud.unlock();

        m_timer.tic("Wait sync");
        while(!if_matchbuff_and_pc_sync(point_cloud_current_timestamp))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        *(m_logger_timer.get_ostream()) << m_timer.toc_string("Wait sync") << std::endl;

        pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>()); // filtered
        pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());

        if (if_input_downsample_mode) // 1
        {
            down_sample_filter_corner.setInputCloud(current_pc_corner_latest.makeShared());
            down_sample_filter_corner.filter(*laserCloudCornerStack);
            down_sample_filter_surface.setInputCloud(current_pc_surf_latest.makeShared());
            down_sample_filter_surface.filter(*laserCloudSurfStack);
        }
        else
        {
            *laserCloudCornerStack = current_pc_corner_latest;
            *laserCloudSurfStack = current_pc_surf_latest;
        }

        size_t laser_corner_pt_num = laserCloudCornerStack->points.size();
        size_t laser_surface_pt_num = laserCloudSurfStack->points.size();

        if (m_if_save_to_pcd_files && PCD_SAVE_RAW)
            m_pcl_tools_raw.save_to_pcd_files("raw", current_pc_full, current_frame_index);

        q_w_last = q_w_curr;
        t_w_last = t_w_curr;

        pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map(new pcl::PointCloud<PointType>());

        mutex_buff_for_matching_corner.lock();
        *laser_cloud_corner_from_map = *laser_cloud_corner_from_map_latest;
        kdtree_corner_from_map = kdtree_corner_from_map_last;
        mutex_buff_for_matching_surface.unlock();

        mutex_buff_for_matching_surface.lock();
        *laser_cloud_surf_from_map = *laser_cloud_surf_from_map_latest;
        kdtree_surf_from_map = kdtree_surf_from_map_last;
        mutex_buff_for_matching_corner.unlock();

        int reg_res = pc_reg.find_out_incremental_transfrom(laser_cloud_corner_from_map, laser_cloud_surf_from_map,
                                                            kdtree_corner_from_map, kdtree_surf_from_map,
                                                            laserCloudCornerStack, laserCloudSurfStack);

        if (reg_res == 0)
            return 0;

        m_timer.tic("Add new frame");

        PointType pointSel;
        
        pcl::PointCloud<PointType>::Ptr pc_new_feature_corners(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr pc_new_feature_surface(new pcl::PointCloud<PointType>());
        for (size_t i = 0; i < laser_corner_pt_num; i++)
        {
            pc_reg.pointAssociateToMap(&laserCloudCornerStack->points[i],
                                       &pointSel,
                                       refine_blur(laserCloudCornerStack->points[i].intensity, minimum_pt_time_stamp, maximum_pt_time_stamp),
                                       g_if_undistore); // 0
            pc_new_feature_corners->push_back(pointSel);
        }

        for (size_t i = 0; i < laser_surface_pt_num; i++)
        {
            pc_reg.pointAssociateToMap(&laserCloudSurfStack->points[i],
                                       &pointSel,
                                       refine_blur(laserCloudSurfStack->points[i].intensity, minimum_pt_time_stamp, maximum_pt_time_stamp),
                                       g_if_undistore);
            pc_new_feature_surface->push_back(pointSel);
        }

        down_sample_filter_corner.setInputCloud(pc_new_feature_corners);
        down_sample_filter_corner.filter(*pc_new_feature_corners);
        down_sample_filter_surface.setInputCloud(pc_new_feature_surface);
        down_sample_filter_surface.filter(*pc_new_feature_surface);

        double r_diff = q_w_curr.angularDistance(last_his_add_q) * 57.3;
        double t_diff = (t_w_curr - last_his_add_t).norm();

        pc_reg.pointcloudAssociateToMap(current_pc_full, current_pc_full, g_if_undistore);

        mutex_mapping.lock();

        if (pc_corner_history.size() < (size_t)maximum_history_size ||
            t_diff > history_add_t_step ||
            r_diff > history_add_angle_step * 57.3)
        {
            last_his_add_q = q_w_curr;
            last_his_add_t = t_w_curr;

            pc_corner_history.push_back(*pc_new_feature_corners);
            pc_surface_history.push_back(*pc_new_feature_surface);
            mutex_dump_full_history.lock();
            pc_full_history.push_back(current_pc_full);
            his_reg_error.push_back(pc_reg.inlier_final_threshold);
            mutex_dump_full_history.unlock();
        }

        if (pc_corner_history.size() > (size_t)maximum_history_size)
        {
            (pc_corner_history.front()).clear();
            pc_corner_history.pop_front();
        }

        if (pc_surface_history.size() > (size_t)maximum_history_size)
        {
            (pc_surface_history.front()).clear();
            pc_surface_history.pop_front();
        }

        if (pc_full_history.size() > (size_t)maximum_history_size)
        {
            mutex_dump_full_history.lock();
            (pc_full_history.front()).clear();
            pc_full_history.pop_front();
            his_reg_error.pop_front();
            mutex_dump_full_history.unlock();
        }

        if_mapping_updated_corner = true;
        if_mapping_updated_surface = true;

        pt_cell_map_corners.append_cloud(PCL_TOOLS::pcl_pts_to_eigen_pts<float, PointType>(pc_new_feature_corners));
        pt_cell_map_planes.append_cloud(PCL_TOOLS::pcl_pts_to_eigen_pts<float, PointType>(pc_new_feature_surface));

        *(logger_common.get_ostream()) << "New added regtime "<< point_cloud_current_timestamp << endl;

        if (last_pc_reg_time < point_cloud_current_timestamp || point_cloud_current_timestamp < 10.0)
        {
            q_w_curr = pc_reg.q_w_curr;
            t_w_curr = pc_reg.t_w_curr;
            last_pc_reg_time = point_cloud_current_timestamp;
        }
        else
            *(logger_common.get_ostream()) << "***** older update, reject update pose *****" << endl;

        *(logger_pcd.get_ostream()) << "--------------------" << endl;
        logger_pcd.printf("Curr_Q = %f,%f,%f,%f\r\n", q_w_curr.w(), q_w_curr.x(), q_w_curr.y(), q_w_curr.z());
        logger_pcd.printf("Curr_T = %f,%f,%f\r\n", t_w_curr(0), t_w_curr(1), t_w_curr(2));
        logger_pcd.printf("Incre_Q = %f,%f,%f,%f\r\n", pc_reg.q_w_incre.w(), pc_reg.q_w_incre.x(), pc_reg.q_w_incre.y(), pc_reg.q_w_incre.z());
        logger_pcd.printf("Incre_T = %f,%f,%f\r\n", pc_reg.t_w_incre(0), pc_reg.t_w_incre(1), pc_reg.t_w_incre(2));
        logger_pcd.printf("Cost=%f,blk_size = %d \r\n", m_final_opt_summary.final_cost, m_final_opt_summary.num_residual_blocks);
        *(logger_pcd.get_ostream()) << m_final_opt_summary.BriefReport() << endl;

        mutex_mapping.unlock();

        if (thread_match_buff_refresh.size() < (size_t)maximum_mapping_buff_thread)
        {
            std::future<void> *mapping_refresh_service = 
                new std::future<void>(std::async(std::launch::async, &LaserMapping::service_update_buff_for_matching, this));
            thread_match_buff_refresh.push_back(mapping_refresh_service);
        }

        pt_cell_map_full.append_cloud(PCL_TOOLS::pcl_pts_to_eigen_pts<float, PointType>(current_pc_full.makeShared()));

        m_mutex_ros_pub.lock();
        sensor_msgs::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(current_pc_full, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = ros::Time().fromSec(time_odom);
        laserCloudFullRes3.header.frame_id = "/camera_init";
        pub_pc_full.publish(laserCloudFullRes3); //single_frame_with_pose_tranfromed

        //publish surround map for every 5 frame
        if (PUB_DEBUG_INFO) // 0
        {
            pcl::PointCloud<PointType> pc_feature_pub_corners, pc_feature_pub_surface;
            sensor_msgs::PointCloud2   laserCloudMsg;

            pc_reg.pointcloudAssociateToMap(current_pc_surf_latest, pc_feature_pub_surface, g_if_undistore);
            pcl::toROSMsg(pc_feature_pub_surface, laserCloudMsg);
            laserCloudMsg.header.stamp = ros::Time().fromSec(time_odom);
            laserCloudMsg.header.frame_id = "/camera_init";
            pub_last_surface_pts.publish(laserCloudMsg);
            pc_reg.pointcloudAssociateToMap(current_pc_corner_latest, pc_feature_pub_corners, g_if_undistore);
            pcl::toROSMsg(pc_feature_pub_corners, laserCloudMsg);
            laserCloudMsg.header.stamp = ros::Time().fromSec(time_odom);
            laserCloudMsg.header.frame_id = "/camera_init";
            pub_last_corner_pts.publish(laserCloudMsg);
        }

        sensor_msgs::PointCloud2 laserCloudMsg;
        pcl::toROSMsg(*laser_cloud_surf_from_map, laserCloudMsg);
        laserCloudMsg.header.stamp = ros::Time().fromSec(time_odom);
        laserCloudMsg.header.frame_id = "/camera_init";
        pub_match_surface_pts.publish(laserCloudMsg);

        pcl::toROSMsg(*laser_cloud_corner_from_map, laserCloudMsg);
        laserCloudMsg.header.stamp = ros::Time().fromSec(time_odom);
        laserCloudMsg.header.frame_id = "/camera_init";
        pub_match_corner_pts.publish(laserCloudMsg);

        if (m_if_save_to_pcd_files)
        {
            m_pcl_tools_aftmap.save_to_pcd_files("aft_mapp", current_pc_full, current_frame_index);
            *(logger_pcd.get_ostream()) << "Save to: " << m_pcl_tools_aftmap.m_save_file_name << endl;
        }
        
        nav_msgs::Odometry odomAftMapped;
        odomAftMapped.header.frame_id = "/camera_init";
        odomAftMapped.child_frame_id = "/aft_mapped";
        odomAftMapped.header.stamp = ros::Time().fromSec(time_odom);

        odomAftMapped.pose.pose.orientation.x = q_w_curr.x();
        odomAftMapped.pose.pose.orientation.y = q_w_curr.y();
        odomAftMapped.pose.pose.orientation.z = q_w_curr.z();
        odomAftMapped.pose.pose.orientation.w = q_w_curr.w();

        odomAftMapped.pose.pose.position.x = t_w_curr.x();
        odomAftMapped.pose.pose.position.y = t_w_curr.y();
        odomAftMapped.pose.pose.position.z = t_w_curr.z();

        pub_odom_aft_mapped.publish(odomAftMapped); // name: Odometry aft_mapped_to_init

        geometry_msgs::PoseStamped pose_aft_mapped;
        pose_aft_mapped.header = odomAftMapped.header;
        pose_aft_mapped.pose = odomAftMapped.pose.pose;
        laser_after_mapped_path.header.stamp = odomAftMapped.header.stamp;
        laser_after_mapped_path.header.frame_id = "/camera_init";
        laser_after_mapped_path.poses.push_back(pose_aft_mapped);
        pub_laser_aft_mapped_path.publish(laser_after_mapped_path);

        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        transform.setOrigin(tf::Vector3(t_w_curr(0), t_w_curr(1), t_w_curr(2)));

        q.setW(q_w_curr.w());
        q.setX(q_w_curr.x());
        q.setY(q_w_curr.y());
        q.setZ(q_w_curr.z());
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped"));

        m_mutex_ros_pub.unlock();
        *(m_logger_timer.get_ostream()) << m_timer.toc_string("Add new frame") << std::endl;
        *(m_logger_timer.get_ostream()) << m_timer.toc_string("Frame process") << std::endl;
        
        return 1;
    }

    void process()
    {
        double first_time_stamp = -1;
        m_last_max_blur = 0.0;

        service_pub_surround_pts_ = new std::future<void>(std::async(std::launch::async, &LaserMapping::service_pub_surround_pts, this));
        if (if_loop_closure)
            service_loop_detection_ = new std::future<void>(std::async(std::launch::async, &LaserMapping::service_loop_detection, this));

        while (1)
        {            
            logger_common.printf("------------------\r\n");
            m_logger_timer.printf("------------------\r\n");
            
            while (queue_avail_data.empty())
                sleep(0.0001);

            mutex_buf.lock();
            while (queue_avail_data.size() >= (size_t)max_buffer_size)
            {
                ROS_WARN("Drop lidar frame in mapping for real time performance !!!");
                (*logger_common.get_ostream()) << "Drop lidar frame in mapping for real time performance !!!" << endl;
                queue_avail_data.pop();
            }
            DataPair *current_data_pair = queue_avail_data.front();
            queue_avail_data.pop();
            mutex_buf.unlock();

            m_timer.tic("Prepare to enter thread");

            time_pc_corner_past = current_data_pair->pc_corner->header.stamp.toSec();

            if (first_time_stamp < 0)
                first_time_stamp = time_pc_corner_past;

            *logger_common.get_ostream() << "Messgage time stamp = " << time_pc_corner_past - first_time_stamp << endl;

            mutex_querypointcloud.lock();
            pc_corner_latest->clear();
            pcl::fromROSMsg(*current_data_pair->pc_corner, *pc_corner_latest);
            pc_surf_latest->clear();
            pcl::fromROSMsg(*current_data_pair->pc_plane, *pc_surf_latest);
            pc_full->clear();
            pcl::fromROSMsg(*current_data_pair->pc_full, *pc_full);
            mutex_querypointcloud.unlock();
            
            delete current_data_pair;

            COMMON_TOOLS::maintain_maximum_thread_pool<std::future<int>*>(thread_pool, maximum_parallel_thread);

            std::future<int> *thd = new std::future<int>(std::async(std::launch::async, &LaserMapping::process_new_scan, this));

            *m_logger_timer.get_ostream()<< m_timer.toc_string("Prepare to enter thread") << std::endl;

            thread_pool.push_back(thd);

            std::this_thread::sleep_for(std::chrono::nanoseconds(10));
        }
    }
};

#endif // LASER_MAPPING_HPP
