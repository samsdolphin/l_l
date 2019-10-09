#ifndef LASER_FEATURE_EXTRACTION_H
#define LASER_FEATURE_EXTRACTION_H

#include <cmath>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <string>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>
#include <vector>

#include "livox_feature_extractor.hpp"
#include "tools/common.h"
#include "tools/logger.hpp"

using std::atan2;
using std::cos;
using std::sin;
using namespace COMMON_TOOLS;

class LaserFeature
{
public:
    const double m_para_scanPeriod = 0.1;

    int if_pub_debug_feature = 1;

    const int m_para_system_delay = 20;
    int         m_para_system_init_count = 0;
    bool        m_para_systemInited = false;
    float       m_pc_curvature[400000];
    int         m_pc_sort_idx[400000];
    int         m_pc_neighbor_picked[400000];
    int         m_pc_cloud_label[400000];
    int m_if_motion_deblur = 0;
    int m_odom_mode = 0; //0 = for odom, 1 = for mapping
    float       plane_resolution;
    float       line_resolution;
    File_logger m_file_logger;

    bool        m_if_pub_each_line = false;
    int         m_lidar_type = 0; // 0 is velodyne, 1 is livox
    int laser_split_number = 64;
    LivoxLaser livox_laser;
    ros::Time   m_init_timestamp;

    bool comp(int i, int j)
    {
        return (m_pc_curvature[i] < m_pc_curvature[j]);
    }

    ros::Publisher pub_pc_corners;
    ros::Publisher pub_pc_surface;
    ros::Publisher pub_pc_full;
    ros::Publisher pub_laser_pc;
    ros::Publisher pub_pc_sharp_corner;
    ros::Publisher pub_pc_less_sharp_corner;
    ros::Publisher pub_pc_surface_flat;
    ros::Publisher pub_pc_surface_less_flat;
    ros::Publisher pub_pc_removed_pt;
    std::vector<ros::Publisher> pub_each_scan;

    ros::Subscriber sub_input_laser_cloud;

    double MINIMUM_RANGE = 0.01;

    sensor_msgs::PointCloud2 temp_out_msg;
    pcl::VoxelGrid<PointType> voxel_filter_for_surface;
    pcl::VoxelGrid<PointType> voxel_filter_for_corner;

    int init_ros_env()
    {
        ros::NodeHandle nh;
        m_init_timestamp = ros::Time::now();
        init_livox_lidar_para();

        nh.param<int>("scan_line", laser_split_number, 16);
        nh.param<float>("mapping_plane_resolution", plane_resolution, 0.8);
        nh.param<float>("mapping_line_resolution", line_resolution, 0.8);
        nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);
        nh.param<int>("if_motion_deblur", m_if_motion_deblur, 1);
        nh.param<int>("odom_mode", m_odom_mode, 0);

        double livox_corners, livox_surface, minimum_view_angle;
        nh.param<double>("corner_curvature", livox_corners, 0.05);
        nh.param<double>("surface_curvature", livox_surface, 0.01);
        nh.param<double>("minimum_view_angle", minimum_view_angle, 10);
        livox_laser.thr_corner_curvature = livox_corners;
        livox_laser.thr_surface_curvature = livox_surface;
        livox_laser.minimum_view_angle = minimum_view_angle;

        printf("scan line number %d \n", laser_split_number);

        if (laser_split_number != 16 && laser_split_number != 64)
        {
            printf("only support velodyne with 16 or 64 scan line!");
            return 0;
        }

        string log_save_dir_name;
        nh.param<std::string>("log_save_dir", log_save_dir_name, "../");
        m_file_logger.set_log_dir(log_save_dir_name);
        m_file_logger.init("scanRegistration.log");

        sub_input_laser_cloud = nh.subscribe<sensor_msgs::PointCloud2>("/laser_points", 10000, &LaserFeature::laserCloudHandler, this);

        pub_laser_pc = nh.advertise<sensor_msgs::PointCloud2>("/laser_points_2", 10000);
        pub_pc_sharp_corner = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp", 10000);
        pub_pc_less_sharp_corner = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp", 10000);
        pub_pc_surface_flat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat", 10000);
        pub_pc_surface_less_flat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat", 10000);
        pub_pc_removed_pt = nh.advertise<sensor_msgs::PointCloud2>("/laser_remove_points", 10000);

        pub_pc_corners = nh.advertise<sensor_msgs::PointCloud2>("/pc2_corners", 10000);
        pub_pc_surface = nh.advertise<sensor_msgs::PointCloud2>("/pc2_surface", 10000);
        pub_pc_full = nh.advertise<sensor_msgs::PointCloud2>("/pc2_full", 10000);

        voxel_filter_for_surface.setLeafSize(plane_resolution / 2, plane_resolution / 2, plane_resolution / 2);
        voxel_filter_for_corner.setLeafSize(line_resolution, line_resolution, line_resolution);

        if (m_if_pub_each_line)
            for (int i = 0; i < laser_split_number; i++)
            {
                ros::Publisher tmp = nh.advertise<sensor_msgs::PointCloud2>("/laser_scanid_" + std::to_string(i), 100);
                pub_each_scan.push_back(tmp);
            }

        return 0;
    }

    ~LaserFeature(){};

    LaserFeature()
    {
        init_ros_env();
    };

    template <typename PointT>
    void removeClosedPointCloud(const pcl::PointCloud<PointT> &cloud_in,
                                pcl::PointCloud<PointT> &cloud_out, float thres)
    {
        if (&cloud_in != &cloud_out)
        {
            cloud_out.header = cloud_in.header;
            cloud_out.points.resize(cloud_in.points.size());
        }

        size_t j = 0;

        for (size_t i = 0; i < cloud_in.points.size(); ++i)
        {
            if (cloud_in.points[i].x * cloud_in.points[i].x + cloud_in.points[i].y * cloud_in.points[i].y + cloud_in.points[i].z * cloud_in.points[i].z < thres * thres)
                continue;

            cloud_out.points[j] = cloud_in.points[i];
            j++;
        }

        if (j != cloud_in.points.size())
        {
            cloud_out.points.resize(j);
        }

        cloud_out.height = 1;
        cloud_out.width = static_cast<uint32_t>(j);
        cloud_out.is_dense = true;
    }

    void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudMsg)
    {
        std::vector<pcl::PointCloud<PointType>> laserCloudScans(laser_split_number);

        if (!m_para_systemInited)
        {
            m_para_system_init_count++;

            if (m_para_system_init_count >= m_para_system_delay)
                m_para_systemInited = true;
            else
                return;
        }

        std::vector<int> scanStartInd(1000, 0);
        std::vector<int> scanEndInd(1000, 0);

        pcl::PointCloud<pcl::PointXYZI> laserCloudIn;
        pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);
        size_t raw_pts_num = laserCloudIn.size();

        m_file_logger.printf(" Time: %.5f, num_raw: %d, num_filted: %d\r\n", laserCloudMsg->header.stamp.toSec(), raw_pts_num, laserCloudIn.size());

        laserCloudScans = livox_laser.extract_laser_features(laserCloudIn, laserCloudMsg->header.stamp.toSec()); // intensity = idx / pts_size

        if (laserCloudScans.size() <= 5) // less than 5 split
            return;

        laser_split_number = laserCloudScans.size();
        scanStartInd.resize(laser_split_number);
        scanEndInd.resize(laser_split_number);
        std::fill(scanStartInd.begin(), scanStartInd.end(), 0);
        std::fill(scanEndInd.begin(), scanEndInd.end(), 0);

        if (if_pub_debug_feature)
        {
            int piece_wise = 3;
            if(m_if_motion_deblur) // 0
                piece_wise = 3;

            vector<float> piece_wise_start(piece_wise);
            vector<float> piece_wise_end(piece_wise);

            for (int i = 0; i < piece_wise; i++)
            {
                int start_scans, end_scans;

                start_scans = int(laser_split_number * i / piece_wise);
                end_scans = int(laser_split_number * (i + 1) / piece_wise) - 1;

                int end_idx = laserCloudScans[end_scans].size() - 1;
                piece_wise_start[i] = ((float)livox_laser.find_pt_info(laserCloudScans[start_scans].points[0])->idx) / livox_laser.pts_info_vec.size();
                piece_wise_end[i] = ((float)livox_laser.find_pt_info(laserCloudScans[end_scans].points[end_idx])->idx) / livox_laser.pts_info_vec.size();
            }

            for (int i = 0; i < piece_wise; i++)
            {
                pcl::PointCloud<PointType>::Ptr livox_corners(new pcl::PointCloud<PointType>()),
                                                livox_surface(new pcl::PointCloud<PointType>()),
                                                livox_full(new pcl::PointCloud<PointType>());

                livox_laser.get_features(*livox_corners, *livox_surface, *livox_full, piece_wise_start[i], piece_wise_end[i]); // intensity = pt.time_stamp
                ros::Time current_time = ros::Time::now();

                pcl::toROSMsg(*livox_full, temp_out_msg);
                temp_out_msg.header.stamp = current_time;
                temp_out_msg.header.frame_id = "/camera_init";
                pub_pc_full.publish(temp_out_msg);

                voxel_filter_for_surface.setInputCloud(livox_surface);
                voxel_filter_for_surface.filter(*livox_surface);
                pcl::toROSMsg(*livox_surface, temp_out_msg);
                temp_out_msg.header.stamp = current_time;
                temp_out_msg.header.frame_id = "/camera_init";
                pub_pc_surface.publish(temp_out_msg);

                voxel_filter_for_corner.setInputCloud(livox_corners);
                voxel_filter_for_corner.filter(*livox_corners);
                pcl::toROSMsg(*livox_corners, temp_out_msg);
                temp_out_msg.header.stamp = current_time;
                temp_out_msg.header.frame_id = "/camera_init";
                pub_pc_corners.publish(temp_out_msg);

                if (m_odom_mode == 0) // odometry mode
                    break;
            }
        }
    }

    void init_livox_lidar_para()
    {
        std::string lidar_tpye_name;
        std::cout << "~~~~~ Init livox lidar parameters ~~~~~" << endl;
        if (ros::param::get("lidar_type", lidar_tpye_name))
        {
            printf("***** I get lidar_type declaration, lidar_type_name = %s ***** \r\n", lidar_tpye_name.c_str());

            if (lidar_tpye_name.compare("livox") == 0)
            {
                m_lidar_type = 1;
                std::cout << "Set lidar type = livox" << std::endl;
            }
            else
            {
                std::cout << "Set lidar type = velodyne" << std::endl;
                m_lidar_type = 0;
            }
        }
        else
        {
            printf("***** No lidar_type declaration ***** \r\n");
            m_lidar_type = 0;
            std::cout << "Set lidar type = velodyne" << std::endl;
        }

        if (ros::param::get("livox_min_dis", livox_laser.livox_min_allow_dis))
            std::cout << "Set livox lidar minimum distance= " << livox_laser.livox_min_allow_dis << std::endl;

        if (ros::param::get("livox_min_sigma", livox_laser.livox_min_sigma))
            std::cout << "Set livox lidar minimum sigama =  " << livox_laser.livox_min_sigma << std::endl;

        std::cout << "~~~~~ End ~~~~~" << endl;
    }
};

#endif // LASER_FEATURE_EXTRACTION_H
