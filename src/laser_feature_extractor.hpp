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

    int m_if_pub_debug_feature = 1;

    const int m_para_system_delay = 20;
    int         m_para_system_init_count = 0;
    bool        m_para_systemInited = false;
    float       m_pc_curvature[400000];
    int         m_pc_sort_idx[400000];
    int         m_pc_neighbor_picked[400000];
    int         m_pc_cloud_label[400000];
    int         m_if_motion_deblur = 0;
    int         m_odom_mode = 0; //0 = for odom, 1 = for mapping
    float       plane_resolution;
    float       line_resolution;
    File_logger m_file_logger;

    bool        m_if_pub_each_line = false;
    int         m_lidar_type = 0; // 0 is velodyne, 1 is livox
    int         laser_scan_number = 64;
    Livox_laser m_livox;
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
    pcl::VoxelGrid<PointType> m_voxel_filter_for_surface;
    pcl::VoxelGrid<PointType> m_voxel_filter_for_corner;

    int init_ros_env()
    {
        ros::NodeHandle nh;
        m_init_timestamp = ros::Time::now();
        init_livox_lidar_para();

        nh.param<int>("scan_line", laser_scan_number, 16);
        nh.param<float>("mapping_plane_resolution", plane_resolution, 0.8);
        nh.param<float>("mapping_line_resolution", line_resolution, 0.8);
        nh.param<double>("minimum_range", MINIMUM_RANGE, 0.1);
        nh.param<int>("if_motion_deblur", m_if_motion_deblur, 1);
        nh.param<int>("odom_mode", m_odom_mode, 0);

        double livox_corners, livox_surface, minimum_view_angle;
        nh.param<double>("corner_curvature", livox_corners, 0.05);
        nh.param<double>("surface_curvature", livox_surface, 0.01);
        nh.param<double>("minimum_view_angle", minimum_view_angle, 10);
        m_livox.thr_corner_curvature = livox_corners;
        m_livox.thr_surface_curvature = livox_surface;
        m_livox.minimum_view_angle = minimum_view_angle;

        printf("scan line number %d \n", laser_scan_number);

        if (laser_scan_number != 16 && laser_scan_number != 64)
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

        m_voxel_filter_for_surface.setLeafSize(plane_resolution / 2, plane_resolution / 2, plane_resolution / 2);
        m_voxel_filter_for_corner.setLeafSize(line_resolution, line_resolution, line_resolution);

        if (m_if_pub_each_line)
            for (int i = 0; i < laser_scan_number; i++)
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
        std::vector<pcl::PointCloud<PointType>> laserCloudScans(laser_scan_number);

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
        int raw_pts_num = laserCloudIn.size();

        m_file_logger.printf(" Time: %.5f, num_raw: %d, num_filted: %d\r\n", laserCloudMsg->header.stamp.toSec(), raw_pts_num, laserCloudIn.size());

        size_t cloudSize = laserCloudIn.points.size();

        if (m_lidar_type)
        {
            // LIVOX LIDAR
            laserCloudScans = m_livox.extract_laser_features(laserCloudIn, laserCloudMsg->header.stamp.toSec());

            if (laserCloudScans.size() <= 5) // less than 5 scan
                return;

            laser_scan_number = laserCloudScans.size() * 1.0;

            scanStartInd.resize(laser_scan_number);
            scanEndInd.resize(laser_scan_number);
            std::fill(scanStartInd.begin(), scanStartInd.end(), 0);
            std::fill(scanEndInd.begin(), scanEndInd.end(), 0);

            if (m_if_pub_debug_feature)
            {
                int piece_wise = 3;
                if(m_if_motion_deblur)
                    piece_wise = 3;

                vector<float> piece_wise_start(piece_wise);
                vector<float> piece_wise_end(piece_wise);

                for (int i = 0; i < piece_wise; i++)
                {
                    int start_scans, end_scans;

                    start_scans = int((laser_scan_number * (i)) / piece_wise);
                    end_scans = int((laser_scan_number * (i + 1)) / piece_wise) - 1;

                    int end_idx = laserCloudScans[end_scans].size() - 1;
                    piece_wise_start[i] = ((float)m_livox.find_pt_info(laserCloudScans[start_scans].points[0])->idx) / m_livox.m_pts_info_vec.size();
                    piece_wise_end[i] = ((float)m_livox.find_pt_info(laserCloudScans[end_scans].points[end_idx])->idx) / m_livox.m_pts_info_vec.size();
                }

                for (int i = 0; i < piece_wise; i++)
                {
                    pcl::PointCloud<PointType>::Ptr livox_corners(new pcl::PointCloud<PointType>()),
                                                    livox_surface(new pcl::PointCloud<PointType>()),
                                                    livox_full(new pcl::PointCloud<PointType>());

                    m_livox.get_features(*livox_corners, *livox_surface, *livox_full, piece_wise_start[i], piece_wise_end[i]);

                    ros::Time current_time = ros::Time::now();

                    pcl::toROSMsg(*livox_full, temp_out_msg);
                    temp_out_msg.header.stamp = current_time;
                    temp_out_msg.header.frame_id = "/camera_init";
                    pub_pc_full.publish(temp_out_msg);

                    m_voxel_filter_for_surface.setInputCloud(livox_surface);
                    m_voxel_filter_for_surface.filter(*livox_surface);
                    pcl::toROSMsg(*livox_surface, temp_out_msg);
                    temp_out_msg.header.stamp = current_time;
                    temp_out_msg.header.frame_id = "/camera_init";
                    pub_pc_surface.publish(temp_out_msg);

                    m_voxel_filter_for_corner.setInputCloud(livox_corners);
                    m_voxel_filter_for_corner.filter(*livox_corners);
                    pcl::toROSMsg(*livox_corners, temp_out_msg);
                    temp_out_msg.header.stamp = current_time;
                    temp_out_msg.header.frame_id = "/camera_init";
                    pub_pc_corners.publish(temp_out_msg);

                    if (m_odom_mode == 0) // odometry mode
                        break;
                }
            }
            return;
        }
        else
        {
            // VELODYNE LIDAR
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);
            removeClosedPointCloud(laserCloudIn, laserCloudIn, MINIMUM_RANGE);

            float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);
            float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y, laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;

            if (endOri - startOri > 3 * M_PI)
                endOri -= 2 * M_PI;
            else if (endOri - startOri < M_PI)
                endOri += 2 * M_PI;
            
            bool halfPassed = false;
            int count = cloudSize;
            PointType point;

            for (size_t i = 0; i < cloudSize; i++)
            {
                point.x = laserCloudIn.points[i].x;
                point.y = laserCloudIn.points[i].y;
                point.z = laserCloudIn.points[i].z;

                float angle = atan(point.z / sqrt(point.x * point.x + point.y * point.y)) * 180 / M_PI;
                int   scanID = 0;

                if (laser_scan_number == 16)
                {
                    scanID = int((angle + 15) / 2 + 0.5);

                    if (scanID > (laser_scan_number - 1) || scanID < 0)
                    {
                        count--;
                        continue;
                    }
                }
                else if (laser_scan_number == 64)
                {
                    if (angle >= -8.83)
                        scanID = int((2 - angle) * 3.0 + 0.5);
                    else
                        scanID = laser_scan_number / 2 + int((-8.83 - angle) * 2.0 + 0.5);

                    if (angle > 2 || angle < -24.33 || scanID > 50 || scanID < 0)
                    {
                        count--;
                        continue;
                    }
                }
                else
                {
                    printf("wrong scan number\n");
                    ROS_BREAK();
                }
                
                float ori = -atan2(point.y, point.x);

                if (!halfPassed)
                {
                    if (ori < startOri - M_PI / 2)
                        ori += 2 * M_PI;
                    else if (ori > startOri + M_PI * 3 / 2)
                        ori -= 2 * M_PI;

                    if (ori - startOri > M_PI)
                        halfPassed = true;
                }
                else
                {
                    ori += 2 * M_PI;

                    if (ori < endOri - M_PI * 3 / 2)
                        ori += 2 * M_PI;
                    else if (ori > endOri + M_PI / 2)
                        ori -= 2 * M_PI;
                }

                float relTime = (ori - startOri) / (endOri - startOri);
                point.intensity = scanID + m_para_scanPeriod * relTime;
                laserCloudScans[scanID].push_back(point);
            }
        }

        pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
        laserCloud->clear();
        cloudSize = 0;

        for (int i = 0; i < laser_scan_number; i++)
        {
            scanStartInd[i] = laserCloud->size() + 5;
            *laserCloud += laserCloudScans[i];
            scanEndInd[i] = laserCloud->size() - 6;
            cloudSize += laserCloudScans[i].size();
        }

        for (size_t i = 5; i < cloudSize - 5; i++)
        {
            float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x + laserCloud->points[i - 3].x + laserCloud->points[i - 2].x + laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x + laserCloud->points[i + 1].x + laserCloud->points[i + 2].x + laserCloud->points[i + 3].x + laserCloud->points[i + 4].x + laserCloud->points[i + 5].x;
            float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y + laserCloud->points[i - 3].y + laserCloud->points[i - 2].y + laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y + laserCloud->points[i + 1].y + laserCloud->points[i + 2].y + laserCloud->points[i + 3].y + laserCloud->points[i + 4].y + laserCloud->points[i + 5].y;
            float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z + laserCloud->points[i - 3].z + laserCloud->points[i - 2].z + laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z + laserCloud->points[i + 1].z + laserCloud->points[i + 2].z + laserCloud->points[i + 3].z + laserCloud->points[i + 4].z + laserCloud->points[i + 5].z;
            float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;
            m_pc_curvature[i] = diff;
            m_pc_sort_idx[i] = i;
            m_pc_neighbor_picked[i] = 0;
            m_pc_cloud_label[i] = 0;
            
            if (diff > 0.1)
            {
                float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x +
                                    laserCloud->points[i].y * laserCloud->points[i].y +
                                    laserCloud->points[i].z * laserCloud->points[i].z);
                float depth2 = sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x +
                                    laserCloud->points[i + 1].y * laserCloud->points[i + 1].y +
                                    laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);

                if (depth1 > depth2)
                {
                    diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * depth2 / depth1;
                    diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * depth2 / depth1;
                    diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * depth2 / depth1;

                    if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1)
                    {
                        m_pc_neighbor_picked[i - 5] = 1;
                        m_pc_neighbor_picked[i - 4] = 1;
                        m_pc_neighbor_picked[i - 3] = 1;
                        m_pc_neighbor_picked[i - 2] = 1;
                        m_pc_neighbor_picked[i - 1] = 1;
                        m_pc_neighbor_picked[i] = 1;
                    }
                }
                else
                {
                    diffX = laserCloud->points[i + 1].x * depth1 / depth2 - laserCloud->points[i].x;
                    diffY = laserCloud->points[i + 1].y * depth1 / depth2 - laserCloud->points[i].y;
                    diffZ = laserCloud->points[i + 1].z * depth1 / depth2 - laserCloud->points[i].z;

                    if (sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1)
                    {
                        m_pc_neighbor_picked[i + 1] = 1;
                        m_pc_neighbor_picked[i + 2] = 1;
                        m_pc_neighbor_picked[i + 3] = 1;
                        m_pc_neighbor_picked[i + 4] = 1;
                        m_pc_neighbor_picked[i + 5] = 1;
                        m_pc_neighbor_picked[i + 6] = 1;
                    }
                }
            }

            float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
            float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
            float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
            float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

            float dis = laserCloud->points[i].x * laserCloud->points[i].x +
                        laserCloud->points[i].y * laserCloud->points[i].y +
                        laserCloud->points[i].z * laserCloud->points[i].z;

            if (diff > 0.0002 * dis && diff2 > 0.0002 * dis)
                m_pc_neighbor_picked[i] = 1;
        }

#if !IF_LIVOX_HANDLER_REMOVE
    if (m_lidar_type != 0)
    {
        Livox_laser::Pt_infos *pt_info;
        for (unsigned int idx = 0; idx < cloudSize; idx++)
        {
            pt_info = m_livox.find_pt_info(laserCloud->points[idx]);

            if (pt_info->pt_type != Livox_laser::e_pt_normal)
                m_pc_neighbor_picked[idx] = 1;
        }
    }
#endif

        pcl::PointCloud<PointType> cornerPointsSharp;
        pcl::PointCloud<PointType> cornerPointsLessSharp;
        pcl::PointCloud<PointType> surfPointsFlat;
        pcl::PointCloud<PointType> surfPointsLessFlat;
        float sharp_point_threshold = 0.05;

        //extract corners points and surface points
        for (int i = 0; i < laser_scan_number; i++)
        {
            pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);
            // To ensure the distribution of features point, spilt each scan into 6 parts equally according to their curvature.
            for (int j = 0; j < 6; j++)
            {
                //Starting of each sub-scan.
                int sp = (scanStartInd[i] * (6 - j) + scanEndInd[i] * j) / 6;
                //Ending of each sub-scan.
                int ep = (scanStartInd[i] * (5 - j) + scanEndInd[i] * (j + 1)) / 6 - 1;

                //sort curvature
                for (int k = sp + 1; k <= ep; k++)
                {
                    for (int l = k; l >= sp + 1; l--)
                    {
                        if (m_pc_curvature[m_pc_sort_idx[l]] < m_pc_curvature[m_pc_sort_idx[l - 1]])
                        {
                            int temp = m_pc_sort_idx[l - 1];
                            m_pc_sort_idx[l - 1] = m_pc_sort_idx[l];
                            m_pc_sort_idx[l] = temp;
                        }
                    }
                }

                //select the most shart and flat point
                int largestPickedNum = 0;
                for (int k = ep; k >= sp; k--)
                {
                    int ind = m_pc_sort_idx[k]; //The index of biggest curvature.

                    if (m_pc_neighbor_picked[ind] == 0 && m_pc_curvature[ind] > sharp_point_threshold * 10)
                    {
                        largestPickedNum++;
                        if (largestPickedNum <= 20)
                        {
                            m_pc_cloud_label[ind] = 2; //2 -> the label sharpest points.
                            cornerPointsSharp.push_back(laserCloud->points[ind]);
                            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                        }
                        else if (largestPickedNum <= 200)
                        {
                            m_pc_cloud_label[ind] = 1; //1 -> the label of less sharpest points.
                            cornerPointsLessSharp.push_back(laserCloud->points[ind]);
                        }
                        else
                            break;

                        m_pc_neighbor_picked[ind] = 1;

                        float times = 100;
                        // delete 5 neighbor of sharpest points.
                        for (int l = 1; l <= 5 * times; l++)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;

                            m_pc_neighbor_picked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5 * times; l--)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;

                            m_pc_neighbor_picked[ind + l] = 1;
                        }
                    }
                }

                int smallestPickedNum = 0;
                for (int k = sp; k <= ep; k++)
                {
                    int ind = m_pc_sort_idx[k];

                    if (m_pc_neighbor_picked[ind] == 0 && m_pc_curvature[ind] < sharp_point_threshold)
                    {
                        m_pc_cloud_label[ind] = -1; // -1 the lable of flat points
                        surfPointsFlat.push_back(laserCloud->points[ind]);

                        smallestPickedNum++;
                        if (smallestPickedNum >= 5)
                            break;

                        m_pc_neighbor_picked[ind] = 1;
                        for (int l = 1; l <= 5; l++)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l - 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l - 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l - 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;

                            m_pc_neighbor_picked[ind + l] = 1;
                        }
                        for (int l = -1; l >= -5; l--)
                        {
                            float diffX = laserCloud->points[ind + l].x - laserCloud->points[ind + l + 1].x;
                            float diffY = laserCloud->points[ind + l].y - laserCloud->points[ind + l + 1].y;
                            float diffZ = laserCloud->points[ind + l].z - laserCloud->points[ind + l + 1].z;
                            if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
                                break;

                            m_pc_neighbor_picked[ind + l] = 1;
                        }
                    }
                }

                // The ublabeled point is the less flat points
                for (int k = sp; k <= ep; k++)
                    if (m_pc_cloud_label[k] <= 0)
                        surfPointsLessFlatScan->push_back(laserCloud->points[k]);
            }

            // voxel filter for less sharp points.
            pcl::PointCloud<PointType> surfPointsLessFlatScanDS;

            m_voxel_filter_for_surface.setInputCloud(surfPointsLessFlatScan);
            m_voxel_filter_for_surface.filter(surfPointsLessFlatScanDS);

            surfPointsLessFlat += surfPointsLessFlatScanDS;
        }
        
        sensor_msgs::PointCloud2 laserCloudOutMsg;
        pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
        laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
        laserCloudOutMsg.header.frame_id = "/camera_init";
        pub_laser_pc.publish(laserCloudOutMsg);

        sensor_msgs::PointCloud2 cornerPointsSharpMsg;
        pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
        cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
        cornerPointsSharpMsg.header.frame_id = "/camera_init";
        pub_pc_sharp_corner.publish(cornerPointsSharpMsg);

        sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
        pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
        cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
        cornerPointsLessSharpMsg.header.frame_id = "/camera_init";
        pub_pc_less_sharp_corner.publish(cornerPointsLessSharpMsg);

        sensor_msgs::PointCloud2 surfPointsFlat2;
        pcl::toROSMsg(surfPointsFlat, surfPointsFlat2);
        surfPointsFlat2.header.stamp = laserCloudMsg->header.stamp;
        surfPointsFlat2.header.frame_id = "/camera_init";
        pub_pc_surface_flat.publish(surfPointsFlat2);

        sensor_msgs::PointCloud2 surfPointsLessFlat2;
        pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlat2);
        surfPointsLessFlat2.header.stamp = laserCloudMsg->header.stamp;
        surfPointsLessFlat2.header.frame_id = "/camera_init";
        pub_pc_surface_less_flat.publish(surfPointsLessFlat2);

        // pub each scam
        if (m_if_pub_each_line)
        {
            for (int i = 0; i < laser_scan_number; i++)
            {
                sensor_msgs::PointCloud2 scanMsg;
                pcl::toROSMsg(laserCloudScans[i], scanMsg);
                scanMsg.header.stamp = laserCloudMsg->header.stamp;
                scanMsg.header.frame_id = "/camera_init";
                pub_each_scan[i].publish(scanMsg);
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

        if (ros::param::get("livox_min_dis", m_livox.m_livox_min_allow_dis))
            std::cout << "Set livox lidar minimum distance= " << m_livox.m_livox_min_allow_dis << std::endl;

        if (ros::param::get("livox_min_sigma", m_livox.m_livox_min_sigma))
            std::cout << "Set livox lidar minimum sigama =  " << m_livox.m_livox_min_sigma << std::endl;

        std::cout << "~~~~~ End ~~~~~" << endl;
    }
};

#endif // LASER_FEATURE_EXTRACTION_H