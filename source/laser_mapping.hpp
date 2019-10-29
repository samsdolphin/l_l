// This is the Lidar Odometry And Mapping (LOAM) for solid-state lidar (for example: livox lidar),
// which suffer form motion blur due the continously scan pattern and low range of fov.

// Developer: Lin Jiarong  ziv.lin.ljr@gmail.com

//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

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



int if_motion_deblur = 0;
double history_add_t_step = 0.00;
double history_add_angle_step = 0.00;

using namespace PCL_TOOLS;
using namespace Common_tools;

struct Data_pair
{
	sensor_msgs::PointCloud2ConstPtr m_pc_corner;
	sensor_msgs::PointCloud2ConstPtr m_pc_full;
	sensor_msgs::PointCloud2ConstPtr m_pc_plane;
	bool                             m_has_pc_corner = 0;
	bool                             m_has_pc_full = 0;
	bool                             m_has_pc_plane = 0;

	void add_pc_corner(sensor_msgs::PointCloud2ConstPtr ros_pc)
	{
		m_pc_corner = ros_pc;
		m_has_pc_corner = true;
	}

	void add_pc_plane(sensor_msgs::PointCloud2ConstPtr ros_pc)
	{
		m_pc_plane = ros_pc;
		m_has_pc_plane = true;
	}

	void add_pc_full(sensor_msgs::PointCloud2ConstPtr ros_pc)
	{
		m_pc_full = ros_pc;
		m_has_pc_full = true;
	}

	bool is_completed()
	{
		return (m_has_pc_corner & m_has_pc_full & m_has_pc_plane);
	}
};

class Point_cloud_registration;

class Laser_mapping
{
  public:
	int m_current_frame_index = 0;
	int m_para_min_match_blur = 0.0;
	int m_para_max_match_blur = 0.3;
	//int   m_max_buffer_size = 5;
	int   m_max_buffer_size = 50000000;

	int   m_mapping_init_accumulate_frames = 100;
	int   m_kmean_filter_count = 3;
	int   m_kmean_filter_threshold = 2.0;

	double m_time_pc_corner_past = 0;
	double m_time_pc_surface_past = 0;
	double m_time_pc_full = 0;
	double m_time_odom = 0;
	double m_last_time_stamp = 0;
	double m_minimum_pt_time_stamp = 0;
	double m_maximum_pt_time_stamp = 1.0;
	float  m_last_max_blur = 0.0;

	int m_odom_mode;
	int m_matching_mode = 0;
	int m_if_input_downsample_mode = 1;
	int m_maximum_parallel_thread;
	int m_maximum_mapping_buff_thread = 1; // Maximum number of thead for matching buffer update
	int m_maximum_history_size = 100;

	float  m_para_max_angular_rate = 200.0 / 50.0; // max angular rate = 90.0 /50.0 deg/s
	float m_para_max_speed = 100.0 / 50.0;        // max speed = 10 m/s
	float m_max_final_cost = 100.0;
	int   m_para_icp_max_iterations = 20;
	int   m_para_cere_max_iterations = 100;
	double m_minimum_icp_R_diff = 0.01;
	double m_minimum_icp_T_diff = 0.01;

	string m_pcd_save_dir_name, m_log_save_dir_name, m_loop_save_dir_name;

	std::list<pcl::PointCloud<PointType>> map_corner_history;
	std::list<pcl::PointCloud<PointType>> map_surface_history;
	std::list<pcl::PointCloud<PointType>> m_laser_cloud_full_history;
	std::list<double>                     m_his_reg_error;
	Eigen::Quaterniond                    m_last_his_add_q;
	Eigen::Vector3d m_last_his_add_t;


	//
	std::map<int, float> m_map_life_time_corner;
	std::map<int, float> m_map_life_time_surface;

	// ouput: all visualble cube points
	pcl::PointCloud<PointType>::Ptr m_laser_cloud_surround;

	// surround points in map to build tree
	int m_if_mapping_updated_corner = true;
	int m_if_mapping_updated_surface = true;

	pcl::PointCloud<PointType>::Ptr m_laser_cloud_corner_from_map;
	pcl::PointCloud<PointType>::Ptr m_laser_cloud_surf_from_map;

	pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map_pre;
	pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map_pre;

	//input & output: points in one frame. local --> global
	pcl::PointCloud<PointType>::Ptr laser_cloud_full_cur;

	// input: from odom
	pcl::PointCloud<PointType>::Ptr laser_cloud_corner_cur;
	pcl::PointCloud<PointType>::Ptr laser_cloud_surf_cur;

	//kd-tree
	pcl::KdTreeFLANN<PointType> m_kdtree_corner_from_map;
	pcl::KdTreeFLANN<PointType> m_kdtree_surf_from_map;

	pcl::KdTreeFLANN<PointType> kdtree_corner_from_map_pre;
	pcl::KdTreeFLANN<PointType> kdtree_surf_from_map_pre;

	int m_laser_cloud_valid_Idx[1024];
	int m_laser_cloud_surround_Idx[1024];

	const Eigen::Quaterniond m_q_I = Eigen::Quaterniond(1, 0, 0, 0);

	double m_para_buffer_RT[7] = {0, 0, 0, 1, 0, 0, 0};
	double m_para_buffer_RT_last[7] = {0, 0, 0, 1, 0, 0, 0};

	Eigen::Map<Eigen::Quaterniond> m_q_w_curr = Eigen::Map<Eigen::Quaterniond>(m_para_buffer_RT);
	Eigen::Map<Eigen::Vector3d>    m_t_w_curr = Eigen::Map<Eigen::Vector3d>(m_para_buffer_RT + 4);

	Eigen::Map<Eigen::Quaterniond> m_q_w_last = Eigen::Map<Eigen::Quaterniond>(m_para_buffer_RT_last);
	Eigen::Map<Eigen::Vector3d>    m_t_w_last = Eigen::Map<Eigen::Vector3d>(m_para_buffer_RT_last + 4);

	std::map<double, Data_pair *> m_map_data_pair;
	std::queue<Data_pair *> m_queue_avail_data;

	std::queue<nav_msgs::Odometry::ConstPtr> m_odom_que;
	std::mutex                               mutex_buf;

	float m_line_resolution = 0;
	float m_plane_resolution = 0;
	pcl::VoxelGrid<PointType>                 m_down_sample_filter_corner;
	pcl::VoxelGrid<PointType>                 m_down_sample_filter_surface;
	pcl::StatisticalOutlierRemoval<PointType> m_filter_k_means;

	std::vector<int>   m_point_search_Idx;
	std::vector<float> m_point_search_sq_dis;

	nav_msgs::Path m_laser_after_mapped_path, m_laser_after_loopclosure_path;

	int       m_if_save_to_pcd_files = 1;
	PCL_tools m_pcl_tools_aftmap;
	PCL_tools m_pcl_tools_raw;

	Common_tools::File_logger m_logger_common;
	Common_tools::File_logger m_logger_pcd;
	Common_tools::File_logger m_logger_timer;
	Common_tools::File_logger m_logger_matching_buff;
	Scene_alignment<float>    m_sceene_align;
	Common_tools::Timer m_timer;

	ros::Publisher  m_pub_laser_cloud_surround, m_pub_laser_cloud_map, m_pub_laser_cloud_full_res, m_pub_odom_aft_mapped, m_pub_odom_aft_mapped_hight_frec;
	ros::Publisher  m_pub_laser_aft_mapped_path, m_pub_laser_aft_loopclosure_path;
	ros::NodeHandle m_ros_node_handle;
	ros::Subscriber m_sub_laser_cloud_corner_last, m_sub_laser_cloud_surf_last, m_sub_laser_odom, m_sub_laser_cloud_full_res;

	ceres::Solver::Summary m_final_opt_summary;
	//std::list<std::thread* > m_thread_pool;
	std::list< std::future<int>* >           m_thread_pool;
	std::list< std::future<void> * >         m_thread_match_buff_refresh;

	double                                   m_maximum_in_fov_angle ;
	double                                   m_maximum_pointcloud_delay_time;
	double                                   m_maximum_search_range_corner;
	double                                   m_maximum_search_range_surface;
	double                                   m_surround_pointcloud_resolution;
	double                                   m_lastest_pc_reg_time = -3e8;
	double                                   m_lastest_pc_matching_refresh_time = -3e8;
	double                                   m_lastest_pc_income_time = -3e8;

	std::mutex mutex_mapping;
	std::mutex mutex_querypointcloud;
	std::mutex mutex_buff_for_matching_corner;
	std::mutex mutex_buff_for_matching_surface;
	std::mutex mutex_thread_pool;
	std::mutex mutex_ros_pub;
	std::mutex mutex_dump_full_history;

	float                   m_pt_cell_resolution = 1.0;
	Points_cloud_map<float> m_pt_cell_map_full;
	Points_cloud_map<float> m_pt_cell_map_corners;
	Points_cloud_map<float> m_pt_cell_map_planes;

	int                m_down_sample_replace = 1;
	ros::Publisher     m_pub_last_corner_pts, m_pub_last_surface_pts;
	ros::Publisher     m_pub_match_corner_pts, m_pub_match_surface_pts, m_pub_debug_pts, m_pub_pc_aft_loop;
	std::future<void> *m_mapping_refresh_service_corner, *m_mapping_refresh_service_surface, *m_mapping_refresh_service; // Thread for mapping update
	std::future<void> *m_service_pub_surround_pts, *m_service_loop_detection;                                            // Thread for loop detection and publish surrounding pts

	Common_tools::Timer timer_all;
	std::mutex          timer_log_mutex;

	int   m_loop_closure_if_enable;
	int   m_loop_closure_minimum_keyframe_differen;
	float m_loop_closure_minimum_similarity_linear;
	float m_loop_closure_minimum_similarity_planar;
	float m_loop_closure_map_alignment_resolution;
	float m_loop_closure_map_alignment_inlier_threshold;

	ADD_SCREEN_PRINTF_OUT_METHOD;

	int if_pt_in_fov(const Eigen::Matrix<double,3,1> & pt)
	{
		auto pt_affine = m_q_w_curr.inverse() * (pt - m_t_w_curr);

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
		std::map<int, float>::iterator it = map_life_time.find(index) ;
		if (it == map_life_time.end())
		{
		  map_life_time.insert(std::make_pair(index, m_last_time_stamp));
		}
		else
		{
		  it->second = m_last_time_stamp;
		}
	}

	void update_buff_for_matching()
	{
		if (m_lastest_pc_matching_refresh_time == m_lastest_pc_reg_time)
			return;
		m_timer.tic("Update buff for matching");

		pcl::VoxelGrid<PointType> down_sample_filter_corner = m_down_sample_filter_corner;
		pcl::VoxelGrid<PointType> down_sample_filter_surface = m_down_sample_filter_surface;
		down_sample_filter_corner.setLeafSize(m_line_resolution, m_line_resolution, m_line_resolution);
		down_sample_filter_surface.setLeafSize(m_plane_resolution, m_plane_resolution, m_plane_resolution);
		pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map(new pcl::PointCloud<PointType>());
		
		if (m_matching_mode)
		{
			//pcl::VoxelGrid<PointType> down_sample_filter_corner = m_down_sample_filter_corner;
			//pcl::VoxelGrid<PointType> down_sample_filter_surface = m_down_sample_filter_surface;
			std::vector<Points_cloud_map<float>::Mapping_cell *> corner_cell_vec = m_pt_cell_map_corners.find_cells_in_radius(m_t_w_curr, m_maximum_search_range_corner);
			std::vector<Points_cloud_map<float>::Mapping_cell *> plane_cell_vec = m_pt_cell_map_planes.find_cells_in_radius(m_t_w_curr, m_maximum_search_range_surface);
			int corner_cell_numbers_in_fov = 0;
			int surface_cell_numbers_in_fov = 0;
			pcl::PointCloud<PointType> pc_temp;

			for (size_t i = 0; i < corner_cell_vec.size(); i++)
			{
				int if_in_fov = if_pt_in_fov(corner_cell_vec[i]->m_center.cast<double>());
				if (if_in_fov == 0)
					continue;

				corner_cell_numbers_in_fov++;
				down_sample_filter_corner.setInputCloud(corner_cell_vec[i]->get_pointcloud().makeShared());
				down_sample_filter_corner.filter(pc_temp);
				if (m_down_sample_replace)
					corner_cell_vec[i]->set_pointcloud(pc_temp);

				*laser_cloud_corner_from_map += pc_temp;
			}

			for (size_t i = 0; i < plane_cell_vec.size(); i++)
			{
				int if_in_fov = if_pt_in_fov(plane_cell_vec[i]->m_center.cast<double>());
				if (if_in_fov == 0)
					continue;

				surface_cell_numbers_in_fov++;
				down_sample_filter_surface.setInputCloud(plane_cell_vec[i]->get_pointcloud().makeShared());
				down_sample_filter_surface.filter(pc_temp);
				if (m_down_sample_replace)
					plane_cell_vec[i]->set_pointcloud(pc_temp);

				*laser_cloud_surf_from_map += pc_temp;
			}
		}
		else
		{
			mutex_mapping.lock();
			for (auto it = map_corner_history.begin(); it != map_corner_history.end(); it++)
				*laser_cloud_corner_from_map += (*it);

			for (auto it = map_surface_history.begin(); it != map_surface_history.end(); it++)
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

		m_if_mapping_updated_corner = false;
		m_if_mapping_updated_surface = false;

		mutex_buff_for_matching_corner.lock();
		*laser_cloud_corner_from_map_pre = *laser_cloud_corner_from_map;
		kdtree_corner_from_map_pre = kdtree_corner_from_map;
		mutex_buff_for_matching_surface.unlock();

		mutex_buff_for_matching_surface.lock();
		*laser_cloud_surf_from_map_pre = *laser_cloud_surf_from_map;
		kdtree_surf_from_map_pre = kdtree_surf_from_map;
		mutex_buff_for_matching_corner.unlock();

		if ((m_lastest_pc_reg_time > m_lastest_pc_matching_refresh_time) || (m_lastest_pc_reg_time < 10))
			m_lastest_pc_matching_refresh_time = m_lastest_pc_reg_time;

		*m_logger_matching_buff.get_ostream() << m_timer.toc_string("Update buff for matching") << std::endl;
	}

	void service_update_buff_for_matching()
	{
		while (1)
		{
			std::this_thread::sleep_for (std::chrono::nanoseconds(100));
			update_buff_for_matching();
		}
	}

	Laser_mapping()
	{
		laser_cloud_corner_cur = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
		laser_cloud_surf_cur = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
		m_laser_cloud_surround = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());

		m_laser_cloud_corner_from_map = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
		m_laser_cloud_surf_from_map = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());

		laser_cloud_corner_from_map_pre = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());
		laser_cloud_surf_from_map_pre = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());

		laser_cloud_full_cur = pcl::PointCloud<PointType>::Ptr(new pcl::PointCloud<PointType>());

		init_parameters(m_ros_node_handle);

		//livox_corners
		m_sub_laser_cloud_corner_last = m_ros_node_handle.subscribe<sensor_msgs::PointCloud2>("/pc2_corners", 10000, &Laser_mapping::laserCloudCornerLastHandler, this);
		m_sub_laser_cloud_surf_last = m_ros_node_handle.subscribe<sensor_msgs::PointCloud2>("/pc2_surface", 10000, &Laser_mapping::laserCloudSurfLastHandler, this);
		m_sub_laser_cloud_full_res = m_ros_node_handle.subscribe<sensor_msgs::PointCloud2>("/pc2_full", 10000, &Laser_mapping::laserCloudFullResHandler, this);

		m_pub_laser_cloud_surround = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 10000);

		m_pub_last_corner_pts = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/features_corners", 10000);
		m_pub_last_surface_pts = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/features_surface", 10000);

		m_pub_match_corner_pts = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/match_pc_corners", 10000);
		m_pub_match_surface_pts = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/match_pc_surface", 10000);
		m_pub_pc_aft_loop = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/pc_aft_loop_closure", 10000);
		m_pub_debug_pts =  m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/pc_debug", 10000);

		m_pub_laser_cloud_map = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/laser_cloud_map", 10000);
		m_pub_laser_cloud_full_res = m_ros_node_handle.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registered", 10000);
		m_pub_odom_aft_mapped = m_ros_node_handle.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10000);
		m_pub_odom_aft_mapped_hight_frec = m_ros_node_handle.advertise<nav_msgs::Odometry>("/aft_mapped_to_init_high_frec", 10000);
		m_pub_laser_aft_mapped_path = m_ros_node_handle.advertise<nav_msgs::Path>("/aft_mapped_path", 10000);
		m_pub_laser_aft_loopclosure_path = m_ros_node_handle.advertise<nav_msgs::Path>("/aft_loopclosure_path", 10000);

		m_pt_cell_map_full.set_resolution(m_pt_cell_resolution);
		m_pt_cell_map_corners.set_resolution(m_pt_cell_resolution);
		m_pt_cell_map_planes.set_resolution(m_pt_cell_resolution);

		screen_out << "Laser_mapping init OK" << endl;
	};

	~Laser_mapping(){};

	Data_pair *get_data_pair(const double &time_stamp)
	{
		std::map<double, Data_pair *>::iterator it = m_map_data_pair.find(time_stamp);
		if (it == m_map_data_pair.end())
		{
			Data_pair *date_pair_ptr = new Data_pair();
			m_map_data_pair.insert(std::make_pair(time_stamp, date_pair_ptr));
			return date_pair_ptr;
		}
		else
		{
			return it->second;
		}
	}

	template <typename T>
	T get_ros_parameter(ros::NodeHandle &nh, const std::string parameter_name, T &parameter, T default_val)
	{
		nh.param<T>(parameter_name.c_str(), parameter, default_val);
		ENABLE_SCREEN_PRINTF;
		screen_out << "[Laser_mapping_ros_param]: " << parameter_name << " ==> " << parameter << std::endl;
		return parameter;
	}

	void init_parameters(ros::NodeHandle &nh)
	{

		get_ros_parameter<float>(nh, "feature_extraction/mapping_line_resolution", m_line_resolution, 0.4);
		get_ros_parameter<float>(nh, "feature_extraction/mapping_plane_resolution", m_plane_resolution, 0.8);

		if (m_odom_mode == 1)
		{
			//m_max_buffer_size = 3e8;
		}

		get_ros_parameter<int>(nh, "common/if_verbose_screen_printf", m_if_verbose_screen_printf, 1);
		get_ros_parameter<int>(nh, "common/odom_mode", m_odom_mode, 0);
		get_ros_parameter<int>(nh, "common/maximum_parallel_thread", m_maximum_parallel_thread, 4);
		get_ros_parameter<int>(nh, "common/if_motion_deblur", if_motion_deblur, 0);
		get_ros_parameter<int>(nh, "common/if_save_to_pcd_files", m_if_save_to_pcd_files, 0);

		get_ros_parameter<double>(nh, "optimization/minimum_icp_R_diff", m_minimum_icp_R_diff, 0.01);
		get_ros_parameter<double>(nh, "optimization/minimum_icp_T_diff", m_minimum_icp_T_diff, 0.01);
		get_ros_parameter<int>(nh, "optimization/icp_maximum_iteration", m_para_icp_max_iterations, 20);
		get_ros_parameter<int>(nh, "optimization/ceres_maximum_iteration", m_para_cere_max_iterations, 20);
		get_ros_parameter<float>(nh, "optimization/max_allow_incre_R", m_para_max_angular_rate, 200.0 / 50.0);
		get_ros_parameter<float>(nh, "optimization/max_allow_incre_T", m_para_max_speed, 100.0 / 50.0);
		get_ros_parameter<float>(nh, "optimization/max_allow_final_cost", m_max_final_cost, 1.0);

		get_ros_parameter<int>(nh, "mapping/init_accumulate_frames", m_mapping_init_accumulate_frames, 50);
		get_ros_parameter<int>(nh, "mapping/maximum_histroy_buffer", m_maximum_history_size, 100);
		get_ros_parameter<int>(nh, "mapping/maximum_mapping_buffer", m_max_buffer_size, 5);
		get_ros_parameter<int>(nh, "mapping/matching_mode", m_matching_mode, 1);
		get_ros_parameter<int>(nh, "mapping/input_downsample_mode", m_if_input_downsample_mode, 1);
		get_ros_parameter<double>(nh, "mapping/maximum_in_fov_angle", m_maximum_in_fov_angle, 30);
		get_ros_parameter<double>(nh, "mapping/maximum_pointcloud_delay_time", m_maximum_pointcloud_delay_time, 0.1);
		get_ros_parameter<double>(nh, "mapping/maximum_in_fov_angle", m_maximum_in_fov_angle, 30);
		get_ros_parameter<double>(nh, "mapping/maximum_search_range_corner", m_maximum_search_range_corner, 100);
		get_ros_parameter<double>(nh, "mapping/maximum_search_range_surface", m_maximum_search_range_surface, 100);
		get_ros_parameter<double>(nh, "mapping/surround_pointcloud_resolution", m_surround_pointcloud_resolution, 0.5);

		get_ros_parameter<int>(nh, "loop_closure/if_enable_loop_closure", m_loop_closure_if_enable, 0);
		get_ros_parameter<int>(nh, "loop_closure/minimum_keyframe_differen", m_loop_closure_minimum_keyframe_differen, 200);
		get_ros_parameter<float>(nh, "loop_closure/minimum_similarity_linear", m_loop_closure_minimum_similarity_linear, 0.65);
		get_ros_parameter<float>(nh, "loop_closure/minimum_similarity_planar", m_loop_closure_minimum_similarity_planar, 0.95);
		get_ros_parameter<float>(nh, "loop_closure/map_alignment_resolution", m_loop_closure_map_alignment_resolution, 0.2);
		get_ros_parameter<float>(nh, "loop_closure/map_alignment_inlier_threshold", m_loop_closure_map_alignment_inlier_threshold, 0.35);

		get_ros_parameter<std::string>(nh, "common/log_save_dir", m_log_save_dir_name, "../");
		m_logger_common.set_log_dir(m_log_save_dir_name);
		m_logger_common.init("mapping.log");
		m_logger_timer.set_log_dir(m_log_save_dir_name);
		m_logger_timer.init("timer.log");
		m_logger_matching_buff.set_log_dir(m_log_save_dir_name);
		m_logger_matching_buff.init("match_buff.log");

		get_ros_parameter<std::string>(nh, "common/pcd_save_dir", m_pcd_save_dir_name, std::string("./"));
		get_ros_parameter<std::string>(nh, "common/loop_save_dir", m_loop_save_dir_name, m_pcd_save_dir_name.append("_loop"));
		m_sceene_align.init(m_loop_save_dir_name);

		if (m_if_save_to_pcd_files)
		{
			m_pcl_tools_aftmap.set_save_dir_name(m_pcd_save_dir_name);
			m_pcl_tools_raw.set_save_dir_name(m_pcd_save_dir_name);

		}

		m_logger_pcd.set_log_dir(m_log_save_dir_name);
		m_logger_pcd.init("poses.log");

		LOG_FILE_LINE(m_logger_common);
		*m_logger_common.get_ostream() << m_logger_common.version();

		screen_printf("line resolution %f plane resolution %f \n", m_line_resolution, m_plane_resolution);
		m_logger_common.printf("line resolution %f plane resolution %f \n", m_line_resolution, m_plane_resolution);
		m_down_sample_filter_corner.setLeafSize(m_line_resolution, m_line_resolution, m_line_resolution);
		m_down_sample_filter_surface.setLeafSize(m_plane_resolution, m_plane_resolution, m_plane_resolution);

		m_filter_k_means.setMeanK(m_kmean_filter_count);
		m_filter_k_means.setStddevMulThresh(m_kmean_filter_threshold);
	}

	void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
	{
		std::unique_lock<std::mutex> lock(mutex_buf);
		Data_pair *                  data_pair = get_data_pair(laserCloudCornerLast2->header.stamp.toSec());
		data_pair->add_pc_corner(laserCloudCornerLast2);
		if (data_pair->is_completed())
		{
			m_queue_avail_data.push(data_pair);
		}
	}

	void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
	{
		std::unique_lock<std::mutex> lock(mutex_buf);
		Data_pair *                  data_pair = get_data_pair(laserCloudSurfLast2->header.stamp.toSec());
		data_pair->add_pc_plane(laserCloudSurfLast2);
		if (data_pair->is_completed())
		{
			m_queue_avail_data.push(data_pair);
		}
	}

	void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
	{
		std::unique_lock<std::mutex> lock(mutex_buf);
		Data_pair *                  data_pair = get_data_pair(laserCloudFullRes2->header.stamp.toSec());
		data_pair->add_pc_full(laserCloudFullRes2);
		if (data_pair->is_completed())
		{
			m_queue_avail_data.push(data_pair);
		}
	}

	template < typename T, typename TT >
	static void save_mat_to_json_writter(T &writer, const std::string &name, const TT &eigen_mat)
	{
		writer.Key(name.c_str()); // output a key,
		writer.StartArray();        // Between StartArray()/EndArray(),
		for (size_t i = 0; i < (size_t)(eigen_mat.cols() * eigen_mat.rows()); i++)
			writer.Double(eigen_mat(i));
		writer.EndArray();
	}

	template < typename T, typename TT >
	static void save_quaternion_to_json_writter(T &writer, const std::string &name, const Eigen::Quaternion<TT> & q_curr)
	{
	  writer.Key(name.c_str());
	  writer.StartArray();
	  writer.Double(q_curr.w());
	  writer.Double(q_curr.x());
	  writer.Double(q_curr.y());
	  writer.Double(q_curr.z());
	  writer.EndArray();
	}

	template < typename T, typename TT >
	static void save_data_vec_to_json_writter(T &writer, const std::string &name, TT & data_vec)
	{
	  writer.Key(name.c_str());
	  writer.StartArray();
	  for (auto it = data_vec.begin(); it!=data_vec.end(); it++)
	  {
		writer.Double(*it);
	 }
	  writer.EndArray();
	}

	void dump_pose_and_regerror(std::string file_name, Eigen::Quaterniond & q_curr,
										   Eigen::Vector3d & t_curr,
										  std::list<double> & reg_err_vec)
	{
	  rapidjson::Document     document;
	  rapidjson::StringBuffer sb;
	  //rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(sb);
	  rapidjson::Writer< rapidjson::StringBuffer > writer(sb);
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
	  else
	  {
		for (int i =0;i< 109;i++)
		{
		  screen_out << "Write data to file: " << file_name << " error!!!" << std::endl;
		}
	 }
	}

	void loop_closure_pub_optimzed_path(const Ceres_pose_graph_3d::MapOfPoses & pose3d_aft_loopclosure)
	{

		nav_msgs::Odometry odom;
		m_laser_after_loopclosure_path.header.stamp = ros::Time::now();
		m_laser_after_loopclosure_path.header.frame_id = "/camera_init";
		for (auto it = pose3d_aft_loopclosure.begin();
			  it != pose3d_aft_loopclosure.end(); it++)
		{
			geometry_msgs::PoseStamped pose_stamp;
			Ceres_pose_graph_3d::Pose3d pose_3d = it->second;
			pose_stamp.pose.orientation.x = pose_3d.q.x();
			pose_stamp.pose.orientation.y = pose_3d.q.y();
			pose_stamp.pose.orientation.z = pose_3d.q.z();
			pose_stamp.pose.orientation.w = pose_3d.q.w();

			pose_stamp.pose.position.x = pose_3d.p(0);
			pose_stamp.pose.position.y = pose_3d.p(1);
			pose_stamp.pose.position.z = pose_3d.p(2);

			pose_stamp.header.frame_id = "/camera_init";
			//cout << "Pose : q = [" << pose_3d.q.coeffs().transpose() << "], p = [" <<pose_3d.p.transpose() << "]\r\n";
			m_laser_after_loopclosure_path.poses.push_back(pose_stamp);
		}
		//m_laser_after_loopclosure_path.child_frame_id = "/aft_mapped";
		m_pub_laser_aft_loopclosure_path.publish(m_laser_after_loopclosure_path);
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
		std::vector< Points_cloud_map<float> * > pt_map_vec;
		Scene_alignment<float> scene_align;
		Mapping_refine<PointType> map_rfn;
		std::vector<std::string> m_filename_vec;

		std::map<int, std::string>                                            map_file_name;
		Ceres_pose_graph_3d::MapOfPoses                                       pose3d_map, pose3d_map_ori;
		Ceres_pose_graph_3d::VectorOfPose                                     pose3d_vec;
		Ceres_pose_graph_3d::VectorOfConstraints                              constrain_vec;

		float avail_ratio_plane = 0.05;    // 0.05 for 300 scans, 0.15 for 1000 scans
		float avail_ratio_line = 0.05;
		scene_align.init(m_loop_save_dir_name);
		map_rfn.set_save_dir(std::string(m_loop_save_dir_name).append("/mapping_refined"));
		map_rfn.set_down_sample_resolution(0.2);
		FILE * fp = fopen(std::string(m_loop_save_dir_name).append("/loop.log").c_str(), "w+");
		std::map<int,pcl::PointCloud<PointType> > map_id_pc;
		int if_end=  0;
		pcl::VoxelGrid< PointType > down_sample_filter;
		down_sample_filter.setLeafSize(m_surround_pointcloud_resolution,m_surround_pointcloud_resolution,m_surround_pointcloud_resolution);
		while (1)
		{

			std::this_thread::sleep_for (std::chrono::milliseconds(1));

			if ((m_current_frame_index - last_update_index < 100) || (m_current_frame_index < 100))
			{
				continue;
			}

			if (m_laser_cloud_full_history.size() < 0.95 * m_maximum_history_size)
			{
				continue;
			}

			mutex_dump_full_history.lock();
			q_curr = m_q_w_curr;
			t_curr = m_t_w_curr;
			reg_error_his = m_his_reg_error;
			curren_frame_idx = m_current_frame_index;
			Points_cloud_map<float> *pt_cell_map_temp = new Points_cloud_map<float>();
			pt_cell_map_temp->set_resolution(1.0);
			pt_full.clear();
			for (auto it = m_laser_cloud_full_history.begin(); it != m_laser_cloud_full_history.end(); it++)
			{
				pt_full += (*it);
			}


			pt_cell_map_temp->set_point_cloud(PCL_TOOLS::pcl_pts_to_eigen_pts<float, PointType>(pt_full.makeShared()));

			mutex_dump_full_history.unlock();

			if (1)
			{
			  down_sample_filter.setInputCloud(pt_full.makeShared());
			  down_sample_filter.filter(pt_full);
			}

			map_id_pc.insert(std::make_pair(map_id_pc.size(), pt_full));

			pose3d_vec.push_back(Ceres_pose_graph_3d::Pose3d(q_curr, t_curr));
			pose3d_map.insert(std::make_pair(pose3d_map.size(), Ceres_pose_graph_3d::Pose3d(q_curr, t_curr)));
			if (pose3d_vec.size() >= 2)
			{
				Ceres_pose_graph_3d::Constraint3d temp_csn;
				Eigen::Vector3d                   relative_T = pose3d_vec[pose3d_vec.size() - 2].q.inverse() * (t_curr - pose3d_vec[pose3d_vec.size() - 2].p);
				Eigen::Quaterniond                relative_Q = pose3d_vec[pose3d_vec.size() - 2].q.inverse() * q_curr;
				/*
						std::cout << "-----------" << std::endl;
						std::cout << relative_Q.coeffs().transpose() << std::endl;
						std::cout << relative_T.transpose() << std::endl;
						*/
				temp_csn = Ceres_pose_graph_3d::Constraint3d(pose3d_vec.size() - 2, pose3d_vec.size() - 1,
															  relative_Q, relative_T);
				constrain_vec.push_back(temp_csn);
			}

			// Save pose
			json_file_name = std::string(m_loop_save_dir_name).append("/pose_").append(std::to_string(curren_frame_idx)).append(".json");
			dump_pose_and_regerror(json_file_name, q_curr, t_curr, reg_error_his);
			last_update_index = m_current_frame_index;
			m_timer.tic("Find loop");
			pt_cell_map_temp->analyze_mapping(1);
			float ratio_non_zero_plane = pt_cell_map_temp->m_ratio_nonzero_plane;
			float ratio_non_zero_line = pt_cell_map_temp->m_ratio_nonzero_line;

			// Save mappgin
			json_file_name = std::string("mapping_").append(std::to_string(curren_frame_idx)).append(".json");
			pt_cell_map_temp->save_to_file(std::string(m_loop_save_dir_name), json_file_name);

			pt_map_vec.push_back(pt_cell_map_temp);

			//
			 map_file_name.insert(std::make_pair(map_file_name.size(), std::string(m_loop_save_dir_name).append("/").append(json_file_name)));
			 m_filename_vec.push_back(std::string(m_loop_save_dir_name).append("/").append(json_file_name)) ;
			 float sim_plane_res_cv = 0, sim_plane_res = 0;
			 float sim_line_res_cv = 0, sim_line_res = 0;
			 float sim_plane_res_roi = 0, sim_line_res_roi = 0;
			 //float non_zero_ratio_plane = 0;
			 for (size_t his = 0; his < pt_map_vec.size(); his++)
			 {
				 if (if_end)
				 {
					 break;
				}
				 float ratio_non_zero_plane_his = pt_map_vec[his]->m_ratio_nonzero_plane;
				 float ratio_non_zero_line_his = pt_map_vec[his]->m_ratio_nonzero_line;

				 if ((ratio_non_zero_plane_his < avail_ratio_plane) && (ratio_non_zero_line_his < avail_ratio_line))
					 continue;

				 if (abs(pt_map_vec[his]->m_roi_range - pt_cell_map_temp->m_roi_range) > 5.0)
					 continue;

				 sim_plane_res = pt_cell_map_temp->max_similiarity_of_two_image(pt_cell_map_temp->m_feature_img_plane, pt_map_vec[his]->m_feature_img_plane);
				 sim_line_res = pt_cell_map_temp->max_similiarity_of_two_image(pt_cell_map_temp->m_feature_img_line, pt_map_vec[his]->m_feature_img_line);

				 //if (idx - his > 100 && sim_line_res > 0.80 &&sim_plane_res > 0.95)
				 if ((pt_map_vec.size() - his > m_loop_closure_minimum_keyframe_differen) &&
					  (((sim_line_res > m_loop_closure_minimum_similarity_linear) && (sim_plane_res > 0.93)) ||
						  (sim_plane_res > m_loop_closure_minimum_similarity_planar)))
				 //if (his > 20 && sim_plane_res > 0.95)
				 //if (1)
				 {
					 //printf("%d -- %d, sim = %f , %f\r\n", idx, his, sim_line_res, sim_plane_res);
					 sim_plane_res_roi = pt_cell_map_temp->max_similiarity_of_two_image(pt_cell_map_temp->m_feature_img_plane_roi, pt_map_vec[his]->m_feature_img_plane_roi);
					 sim_line_res_roi = pt_cell_map_temp->max_similiarity_of_two_image(pt_cell_map_temp->m_feature_img_line_roi, pt_map_vec[his]->m_feature_img_line_roi);

					 if (((sim_line_res_roi > m_loop_closure_minimum_similarity_linear) && (sim_plane_res_roi > 0.93)) || (sim_plane_res_roi > m_loop_closure_minimum_similarity_planar))
					 {
						 screen_printf("Inlier loop detection wait ICP to find the transfrom\r\n");
						 fprintf(fp, "Inlier loop detection wait ICP to find the transfrom\r\n");
						 fflush(fp);
					}
					 else
					 {
						 continue;
					}

					 screen_printf("----------------------------\r\n");
					 screen_printf("%s -- %s\r\n", m_filename_vec[pt_map_vec.size() - 1].c_str(), m_filename_vec[his].c_str());
					 screen_printf("Nonzero_ratio %.3f , %.3f, %.3f, %.3f \r\n ", ratio_non_zero_plane, ratio_non_zero_line, ratio_non_zero_plane_his, ratio_non_zero_line_his);
					 screen_printf("Similarity = %.3f , %.3f, %.3f, %.3f\r\n", sim_line_res, sim_plane_res, sim_line_res_cv, sim_plane_res_cv);
					 screen_printf(" Roi similarity [%.2f, %.2f] = %.3f , %.3f \r\n", pt_cell_map_temp->m_roi_range, pt_map_vec[his]->m_roi_range, sim_line_res_roi, sim_plane_res_roi);

					 fprintf(fp, "----------------------------\r\n");
					 fprintf(fp, "%s -- %s\r\n", m_filename_vec[pt_map_vec.size() - 1].c_str(), m_filename_vec[his].c_str());
					 fprintf(fp, "Nonzero_ratio %.3f , %.3f, %.3f, %.3f \r\n ", ratio_non_zero_plane, ratio_non_zero_line, ratio_non_zero_plane_his, ratio_non_zero_line_his);
					 fprintf(fp, "Similarity = %.3f , %.3f, %.3f, %.3f\r\n", sim_line_res, sim_plane_res, sim_line_res_cv, sim_plane_res_cv);
					 fprintf(fp, "Roi similarity [%.2f, %.2f] = %.3f , %.3f \r\n", pt_cell_map_temp->m_roi_range, pt_map_vec[his]->m_roi_range, sim_line_res_roi, sim_plane_res_roi);
					 fflush(fp);
					 Points_cloud_map<float> *pt_cell_map_his = new Points_cloud_map<float>();
					 pt_cell_map_his->set_resolution(1.0);
					 pt_cell_map_his->set_point_cloud(pt_cell_map_his->load_pts_from_file(m_filename_vec[his]));
					 pt_cell_map_his->analyze_mapping(1);
					 //scene_align.set_downsample_resolution(0.1, 0.1);
					 scene_align.set_downsample_resolution(m_loop_closure_map_alignment_resolution, m_loop_closure_map_alignment_resolution);
					 double ICP_SCORE = scene_align.find_tranfrom_of_two_mappings(pt_cell_map_his, pt_cell_map_temp, 1);
					 //double ICP_SCORE = scene_align.find_tranfrom_of_two_mappings(  pt_cell_map_temp, pt_cell_map_his, 1);
					 pt_cell_map_his->clear_data();
					 delete pt_cell_map_his;
					 if (scene_align.pc_reg.m_inlier_final_threshold > 1.0)
					 {
						 his += 10;
					}

					 screen_printf("ICP inlier threshold = %lf, %lf\r\n", ICP_SCORE, scene_align.pc_reg.m_inlier_final_threshold);
					 screen_printf("%s\r\n", scene_align.pc_reg.m_final_opt_summary.BriefReport().c_str());

					 fprintf(fp, "ICP inlier threshold = %lf, %lf\r\n", ICP_SCORE, scene_align.pc_reg.m_inlier_final_threshold);
					 fprintf(fp, "%s\r\n", scene_align.pc_reg.m_final_opt_summary.BriefReport().c_str());
					 if (scene_align.pc_reg.m_inlier_final_threshold < m_loop_closure_map_alignment_inlier_threshold)
					 {
						 printf("I believe this is true loop.\r\n");
						 fprintf(fp, "I believe this is true loop.\r\n");
						 auto Q_a = pose3d_vec[his].q;
						 auto Q_b = pose3d_vec[pose3d_vec.size() - 1].q;
						 auto T_a = pose3d_vec[his].p;
						 auto T_b = pose3d_vec[pose3d_vec.size() - 1].p;
						 auto ICP_q = scene_align.pc_reg.m_q_w_curr;
						 auto ICP_t = scene_align.pc_reg.m_t_w_curr;
						 for (int i = 0; i < 10; i++)
						 {
							 screen_out << "-------------------------------------" << std::endl;
							 screen_out << ICP_q.coeffs().transpose() << std::endl;
							 screen_out << ICP_t.transpose() << std::endl;
						}
						 Ceres_pose_graph_3d::VectorOfConstraints                              constrain_vec_temp;
						 constrain_vec_temp = constrain_vec;
						 constrain_vec_temp.push_back(Scene_alignment<float>::add_constrain_of_loop(pose3d_vec.size() - 1, his, Q_a, T_a, Q_b, T_b, ICP_q, ICP_t));
						 std::string path_name = m_loop_save_dir_name;
						 std::string g2o_filename = std::string(path_name).append("/loop.g2o");
						 pose3d_map_ori = pose3d_map;
						 auto temp_pose_3d_map = pose3d_map;
						 Scene_alignment<float>::save_edge_and_vertex_to_g2o(g2o_filename.c_str(), temp_pose_3d_map, constrain_vec_temp);
						 Ceres_pose_graph_3d::pose_graph_optimization(temp_pose_3d_map, constrain_vec_temp);
						 Ceres_pose_graph_3d::OutputPoses(std::string(path_name).append("/poses_ori.txt"), pose3d_map_ori);
						 Ceres_pose_graph_3d::OutputPoses(std::string(path_name).append("/poses_opm.txt"), temp_pose_3d_map); 
						 scene_align.dump_file_name(std::string(path_name).append("/file_name.txt"), map_file_name);

						 loop_closure_pub_optimzed_path(temp_pose_3d_map);

						 for (int pc_idx = (int)map_id_pc.size() -1; pc_idx>=0; pc_idx-=2)
						 {
							  screen_out << "*** Refine pointcloud, curren idx = " << pc_idx << " ***" << endl;
							  auto refined_pt = map_rfn.refine_pointcloud(map_id_pc, pose3d_map_ori, temp_pose_3d_map, pc_idx, 0);
							  pcl::toROSMsg(refined_pt, ros_laser_cloud_surround);
							  ros_laser_cloud_surround.header.stamp = ros::Time::now();
							  ros_laser_cloud_surround.header.frame_id = "/camera_init";
							  m_pub_pc_aft_loop.publish(ros_laser_cloud_surround);
							  std::this_thread::sleep_for (std::chrono::milliseconds(10));
						}
						 //map_rfn.refine_mapping(path_name, 0);
						 if (0)
						 {
							 map_rfn.refine_mapping(map_id_pc, pose3d_map_ori, temp_pose_3d_map, 1);
							 pcl::toROSMsg(map_rfn.m_pts_aft_refind, ros_laser_cloud_surround);
							 ros_laser_cloud_surround.header.stamp = ros::Time::now();
							 ros_laser_cloud_surround.header.frame_id = "/camera_init";
							 m_pub_pc_aft_loop.publish(ros_laser_cloud_surround);
						}
						 if_end = 1;
						 break;
						 // TODO, add constrain.
					}
					 else
					 {
					   his+=5;
					}
					 if (if_end)
					 {
					   break;
					}
				}
				 if (if_end)
				 {
				   std::this_thread::sleep_for (std::chrono::milliseconds(500));
				   break;
				}
			}
			scene_align.dump_file_name(std::string(m_loop_save_dir_name).append("/file_name.txt"), map_file_name);
			pt_cell_map_temp->clear_data();
			fflush(fp);

			if (1)
			{
			  pcl::toROSMsg(pt_full, ros_laser_cloud_surround);
			  ros_laser_cloud_surround.header.stamp = ros::Time::now();
			  ros_laser_cloud_surround.header.frame_id = "/camera_init";
			  m_pub_debug_pts.publish(ros_laser_cloud_surround);
			}
			if (if_end)
			{
			  std::this_thread::sleep_for (std::chrono::milliseconds(500));
			  break;
			}
		}
	}

	void service_pub_surround_pts()
	{
		pcl::VoxelGrid<PointType> down_sample_filter_surface;
		down_sample_filter_surface.setLeafSize(m_surround_pointcloud_resolution, m_surround_pointcloud_resolution, m_surround_pointcloud_resolution);
		pcl::PointCloud<PointType>                           pc_temp;
		sensor_msgs::PointCloud2 ros_laser_cloud_surround;
		std::this_thread::sleep_for (std::chrono::nanoseconds(10));
		pcl::PointCloud<PointType>::Ptr                      laser_cloud_surround(new pcl::PointCloud<PointType>());
		laser_cloud_surround->reserve(1e8);
		std::this_thread::sleep_for (std::chrono::milliseconds(1000));
		int last_update_index = 0;
		while (1)
		{
			while (m_current_frame_index - last_update_index < 100)
			{
				std::this_thread::sleep_for (std::chrono::milliseconds(1500));
			}
			last_update_index = m_current_frame_index;
			pcl::PointCloud<PointType> pc_temp;
			laser_cloud_surround->clear();
			if (m_pt_cell_map_full.get_cells_size() == 0)
			  continue;
			std::vector<Points_cloud_map<float>::Mapping_cell *> cell_vec = m_pt_cell_map_full.find_cells_in_radius(m_t_w_curr, 1000.0);
			for (size_t i = 0; i < cell_vec.size(); i++)
			{
				if (m_down_sample_replace)
				{
					down_sample_filter_surface.setInputCloud(cell_vec[i]->get_pointcloud().makeShared());
					down_sample_filter_surface.filter(pc_temp);

					cell_vec[i]->set_pointcloud(pc_temp);

					*laser_cloud_surround += pc_temp;
				}
				else
				{
					*laser_cloud_surround += cell_vec[i]->get_pointcloud();
				}
			}
			if (laser_cloud_surround->points.size())
			{
				down_sample_filter_surface.setInputCloud(laser_cloud_surround);
				down_sample_filter_surface.filter(*laser_cloud_surround);
				pcl::toROSMsg(*laser_cloud_surround, ros_laser_cloud_surround);
				ros_laser_cloud_surround.header.stamp = ros::Time::now();
				ros_laser_cloud_surround.header.frame_id = "/camera_init";
				m_pub_laser_cloud_surround.publish(ros_laser_cloud_surround);
			}
			//screen_out << "~~~~~~~~~~~ " << "pub_surround_service, size = " << laser_cloud_surround->points.size()  << " ~~~~~~~~~~~" << endl;
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
		Eigen::Vector3d    t_wodom_curr;
		q_wodom_curr.x() = laserOdometry->pose.pose.orientation.x;
		q_wodom_curr.y() = laserOdometry->pose.pose.orientation.y;
		q_wodom_curr.z() = laserOdometry->pose.pose.orientation.z;
		q_wodom_curr.w() = laserOdometry->pose.pose.orientation.w;
		t_wodom_curr.x() = laserOdometry->pose.pose.position.x;
		t_wodom_curr.y() = laserOdometry->pose.pose.position.y;
		t_wodom_curr.z() = laserOdometry->pose.pose.position.z;

		Eigen::Quaterniond q_w_curr = Eigen::Quaterniond(1, 0, 0, 0);
		Eigen::Vector3d    t_w_curr = Eigen::Vector3d::Zero();

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
		m_pub_odom_aft_mapped_hight_frec.publish(odomAftMapped);
	}

	void find_min_max_intensity(const pcl::PointCloud<PointType>::Ptr pc_ptr, float &min_I, float &max_I)
	{
		int pt_size = pc_ptr->size();
		min_I = 10000;
		max_I = -min_I;
		for (int i = 0; i < pt_size; i++)
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

	void init_pointcloud_registration(Point_cloud_registration & pc_reg)
	{
		pc_reg.m_logger_common = &m_logger_common;
		pc_reg.m_logger_pcd = &m_logger_pcd;
		pc_reg.m_logger_timer = &m_logger_timer;
		pc_reg.m_timer = &m_timer;
		pc_reg.m_if_motion_deblur = if_motion_deblur;
		pc_reg.m_current_frame_index = m_current_frame_index;
		pc_reg.m_mapping_init_accumulate_frames = m_mapping_init_accumulate_frames;

		pc_reg.m_last_time_stamp = m_last_time_stamp;
		pc_reg.m_para_max_angular_rate = m_para_max_angular_rate;
		pc_reg.m_para_max_speed = m_para_max_speed;
		pc_reg.m_max_final_cost = m_max_final_cost;
		pc_reg.m_para_icp_max_iterations = m_para_icp_max_iterations;
		pc_reg.m_para_cere_max_iterations = m_para_cere_max_iterations;
		pc_reg.m_minimum_pt_time_stamp = m_minimum_pt_time_stamp;
		pc_reg.m_maximum_pt_time_stamp = m_maximum_pt_time_stamp;
		pc_reg.m_minimum_icp_R_diff = m_minimum_icp_R_diff;
		pc_reg.m_minimum_icp_T_diff = m_minimum_icp_T_diff;

		pc_reg.m_q_w_last = m_q_w_curr;
		pc_reg.m_t_w_last = m_t_w_curr;

		pc_reg.m_q_w_curr = m_q_w_curr;
		pc_reg.m_t_w_curr = m_t_w_curr;
	}

	int if_matchbuff_and_pc_sync(float point_cloud_current_timestamp)
	{
		if (m_lastest_pc_matching_refresh_time < 0)
			return 1;
		if (point_cloud_current_timestamp - m_lastest_pc_matching_refresh_time < m_maximum_pointcloud_delay_time)
			return 1;
		if (m_lastest_pc_reg_time == m_lastest_pc_matching_refresh_time)  // All is processed
			return 1;
		screen_printf("*** Current pointcloud timestamp = %.3f, lastest buff timestamp = %.3f, lastest_pc_reg_time = %.3f ***\r\n",
				point_cloud_current_timestamp,
				m_lastest_pc_matching_refresh_time,
				m_lastest_pc_reg_time);
		//cout << "~~~~~~~~~~~~~~~~ Wait sync, " << point_cloud_current_timestamp << ", " << m_lastest_pc_matching_refresh_time << endl;

		return 0;
	}

	int process_new_scan()
	{
		m_timer.tic("Frame process");
		m_timer.tic("Query points for match");

		Common_tools::Timer timer_frame;
		timer_frame.tic();
		pcl::PointCloud<PointType> current_laser_cloud_full, current_laser_cloud_corner, current_laser_cloud_surf;

		pcl::VoxelGrid<PointType> down_sample_filter_corner = m_down_sample_filter_corner;
		pcl::VoxelGrid<PointType> down_sample_filter_surface = m_down_sample_filter_surface;
		pcl::KdTreeFLANN<PointType> kdtree_corner_from_map;
		pcl::KdTreeFLANN<PointType> kdtree_surf_from_map;

		mutex_querypointcloud.lock();
		current_laser_cloud_full = *laser_cloud_full_cur;
		current_laser_cloud_corner = *laser_cloud_corner_cur;
		current_laser_cloud_surf = *laser_cloud_surf_cur;

		float min_t, max_t;
		find_min_max_intensity(current_laser_cloud_full.makeShared(), min_t, max_t);

		double point_cloud_current_timestamp = min_t;
		if (point_cloud_current_timestamp > m_lastest_pc_income_time)
			m_lastest_pc_income_time = point_cloud_current_timestamp;

		point_cloud_current_timestamp = m_lastest_pc_income_time;
		m_time_odom = m_last_time_stamp;
		m_minimum_pt_time_stamp = m_last_time_stamp;
		m_maximum_pt_time_stamp = max_t;
		m_last_time_stamp = max_t;
		Point_cloud_registration pc_reg;
		init_pointcloud_registration(pc_reg);
		m_current_frame_index++;
		double time_odom = ros::Time::now().toSec();
		mutex_querypointcloud.unlock();

		screen_printf("****** Before timestamp info = [%.6f, %.6f, %.6f, %.6f] ****** \r\n", m_minimum_pt_time_stamp, m_maximum_pt_time_stamp, min_t, m_lastest_pc_matching_refresh_time);

		m_timer.tic("Wait sync");
		while(!if_matchbuff_and_pc_sync(point_cloud_current_timestamp))
			std::this_thread::sleep_for (std::chrono::milliseconds(1));

		*(m_logger_timer.get_ostream()) << m_timer.toc_string("Wait sync") << std::endl;
		screen_printf("****** After timestamp info = [%.6f, %.6f, %.6f, %.6f] ****** \r\n", m_minimum_pt_time_stamp, m_maximum_pt_time_stamp, min_t, m_lastest_pc_matching_refresh_time);

		pcl::PointCloud<PointType>::Ptr laserCloudCornerStack(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr laserCloudSurfStack(new pcl::PointCloud<PointType>());

		if (m_if_input_downsample_mode)
		{
			down_sample_filter_corner.setInputCloud(current_laser_cloud_corner.makeShared());
			down_sample_filter_corner.filter(*laserCloudCornerStack);
			down_sample_filter_surface.setInputCloud(current_laser_cloud_surf.makeShared());
			down_sample_filter_surface.filter(*laserCloudSurfStack);
		}
		else
		{
			*laserCloudCornerStack = current_laser_cloud_corner;
			*laserCloudSurfStack = current_laser_cloud_surf;
		}

		int laser_corner_pt_num = laserCloudCornerStack->points.size();
		int laser_surface_pt_num = laserCloudSurfStack->points.size();


		if (m_if_save_to_pcd_files && PCD_SAVE_RAW)
			m_pcl_tools_raw.save_to_pcd_files("raw", current_laser_cloud_full, m_current_frame_index);

		m_q_w_last = m_q_w_curr;
		m_t_w_last = m_t_w_curr;

		pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map(new pcl::PointCloud<PointType>());
		int reg_res = 0;

		mutex_buff_for_matching_corner.lock();
		*laser_cloud_corner_from_map = *laser_cloud_corner_from_map_pre;
		kdtree_corner_from_map = kdtree_corner_from_map_pre;
		mutex_buff_for_matching_surface.unlock();

		mutex_buff_for_matching_surface.lock();
		*laser_cloud_surf_from_map = *laser_cloud_surf_from_map_pre;
		kdtree_surf_from_map = kdtree_surf_from_map_pre;
		mutex_buff_for_matching_corner.unlock();

		reg_res = pc_reg.find_out_incremental_transfrom(laser_cloud_corner_from_map, laser_cloud_surf_from_map,
														kdtree_corner_from_map, kdtree_surf_from_map,
														laserCloudCornerStack, laserCloudSurfStack);
														 
		screen_out << "Input points size = " << laser_corner_pt_num << ", surface size = " << laser_surface_pt_num << endl;
		screen_out << "Input mapping points size = " << laser_cloud_corner_from_map->points.size() <<
				", surface size = " << laser_cloud_surf_from_map->points.size() << endl;
		screen_out << "Registration res = " << reg_res << endl;

		if (reg_res == 0)
			return 0;

		m_timer.tic("Add new frame");

		PointType pointOri, pointSel;

		pcl::PointCloud<PointType>::Ptr map_new_feature_corners(new pcl::PointCloud<PointType>());
		pcl::PointCloud<PointType>::Ptr map_new_feature_surface(new pcl::PointCloud<PointType>());

		for (int i = 0; i < laser_corner_pt_num; i++)
		{
			pc_reg.pointAssociateToMap(&laserCloudCornerStack->points[i],
									   &pointSel,
									   refine_blur(laserCloudCornerStack->points[i].intensity, m_minimum_pt_time_stamp, m_maximum_pt_time_stamp),
									   g_if_undistore);
			map_new_feature_corners->push_back(pointSel);
		}

		for (int i = 0; i < laser_surface_pt_num; i++)
		{
			pc_reg.pointAssociateToMap(&laserCloudSurfStack->points[i],
									   &pointSel,
									   refine_blur(laserCloudSurfStack->points[i].intensity, m_minimum_pt_time_stamp, m_maximum_pt_time_stamp),
									   g_if_undistore);
			map_new_feature_surface->push_back(pointSel);
		}

		
		down_sample_filter_corner.setInputCloud(map_new_feature_corners);
		down_sample_filter_corner.filter(*map_new_feature_corners);
		down_sample_filter_surface.setInputCloud(map_new_feature_surface);
		down_sample_filter_surface.filter(*map_new_feature_surface);

		double r_diff = m_q_w_curr.angularDistance(m_last_his_add_q) * 57.3;
		double t_diff = (m_t_w_curr - m_last_his_add_t).norm();

		pc_reg.pointcloudAssociateToMap(current_laser_cloud_full, current_laser_cloud_full, g_if_undistore);

		mutex_mapping.lock();

		if (map_corner_history.size() < (size_t)m_maximum_history_size ||
			(t_diff > history_add_t_step) || (r_diff > history_add_angle_step * 57.3))
		{
			m_last_his_add_q = m_q_w_curr;
			m_last_his_add_t = m_t_w_curr;

			map_corner_history.push_back(*map_new_feature_corners);
			map_surface_history.push_back(*map_new_feature_surface);
			mutex_dump_full_history.lock();
			m_laser_cloud_full_history.push_back(current_laser_cloud_full);
			m_his_reg_error.push_back(pc_reg.m_inlier_final_threshold);
			mutex_dump_full_history.unlock();
		}
		else
			screen_printf("==== Reject add history, T_norm = %.2f, R_norm = %.2f ====\r\n", t_diff, r_diff);

		screen_out << "m_pt_cell_map_corners.size() = " << m_pt_cell_map_corners.get_cells_size() << endl;
		screen_out << "m_pt_cell_map_planes.size() = " << m_pt_cell_map_planes.get_cells_size() << endl;

		if (map_corner_history.size() > (size_t)m_maximum_history_size)
		{
			(map_corner_history.front()).clear();
			map_corner_history.pop_front();
		}

		if (map_surface_history.size() > (size_t)m_maximum_history_size)
		{
			(map_surface_history.front()).clear();
			map_surface_history.pop_front();
		}

		if (m_laser_cloud_full_history.size() > (size_t)m_maximum_history_size)
		{
			mutex_dump_full_history.lock();
			(m_laser_cloud_full_history.front()).clear();
			m_laser_cloud_full_history.pop_front();
			m_his_reg_error.pop_front();
			mutex_dump_full_history.unlock();
		}

		m_if_mapping_updated_corner = true;
		m_if_mapping_updated_surface = true;

		m_pt_cell_map_corners.append_cloud(PCL_TOOLS::pcl_pts_to_eigen_pts<float, PointType>(map_new_feature_corners));
		m_pt_cell_map_planes.append_cloud(PCL_TOOLS::pcl_pts_to_eigen_pts<float, PointType>(map_new_feature_surface));

		*(m_logger_common.get_ostream()) << "New added regtime "<< point_cloud_current_timestamp << endl;
		if ((m_lastest_pc_reg_time < point_cloud_current_timestamp) || (point_cloud_current_timestamp < 10.0))
		{
			m_q_w_curr = pc_reg.m_q_w_curr;
			m_t_w_curr = pc_reg.m_t_w_curr;
			m_lastest_pc_reg_time = point_cloud_current_timestamp;
		}
		else
			*(m_logger_common.get_ostream()) << "***** older update, reject update pose *****" << endl;

		*(m_logger_pcd.get_ostream()) << "--------------------" << endl;
		m_logger_pcd.printf("Curr_Q = %f,%f,%f,%f\r\n", m_q_w_curr.w(), m_q_w_curr.x(), m_q_w_curr.y(), m_q_w_curr.z());
		m_logger_pcd.printf("Curr_T = %f,%f,%f\r\n", m_t_w_curr(0), m_t_w_curr(1), m_t_w_curr(2));
		m_logger_pcd.printf("Incre_Q = %f,%f,%f,%f\r\n", pc_reg.m_q_w_incre.w(), pc_reg.m_q_w_incre.x(), pc_reg.m_q_w_incre.y(), pc_reg.m_q_w_incre.z());
		m_logger_pcd.printf("Incre_T = %f,%f,%f\r\n", pc_reg.m_t_w_incre(0), pc_reg.m_t_w_incre(1), pc_reg.m_t_w_incre(2));
		m_logger_pcd.printf("Cost=%f,blk_size = %d \r\n", m_final_opt_summary.final_cost, m_final_opt_summary.num_residual_blocks);
		*(m_logger_pcd.get_ostream()) << m_final_opt_summary.BriefReport() << endl;

		mutex_mapping.unlock();

		timer_log_mutex.lock();
		timer_log_mutex.unlock();

		if (m_thread_match_buff_refresh.size() < (size_t)m_maximum_mapping_buff_thread)
		{
			std::future<void> *m_mapping_refresh_service =
				new std::future<void>(std::async(std::launch::async, &Laser_mapping::service_update_buff_for_matching, this));

			m_thread_match_buff_refresh.push_back(m_mapping_refresh_service);
		}

		m_pt_cell_map_full.append_cloud(PCL_TOOLS::pcl_pts_to_eigen_pts<float, PointType>(current_laser_cloud_full.makeShared()));

		
		mutex_ros_pub.lock();
		sensor_msgs::PointCloud2 laserCloudFullRes3;
		pcl::toROSMsg(current_laser_cloud_full, laserCloudFullRes3);
		laserCloudFullRes3.header.stamp = ros::Time().fromSec(time_odom);
		laserCloudFullRes3.header.frame_id = "/camera_init";
		m_pub_laser_cloud_full_res.publish(laserCloudFullRes3); //single_frame_with_pose_tranfromed

		//publish surround map for every 5 frame
		if (PUB_DEBUG_INFO)
		{
			pcl::PointCloud<PointType> pc_feature_pub_corners, pc_feature_pub_surface;
			sensor_msgs::PointCloud2   laserCloudMsg;

			pc_reg.pointcloudAssociateToMap(current_laser_cloud_surf, pc_feature_pub_surface, g_if_undistore);
			pcl::toROSMsg(pc_feature_pub_surface, laserCloudMsg);
			laserCloudMsg.header.stamp = ros::Time().fromSec(time_odom);
			laserCloudMsg.header.frame_id = "/camera_init";
			m_pub_last_surface_pts.publish(laserCloudMsg);
			pc_reg.pointcloudAssociateToMap(current_laser_cloud_corner, pc_feature_pub_corners, g_if_undistore);
			pcl::toROSMsg(pc_feature_pub_corners, laserCloudMsg);
			laserCloudMsg.header.stamp = ros::Time().fromSec(time_odom);
			laserCloudMsg.header.frame_id = "/camera_init";
			m_pub_last_corner_pts.publish(laserCloudMsg);
		}

		sensor_msgs::PointCloud2   laserCloudMsg;
		pcl::toROSMsg(*laser_cloud_surf_from_map, laserCloudMsg);
		laserCloudMsg.header.stamp = ros::Time().fromSec(time_odom);
		laserCloudMsg.header.frame_id = "/camera_init";
		m_pub_match_surface_pts.publish(laserCloudMsg);

		pcl::toROSMsg(*laser_cloud_corner_from_map, laserCloudMsg);
		laserCloudMsg.header.stamp = ros::Time().fromSec(time_odom);
		laserCloudMsg.header.frame_id = "/camera_init";
		m_pub_match_corner_pts.publish(laserCloudMsg);

		if (m_if_save_to_pcd_files)
		{
			m_pcl_tools_aftmap.save_to_pcd_files("aft_mapp", current_laser_cloud_full, m_current_frame_index);
			*(m_logger_pcd.get_ostream()) << "Save to: " << m_pcl_tools_aftmap.m_save_file_name << endl;
		}

		
		nav_msgs::Odometry odomAftMapped;
		odomAftMapped.header.frame_id = "/camera_init";
		odomAftMapped.child_frame_id = "/aft_mapped";
		odomAftMapped.header.stamp = ros::Time().fromSec(time_odom);

		odomAftMapped.pose.pose.orientation.x = m_q_w_curr.x();
		odomAftMapped.pose.pose.orientation.y = m_q_w_curr.y();
		odomAftMapped.pose.pose.orientation.z = m_q_w_curr.z();
		odomAftMapped.pose.pose.orientation.w = m_q_w_curr.w();

		odomAftMapped.pose.pose.position.x = m_t_w_curr.x();
		odomAftMapped.pose.pose.position.y = m_t_w_curr.y();
		odomAftMapped.pose.pose.position.z = m_t_w_curr.z();

		m_pub_odom_aft_mapped.publish(odomAftMapped); // name: Odometry aft_mapped_to_init

		geometry_msgs::PoseStamped pose_aft_mapped;
		pose_aft_mapped.header = odomAftMapped.header;
		pose_aft_mapped.pose = odomAftMapped.pose.pose;
		m_laser_after_mapped_path.header.stamp = odomAftMapped.header.stamp;
		m_laser_after_mapped_path.header.frame_id = "/camera_init";
		if (m_current_frame_index % 10 == 0)
		{
			m_laser_after_mapped_path.poses.push_back(pose_aft_mapped);
			m_pub_laser_aft_mapped_path.publish(m_laser_after_mapped_path);
		}


		static tf::TransformBroadcaster br;
		tf::Transform                   transform;
		tf::Quaternion                  q;
		transform.setOrigin(tf::Vector3(m_t_w_curr(0),
										  m_t_w_curr(1),
										  m_t_w_curr(2)));

		
		q.setW(m_q_w_curr.w());
		q.setX(m_q_w_curr.x());
		q.setY(m_q_w_curr.y());
		q.setZ(m_q_w_curr.z());
		transform.setRotation(q);
		br.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "/camera_init", "/aft_mapped"));

		mutex_ros_pub.unlock();
		*(m_logger_timer.get_ostream()) << m_timer.toc_string("Add new frame") << std::endl;
		*(m_logger_timer.get_ostream()) << m_timer.toc_string("Frame process") << std::endl;
		
		return 1;
	}

	void process()
	{
		double first_time_stamp = -1;
		m_last_max_blur = 0.0;

		m_service_pub_surround_pts = new std::future<void>(std::async(std::launch::async, &Laser_mapping::service_pub_surround_pts, this));

		if (m_loop_closure_if_enable)
			m_service_loop_detection = new std::future<void>(std::async(std::launch::async, &Laser_mapping::service_loop_detection, this));

		timer_all.tic();

		while (1)
		{
			m_logger_common.printf("------------------\r\n");
			m_logger_timer.printf("------------------\r\n");

			while (m_queue_avail_data.empty())
				sleep(0.0001);

			mutex_buf.lock();
			while (m_queue_avail_data.size() >= (unsigned int) m_max_buffer_size)
			{
				ROS_WARN("Drop lidar frame in mapping for real time performance !!!");
				(*m_logger_common.get_ostream()) << "Drop lidar frame in mapping for real time performance !!!" << endl;
				m_queue_avail_data.pop();
			}

			Data_pair *current_data_pair = m_queue_avail_data.front();
			m_queue_avail_data.pop();
			mutex_buf.unlock();


			m_timer.tic("Prepare to enter thread");

			m_time_pc_corner_past = current_data_pair->m_pc_corner->header.stamp.toSec();

			if (first_time_stamp < 0)
				first_time_stamp = m_time_pc_corner_past;

			(*m_logger_common.get_ostream()) << "Messgage time stamp = " << m_time_pc_corner_past - first_time_stamp << endl;

			mutex_querypointcloud.lock();
			laser_cloud_corner_cur->clear();
			pcl::fromROSMsg(*(current_data_pair->m_pc_corner), *laser_cloud_corner_cur);

			laser_cloud_surf_cur->clear();
			pcl::fromROSMsg(*(current_data_pair->m_pc_plane), *laser_cloud_surf_cur);

			laser_cloud_full_cur->clear();
			pcl::fromROSMsg(*(current_data_pair->m_pc_full), *laser_cloud_full_cur);
			mutex_querypointcloud.unlock();

			delete current_data_pair;

			Common_tools::maintain_maximum_thread_pool<std::future<int>*>(m_thread_pool, m_maximum_parallel_thread);

			std::future<int>* thd = new std::future<int>(std::async(std::launch::async, &Laser_mapping::process_new_scan, this));

			*(m_logger_timer.get_ostream())<< m_timer.toc_string("Prepare to enter thread") << std::endl;

			m_thread_pool.push_back(thd);

			std::this_thread::sleep_for (std::chrono::nanoseconds(10));
			
		}

	}
};

#endif // LASER_MAPPING_HPP
