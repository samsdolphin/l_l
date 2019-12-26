// Author: Jiarong Lin          ziv.lin.ljr@gmail.com
// Modified: Xiyuan Liu         liuxiyuan95@gmail.com

#ifndef POINT_CLOUD_REGISTRATION_HPP
#define POINT_CLOUD_REGISTRATION_HPP
#include <ceres/ceres.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#include <mutex>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <future>

#include "ceres_icp.hpp"
#include "common.h"
#include <opencv/cv.h>
#include <sophus/se3.hpp>
#include "tools_logger.hpp"
#include "pcl_tools.hpp"
#include "tools_timer.hpp"
#include "cell_map_keyframe.hpp"
#include "utils_math.hpp"

#define CORNER_MIN_MAP_NUM 0
#define SURFACE_MIN_MAP_NUM 50
#define BLUR_SCALE 1.0
#define USE_SIZED_COST 1
#define USE_SELF_LM 1

using namespace PCL_TOOLS;
using namespace Common_tools;

//class Laser_mapping;
int g_export_full_count = 0;
class Point_cloud_registration
{
public:

    std::string path_name = "/home/sam/Loam_livox";
	std::string file_name = std::string(path_name).append("/test");
    //ceres::LinearSolverType slover_type = ceres::SPARSE_NORMAL_CHOLESKY; 
    ceres::LinearSolverType slover_type = ceres::DENSE_SCHUR; // SPARSE_NORMAL_CHOLESKY | DENSE_QR | DENSE_SCHUR

    int line_search_num = 5;
    int IF_LINE_FEATURE_CHECK = 1;
    int plane_search_num = 5;
    int IF_PLANE_FEATURE_CHECK = 1;
    int ICP_PLANE = 1;
    int ICP_LINE = 1;
    double m_para_buffer_RT[7] = { 0, 0, 0, 1, 0, 0, 0 };
    double m_para_buffer_RT_last[7] = { 0, 0, 0, 1, 0, 0, 0 };
    double buff_incre[7] = {0, 0, 0, 1, 0, 0, 0};
    double para_buff[6] = {0, 0, 0, 0, 0, 0};

    Eigen::Map<Eigen::Quaterniond> m_q_w_incre = Eigen::Map<Eigen::Quaterniond>(buff_incre);
    Eigen::Map<Eigen::Vector3d> m_t_w_incre = Eigen::Map<Eigen::Vector3d>(buff_incre + 4);

    double m_interpolatation_theta;

    int m_if_motion_deblur = 0 ;

    double m_angular_diff = 0;
    double m_t_diff = 0;
    double m_maximum_dis_plane_for_match = 50.0;
    double m_maximum_dis_line_for_match = 2.0;
    Eigen::Matrix<double, 3, 1> m_interpolatation_omega;
    Eigen::Matrix<double, 3, 3> m_interpolatation_omega_hat;
    Eigen::Matrix<double, 3, 3> m_interpolatation_omega_hat_sq2;

    const Eigen::Quaterniond m_q_I = Eigen::Quaterniond(1, 0, 0, 0);

    pcl::KdTreeFLANN<PointType> m_kdtree_corner_from_map;
    pcl::KdTreeFLANN<PointType> m_kdtree_surf_from_map;

    Eigen::Quaterniond m_q_w_curr, m_q_w_last;
    Eigen::Vector3d m_t_w_curr, m_t_w_last;

    Common_tools::File_logger *m_logger_common;
    Common_tools::File_logger *m_logger_pcd;
    Common_tools::File_logger *m_logger_timer;

    Common_tools::Timer *m_timer;
    int    m_current_frame_index;
    int    m_mapping_init_accumulate_frames = 100;
    float  m_last_time_stamp = 0;
    float  m_para_max_angular_rate = 200.0 / 50.0; // max angular rate = 90.0 /50.0 deg/s
    float  m_para_max_speed = 100.0 / 50.0;        // max speed = 10 m/s
    float  m_max_final_cost = 3.0;
    int    m_para_icp_max_iterations = 20;
    int    m_para_cere_max_iterations = 100;
    int    m_para_cere_prerun_times = 2;
    float  m_minimum_pt_time_stamp = 0;
    float  m_maximum_pt_time_stamp = 1.0;
    double m_minimum_icp_R_diff = 0.01;
    double m_minimum_icp_T_diff = 0.01;

    double m_inliner_dis = 0.02;
    double m_inlier_ratio = 0.80;

    double                               m_inlier_threshold;
    ceres::Solver::Summary               summary;
    ceres::Solver::Summary               m_final_opt_summary;
    int                                  m_maximum_allow_residual_block = 1e5;
    Common_tools::Random_generator_float<float> m_rand_float;
    ~Point_cloud_registration()
    {
        
    };

    Point_cloud_registration()
    {
        m_q_w_last.setIdentity();
        m_t_w_last.setZero();
        m_q_w_curr.setIdentity();
        m_t_w_curr.setZero();
        m_if_verbose_screen_printf = 1;
    };

    ADD_SCREEN_PRINTF_OUT_METHOD;

    void reset_incremtal_parameter()
    {
        m_interpolatation_theta = 0;
        m_interpolatation_omega_hat.setZero();
        m_interpolatation_omega_hat_sq2.setZero();
    };

    float refine_blur(float in_blur, const float &min_blur, const float &max_blur)
    {
        float res = 1.0;
        if (m_if_motion_deblur)
        {
            res = (in_blur - min_blur) / (max_blur - min_blur);
            if (!std::isfinite(res) || res > 1.0)
                return 1.0;
            else
                return res;
        }

        return res;
    };

    void set_ceres_solver_bound(ceres::Problem &problem, double* para_buffer_RT)
    {
        for (unsigned int i = 0; i < 3; i++)
        {
            if (USE_SIZED_COST)
            {
                problem.SetParameterLowerBound(para_buffer_RT + 3, i, -m_para_max_speed);
                problem.SetParameterUpperBound(para_buffer_RT + 3, i, +m_para_max_speed);
            }
            else
            {
                problem.SetParameterLowerBound(para_buffer_RT + 4, i, -m_para_max_speed);
                problem.SetParameterUpperBound(para_buffer_RT + 4, i, +m_para_max_speed);
                
            }
        }
    };

    double compute_inlier_residual_threshold(std::vector< double > residuals, double ratio)
    {
        std::set< double > dis_vec;
        for (size_t i = 0; i < (size_t)(residuals.size() / 3); i++)
            dis_vec.insert(fabs(residuals[3 * i + 0]) + fabs(residuals[3 * i + 1]) + fabs(residuals[3 * i + 2]));

        return *(std::next(dis_vec.begin(), (int) ((ratio) * dis_vec.size())));
    }

    int find_out_incremental_transfrom(pcl::PointCloud<PointType>::Ptr in_laser_cloud_corner_from_map,
                                       pcl::PointCloud<PointType>::Ptr in_laser_cloud_surf_from_map,
                                       pcl::KdTreeFLANN<PointType>& kdtree_corner_from_map,
                                       pcl::KdTreeFLANN<PointType>& kdtree_surf_from_map,
                                       pcl::PointCloud<PointType>::Ptr laserCloudCornerStack,
                                       pcl::PointCloud<PointType>::Ptr laserCloudSurfStack)
    {
        Eigen::Map<Eigen::Quaterniond> q_w_incre = Eigen::Map<Eigen::Quaterniond>(buff_incre);
        Eigen::Map<Eigen::Vector3d> t_w_incre = Eigen::Map<Eigen::Vector3d>(buff_incre + 4);
        Eigen::Map<Eigen::Vector3d> a_incre = Eigen::Map<Eigen::Vector3d>(para_buff);
        Eigen::Map<Eigen::Vector3d> t_incre = Eigen::Map<Eigen::Vector3d>(para_buff + 3);

        m_kdtree_corner_from_map = kdtree_corner_from_map;
        m_kdtree_surf_from_map = kdtree_surf_from_map;

        pcl::PointCloud<PointType> laser_cloud_corner_from_map =  *in_laser_cloud_corner_from_map;
        pcl::PointCloud<PointType> laser_cloud_surf_from_map =  *in_laser_cloud_surf_from_map;

        int laserCloudCornerFromMapNum = laser_cloud_corner_from_map.points.size();
        int laserCloudSurfFromMapNum = laser_cloud_surf_from_map.points.size();
        int laser_corner_pt_num = laserCloudCornerStack->points.size();
        int laser_surface_pt_num = laserCloudSurfStack->points.size();

        *(m_logger_timer->get_ostream())<< m_timer->toc_string("Query points for match") << std::endl;
        m_timer->tic("Pose optimization");

        int                    surf_avail_num = 0;
        int                    corner_avail_num = 0;
        float                  minimize_cost = summary.final_cost ;
        PointType              pointOri, pointSel;
        int                    corner_rejection_num = 0;
        int                    surface_rejecetion_num = 0;
        int                    if_undistore_in_matching = 1;

        if (laserCloudCornerFromMapNum > CORNER_MIN_MAP_NUM &&
            laserCloudSurfFromMapNum > SURFACE_MIN_MAP_NUM &&
            m_current_frame_index > m_mapping_init_accumulate_frames)
        {
            FILE *fp = fopen(file_name.c_str(), "a+");
			if (fp == NULL)
				cout << "Open file name " << file_name << " error, please check" << endl;

            m_timer->tic("Build kdtree");
            cout<<"FRAME INDEX "<<m_current_frame_index<<endl;

            *(m_logger_timer->get_ostream()) << m_timer->toc_string("Build kdtree") << std::endl;
            Eigen::Quaterniond q_last_optimize(1.f, 0.f, 0.f, 0.f);
            Eigen::Vector3d    t_last_optimize(0.f, 0.f, 0.f);
            int                iterCount = 0;

            std::vector<int>   m_point_search_Idx;
            std::vector<float> m_point_search_sq_dis;

            for (iterCount = 0; iterCount < m_para_icp_max_iterations; iterCount++)
            {
                m_point_search_Idx.clear();
                m_point_search_sq_dis.clear();
                corner_avail_num = 0;
                surf_avail_num = 0;
                corner_rejection_num = 0;
                surface_rejecetion_num = 0;

                std::vector<Eigen::Vector3d> line, plane, curPt, tarA;
                std::vector<Eigen::Vector3d> line_, plane_, curPt_, tarA_;
                bool remove_ = false;

                ceres::LossFunction* loss_function = new ceres::HuberLoss(0.1);
                ceres::LocalParameterization* q_parameterization = new ceres::EigenQuaternionParameterization();
                ceres::Problem::Options problem_options;
                ceres::ResidualBlockId block_id;
                ceres::Problem problem(problem_options);
                std::vector<ceres::ResidualBlockId> residual_block_ids;

                if (USE_SIZED_COST)
                {
                    problem.AddParameterBlock(para_buff, 3, new LineParameterization());
                    problem.AddParameterBlock(para_buff + 3, 3);
                }
                else
                {
                    problem.AddParameterBlock(buff_incre, 4, q_parameterization);
                    problem.AddParameterBlock(buff_incre + 4, 3);
                }

                for (int i = 0; i < laser_corner_pt_num; i++)
                {
                    if (laser_corner_pt_num > 2 * m_maximum_allow_residual_block)
                        if (m_rand_float.rand_uniform() * laser_corner_pt_num >  2 * m_maximum_allow_residual_block)
                            continue;

                    pointOri = laserCloudCornerStack->points[i];

                    if ((!std::isfinite(pointOri.x)) ||
                        (!std::isfinite(pointOri.y)) ||
                        (!std::isfinite(pointOri.z)))
                        continue;

                    pointAssociateToMap(&pointOri,
                                        &pointSel,
                                        refine_blur(pointOri.intensity, m_minimum_pt_time_stamp, m_maximum_pt_time_stamp),
                                        if_undistore_in_matching);

                    if (m_kdtree_corner_from_map.nearestKSearch(pointSel, line_search_num, m_point_search_Idx, m_point_search_sq_dis) != line_search_num)
                        continue;

                    if (m_point_search_sq_dis[line_search_num - 1] < m_maximum_dis_line_for_match)
                    {
                        bool line_is_avail = true;
                        std::vector<Eigen::Vector3d> nearCorners;
                        Eigen::Vector3d center(0, 0, 0);
                        if (IF_LINE_FEATURE_CHECK)
                        {
                            for (int j = 0; j < line_search_num; j++)
                            {
                                Eigen::Vector3d tmp(laser_cloud_corner_from_map.points[m_point_search_Idx[j]].x,
                                                    laser_cloud_corner_from_map.points[m_point_search_Idx[j]].y,
                                                    laser_cloud_corner_from_map.points[m_point_search_Idx[j]].z);
                                center = center + tmp;
                                nearCorners.push_back(tmp);
                            }

                            center = center / ((float) line_search_num);

                            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();

                            for (int j = 0; j < line_search_num; j++)
                            {
                                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                            }

                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                            // note Eigen library sort eigenvalues in increasing order

                            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
                                line_is_avail = true;
                            else
                                line_is_avail = false;
                        }

                        Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);
                        if (line_is_avail)
                        {
                            if (ICP_LINE)
                            {
                                ceres::CostFunction *cost_function;
                                auto pt_1 = pcl_pt_to_eigend(laser_cloud_corner_from_map.points[m_point_search_Idx[0]]);
                                auto pt_2 = pcl_pt_to_eigend(laser_cloud_corner_from_map.points[m_point_search_Idx[1]]);
                                if ((pt_1 - pt_2).norm() < 0.0001)
                                    continue;

                                if (USE_SIZED_COST)
                                {
                                    cost_function = new QPoint2Line(
                                        curr_point,
                                        pt_1,
                                        pt_2,
                                        Eigen::Matrix<double, 4, 1>(m_q_w_last.w(), m_q_w_last.x(), m_q_w_last.y(), m_q_w_last.z()),
                                        m_t_w_last);
                                    block_id = problem.AddResidualBlock(cost_function, loss_function, para_buff, para_buff + 3);
                                }
                                else
                                {
                                    cost_function = quaternion_point2line<double>::Create(
                                        curr_point,
                                        pt_1,
                                        pt_2,
                                        Eigen::Matrix<double, 4, 1>(m_q_w_last.w(), m_q_w_last.x(), m_q_w_last.y(), m_q_w_last.z()),
                                        m_t_w_last);
                                    block_id = problem.AddResidualBlock(cost_function, loss_function, buff_incre, buff_incre + 4);
                                }
                                line.push_back(pt_2 - pt_1);
                                curPt.push_back(curr_point);
                                tarA.push_back(pt_1);
                                residual_block_ids.push_back(block_id);
                                corner_avail_num++;
                            }
                        }
                        else
                            corner_rejection_num++;
                    }
                }

                for (int i = 0; i < laser_surface_pt_num; i++)
                {
                    if (laser_surface_pt_num > 2 * m_maximum_allow_residual_block)
                        if (m_rand_float.rand_uniform() * laser_surface_pt_num >  2 * m_maximum_allow_residual_block)
                            continue;

                    pointOri = laserCloudSurfStack->points[i];
                    int planeValid = true;
                    pointAssociateToMap(&pointOri,
                                        &pointSel,
                                        refine_blur(pointOri.intensity, m_minimum_pt_time_stamp, m_maximum_pt_time_stamp),
                                        if_undistore_in_matching);
                    m_kdtree_surf_from_map.nearestKSearch(pointSel, plane_search_num, m_point_search_Idx, m_point_search_sq_dis);

                    if (m_point_search_sq_dis[plane_search_num - 1] < m_maximum_dis_plane_for_match)
                    {
                        std::vector<Eigen::Vector3d> nearCorners;
                        Eigen::Vector3d center(0, 0, 0);
                        if (IF_PLANE_FEATURE_CHECK)
                        {
                            for (int j = 0; j < plane_search_num; j++)
                            {
                                Eigen::Vector3d tmp(laser_cloud_corner_from_map.points[m_point_search_Idx[j]].x,
                                                    laser_cloud_corner_from_map.points[m_point_search_Idx[j]].y,
                                                    laser_cloud_corner_from_map.points[m_point_search_Idx[j]].z);
                                center = center + tmp;
                                nearCorners.push_back(tmp);
                            }

                            center = center / (float) (plane_search_num);

                            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();

                            for (int j = 0; j < plane_search_num; j++)
                            {
                                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                            }

                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                            if ((saes.eigenvalues()[2] > 3 * saes.eigenvalues()[0]) &&
                                (saes.eigenvalues()[2] < 10 * saes.eigenvalues()[1]))
                                planeValid = true;
                            else
                                planeValid = false;
                        }

                        Eigen::Vector3d curr_point(pointOri.x, pointOri.y, pointOri.z);

                        if (planeValid)
                        {
                            if (ICP_PLANE)
                            {
                                ceres::CostFunction *cost_function;
                                auto pt_1 = pcl_pt_to_eigend(laser_cloud_surf_from_map.points[m_point_search_Idx[0]]);
                                auto pt_2 = pcl_pt_to_eigend(laser_cloud_surf_from_map.points[m_point_search_Idx[plane_search_num / 2]]);
                                auto pt_3 = pcl_pt_to_eigend(laser_cloud_surf_from_map.points[m_point_search_Idx[plane_search_num - 1]]);

                                if (USE_SIZED_COST)
                                {
                                    cost_function = new QPoint2Plane(
                                        curr_point,
                                        pt_1,
                                        pt_2,
                                        pt_3,
                                        Eigen::Matrix<double, 4, 1>(m_q_w_last.w(), m_q_w_last.x(), m_q_w_last.y(), m_q_w_last.z()),
                                        m_t_w_last);
                                    block_id = problem.AddResidualBlock(cost_function, loss_function, para_buff, para_buff + 3);
                                }
                                else
                                {
                                    cost_function = quaternion_point2plane<double>::Create(
                                        curr_point,
                                        pt_1,
                                        pt_2,
                                        pt_3,
                                        Eigen::Matrix<double, 4, 1>(m_q_w_last.w(), m_q_w_last.x(), m_q_w_last.y(), m_q_w_last.z()),
                                        m_t_w_last);
                                    block_id = problem.AddResidualBlock(cost_function, loss_function, buff_incre, buff_incre + 4);
                                }
                                plane.push_back((pt_3 - pt_1).cross(pt_2 - pt_1));
                                curPt.push_back(curr_point);
                                tarA.push_back(pt_1);
                                residual_block_ids.push_back(block_id);
                                surf_avail_num++;
                            }
                        }
                        else
                            surface_rejecetion_num++;
                    }
                }

                std::vector< ceres::ResidualBlockId > residual_block_ids_temp;
                residual_block_ids_temp.reserve(residual_block_ids.size());

                if (residual_block_ids.size() > (size_t) m_maximum_allow_residual_block)
                {
                    remove_ = true;
                    residual_block_ids_temp.clear();

                    float threshold_to_reserve = (float) m_maximum_allow_residual_block / (float) residual_block_ids.size();
                    float *probability_to_drop = m_rand_float.rand_array_uniform(0, 1.0, residual_block_ids.size());
                    screen_out << "Number of residual blocks too Large, drop them to " << m_maximum_allow_residual_block << endl;
                    for (size_t i = 0; i < residual_block_ids.size(); i++)
                    {
                        if (probability_to_drop[i] > threshold_to_reserve)
                            problem.RemoveResidualBlock(residual_block_ids[i]);
                        else
                        {
                            residual_block_ids_temp.push_back(residual_block_ids[i]);
                            curPt_.push_back(curPt[i]);
                            tarA_.push_back(tarA[i]);
                            if (i < line.size())
                                line_.push_back(line[i]);
                            else
                                plane_.push_back(plane[i - line.size()]);
                        }
                    }
                    residual_block_ids = residual_block_ids_temp;
                    delete probability_to_drop;
                }
                if (remove_)
                {
                    assert(residual_block_ids.size()==tarA_.size());
                    assert(curPt_.size()==line_.size()+plane_.size());
                    line.clear();
                    plane.clear();
                    tarA.clear();
                    curPt.clear();
                    for (size_t i = 0; i < tarA_.size(); i++)
                    {
                        tarA.push_back(tarA_[i]);
                        curPt.push_back(curPt_[i]);
                        if (i < line_.size())
                            line.push_back(line_[i]);
                        else
                            plane.push_back(plane_[i - line_.size()]);
                    }
                    line_.clear();
                    plane_.clear();
                    tarA_.clear();
                    curPt_.clear();
                    remove_ = false;
                }
                
                ceres::Solver::Options options;

                for (size_t ii = 0; ii < 1; ii++)
                {
                    options.linear_solver_type = slover_type;
                    options.max_num_iterations = m_para_cere_max_iterations;
                    options.max_num_iterations = m_para_cere_prerun_times;
                    options.minimizer_progress_to_stdout = false;
                    options.check_gradients = false;
                    options.gradient_check_relative_precision = 1e-10;
                    if (USE_SIZED_COST)
                        set_ceres_solver_bound(problem, para_buff);
                    else
                        set_ceres_solver_bound(problem, buff_incre);

                    ceres::Solve(options, &problem, &summary);
                    residual_block_ids_temp.clear();
                    ceres::Problem::EvaluateOptions eval_options;
                    eval_options.residual_blocks = residual_block_ids;
                    double total_cost = 0.0;
                    std::vector<double> residuals;
                    problem.Evaluate(eval_options, &total_cost, &residuals, nullptr, nullptr);

                    double m_inliner_ratio_threshold = compute_inlier_residual_threshold(residuals, m_inlier_ratio);
                    m_inlier_threshold = std::max(m_inliner_dis, m_inliner_ratio_threshold);
                    assert(residual_block_ids.size()==tarA.size());
                    for (unsigned int i = 0; i < residual_block_ids.size(); i++)
                    {
                        if ((fabs(residuals[3 * i + 0]) + fabs(residuals[3 * i + 1]) + fabs(residuals[3 * i + 2])) > m_inlier_threshold) // std::min(1.0, 10 * avr_cost)
                            problem.RemoveResidualBlock(residual_block_ids[i]);
                        else
                        {
                            residual_block_ids_temp.push_back(residual_block_ids[i]);
                            curPt_.push_back(curPt[i]);
                            tarA_.push_back(tarA[i]);
                            if (i < line.size())
                                line_.push_back(line[i]);
                            else
                                plane_.push_back(plane[i - line.size()]);
                        }
                    }
                    residual_block_ids = residual_block_ids_temp;
                }
                line.clear();
                plane.clear();
                tarA.clear();
                curPt.clear();
                for (size_t i = 0; i < tarA_.size(); i++)
                {
                    tarA.push_back(tarA_[i]);
                    curPt.push_back(curPt_[i]);
                    if (i < line_.size())
                        line.push_back(line_[i]);
                    else
                        plane.push_back(plane_[i - line_.size()]);
                }
                line_.clear();
                plane_.clear();
                tarA_.clear();
                curPt_.clear();
                
                options.linear_solver_type = slover_type;
                options.max_num_iterations = m_para_cere_max_iterations;
                options.minimizer_progress_to_stdout = false;
                options.check_gradients = false;
                options.gradient_check_relative_precision = 1e-10;

                if (USE_SIZED_COST)
                    set_ceres_solver_bound(problem, para_buff);
                else
                    set_ceres_solver_bound(problem, buff_incre);
                ceres::Solve(options, &problem, &summary);

                double para_incre[6] = {0, 0, 0, 0, 0, 0};
                Eigen::Map<Eigen::Vector3d> a_inc = Eigen::Map<Eigen::Vector3d>(para_incre);
                Eigen::Map<Eigen::Vector3d> t_inc = Eigen::Map<Eigen::Vector3d>(para_incre + 3);
                double cost_aft = 0.0;
                double cost_ori = 0.0;
                
                if (USE_SELF_LM)
                {
                    double lambda = 1;
                    double rho = 1;

                    for (int it = 0; it < 20; it++)
                    {
                        size_t num = tarA.size();
                        Eigen::MatrixXd J(num, 6);
                        Eigen::MatrixXd f0(num, 1);
                        J.setZero();
                        f0.setZero();
                        Eigen::Matrix3d L;

                        for (size_t i = 0; i < num; i++)
                        {
                            if (i < line.size())
                            {
                                Eigen::Vector3d l = line[i];
                                L = Eigen::MatrixXd::Identity(3, 3) - l * l.transpose() / pow(l.norm(), 2);
                            }
                            else
                            {
                                Eigen::Vector3d l = plane[i - line.size()];
                                L = l * l.transpose() / pow(l.norm(), 2);
                            }
                            Eigen::Vector3d cur_pt = curPt[i];
                            Eigen::Vector3d pa = tarA[i];
                            Eigen::Quaterniond q_inc = toQuaterniond(a_inc);
                            Eigen::Vector3d temp = L * (m_q_w_last * (q_inc * cur_pt + t_inc) + m_t_w_last - pa);
                            Eigen::Matrix<double, 1, 3> sig = sign(temp);
                            f0(i, 0) = sig * temp;
                            Eigen::Matrix3d R = -L * m_q_w_last * hat(q_inc * cur_pt) * A(a_inc);
                            for (int j = 0; j < 3; j++)
                                J(i, j) = sig * R.col(j);

                            Eigen::Vector3d p_x{1.0, 0.0, 0.0};
                            Eigen::Vector3d p_y{0.0, 1.0, 0.0};
                            Eigen::Vector3d p_z{0.0, 0.0, 1.0};
                            J(i, 3) = sig * (L * m_q_w_last * p_x);
                            J(i, 4) = sig * (L * m_q_w_last * p_y);
                            J(i, 5) = sig * (L * m_q_w_last * p_z);
                        }

                        Eigen::MatrixXd H(6, 6);
                        H = J.transpose() * J;
                        Eigen::MatrixXd JTf(6, 1);
                        JTf = -J.transpose() * f0;
                        Eigen::MatrixXd DD = Eigen::MatrixXd::Identity(6, 6);
                        for (int i = 0; i < 6; i++)
                            DD(i, i) = H(i, i);
                        H += lambda * DD;
                        cv::Mat matA0(6, 6, CV_64F, cv::Scalar::all(0));
                        cv::Mat matB0(6, 1, CV_64F, cv::Scalar::all(0));
                        cv::Mat matX0(6, 1, CV_64F, cv::Scalar::all(0));

                        for (int i = 0; i < 6; i++)
                        {
                            matB0.at<double>(i, 0) = JTf(i, 0);
                            for (int j = 0; j < 6; j++)
                                matA0.at<double>(i, j) = H(i, j);
                        }
                        cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);
                        
                        Eigen::Vector3d t_temp, a_temp;
                        Eigen::Matrix<double, 6, 1> delta_x;
                        for (int i = 0; i < 3; i++)
                        {
                            a_temp(i) = matX0.at<double>(i, 0);
                            t_temp(i) = matX0.at<double>(i + 3, 0);
                            delta_x(i) = matX0.at<double>(i, 0);
                            delta_x(i + 3) = matX0.at<double>(i + 3, 0);
                        }
                        
                        cost_ori = cost_func(line, plane, tarA, curPt, m_q_w_last * toQuaterniond(a_inc), m_q_w_last * t_inc + m_t_w_last);
                        cost_aft = cost_func(line, plane, tarA, curPt, m_q_w_last * toQuaterniond(a_inc + a_temp), m_q_w_last * (t_inc + t_temp) + m_t_w_last);
                        rho = (cost_ori - cost_aft) / pow((J * delta_x).norm(), 2);
                        //std::cout<<"rho: "<<rho<<", lambda: "<<lambda<<std::endl;
                        if (rho > 0.5)
                        {
                            //std::cout<<"cost before: "<<cost_ori<<", cost after: "<<cost_aft<<std::endl;
                            a_inc += a_temp;
                            t_inc += t_temp;
                        }
                        if (rho > 0.75)
                            lambda *= max(1.0 / 3, 1 - pow(2 * rho - 1 ,3));
                        if (rho < 0.25)
                            lambda *= 2;
                        if (lambda > 1e6 || lambda < 1e-6)
                            break;
                    }
                }

                if (USE_SELF_LM)
                {
                    Eigen::Quaterniond q_inc = toQuaterniond(a_inc);
                    m_t_w_curr = m_q_w_last * t_inc + m_t_w_last;
                    m_q_w_curr = m_q_w_last * q_inc;

                    m_angular_diff = (float) m_q_w_curr.angularDistance(m_q_w_last) * 57.3;
                    m_t_diff = (m_t_w_curr - m_t_w_last).norm();
                    minimize_cost = cost_ori / tarA.size();
                    std::cout<<"cost: "<<minimize_cost<<std::endl;

                    if (q_last_optimize.angularDistance(q_inc) < 57.3 * m_minimum_icp_R_diff &&
                        (t_last_optimize - t_inc).norm() < m_minimum_icp_T_diff)
                    {
                        ROS_INFO_ONCE("USE_SELF_LM");
                        screen_out << "----- Terminate, iteration times  = " << iterCount << "-----" << endl;
                        break;
                    }
                    else
                    {
                        q_last_optimize = q_inc;
                        t_last_optimize = t_inc;
                    }
                }
                else if (USE_SIZED_COST)
                {
                    Eigen::Quaterniond q_incre = toQuaterniond(a_incre);
                    m_t_w_curr = m_q_w_last * t_incre + m_t_w_last;
                    m_q_w_curr = m_q_w_last * q_incre;

                    m_angular_diff = (float) m_q_w_curr.angularDistance(m_q_w_last) * 57.3;
                    m_t_diff = (m_t_w_curr - m_t_w_last).norm();
                    minimize_cost = summary.final_cost;

                    if (q_last_optimize.angularDistance(q_incre) < 57.3 * m_minimum_icp_R_diff &&
                        (t_last_optimize - t_incre).norm() < m_minimum_icp_T_diff)
                    {
                        screen_out << "----- Terminate, iteration times  = " << iterCount << "-----" << endl;
                        break;
                    }
                    else
                    {
                        q_last_optimize = q_incre;
                        t_last_optimize = t_incre;
                    }
                }
                else
                {
                    m_t_w_curr = m_q_w_last * t_w_incre + m_t_w_last;
                    m_q_w_curr = m_q_w_last * q_w_incre;

                    m_angular_diff = (float) m_q_w_curr.angularDistance(m_q_w_last) * 57.3;
                    m_t_diff = (m_t_w_curr - m_t_w_last).norm();
                    minimize_cost = summary.final_cost;

                    if (q_last_optimize.angularDistance(q_w_incre) < 57.3 * m_minimum_icp_R_diff &&
                        (t_last_optimize - t_w_incre).norm() < m_minimum_icp_T_diff)
                    {
                        screen_out << "----- Terminate, iteration times  = " << iterCount << "-----" << endl;
                        break;
                    }
                    else
                    {
                        q_last_optimize = q_w_incre;
                        t_last_optimize = t_w_incre;
                    }
                }
            }
            fclose(fp);
            screen_printf("===== corner factor num %d , surf factor num %d=====\n", corner_avail_num, surf_avail_num);
            if (laser_corner_pt_num != 0 && laser_surface_pt_num != 0)
            {
                m_logger_common->printf("Corner  total num %d |  use %d | rate = %d %% \r\n", laser_corner_pt_num, corner_avail_num, (corner_avail_num) *100 / laser_corner_pt_num);
                m_logger_common->printf("Surface total num %d |  use %d | rate = %d %% \r\n", laser_surface_pt_num, surf_avail_num, (surf_avail_num) *100 / laser_surface_pt_num);
            }
            *(m_logger_timer->get_ostream()) << m_timer->toc_string("Pose optimization") << std::endl;
            if (g_export_full_count < 5)
            {
                *(m_logger_common->get_ostream()) << summary.FullReport() << endl;
                g_export_full_count++;
            }
            else
                *(m_logger_common->get_ostream()) << summary.BriefReport() << endl;

            *(m_logger_common->get_ostream()) << "Last R:" << m_q_w_last.toRotationMatrix().eulerAngles(0, 1, 2).transpose() * 57.3 << " ,T = " << m_t_w_last.transpose() << endl;
            *(m_logger_common->get_ostream()) << "Curr R:" << m_q_w_curr.toRotationMatrix().eulerAngles(0, 1, 2).transpose() * 57.3 << " ,T = " << m_t_w_curr.transpose() << endl;
            *(m_logger_common->get_ostream()) << "Iteration time: " << iterCount << endl;

            m_logger_common->printf("Motion blur = %d | ", m_if_motion_deblur);
            m_logger_common->printf("Cost = %.5f| inlier_thr = %.2f |blk_size = %d | corner_num = %d | surf_num = %d | angle dis = %.2f | T dis = %.2f \r\n",
                                    minimize_cost, m_inlier_threshold, summary.num_residual_blocks, corner_avail_num, surf_avail_num, m_angular_diff, m_t_diff);

            m_inlier_threshold = m_inlier_threshold* summary.final_cost/ summary.initial_cost; //

            if (m_angular_diff > m_para_max_angular_rate || minimize_cost > m_max_final_cost)
            {
                *(m_logger_common->get_ostream()) << "**** Reject update **** " << endl;
                *(m_logger_common->get_ostream()) << summary.FullReport() << endl;
                for (int i = 0; i < 7; i++)
                    m_para_buffer_RT[i] = m_para_buffer_RT_last[i];
                m_last_time_stamp = m_minimum_pt_time_stamp;
                m_q_w_curr = m_q_w_last;
                m_t_w_curr = m_t_w_last;
                return 0;
            }
            m_final_opt_summary = summary;
        }
        else
            screen_out << "time Map corner and surf num are not enough" << std::endl;

        return 1;
    }

    int find_out_incremental_transfrom(pcl::PointCloud<PointType>::Ptr in_laser_cloud_corner_from_map,
                                       pcl::PointCloud<PointType>::Ptr in_laser_cloud_surf_from_map,
                                       pcl::PointCloud<PointType>::Ptr laserCloudCornerStack,
                                       pcl::PointCloud<PointType>::Ptr laserCloudSurfStack)
    {
        pcl::PointCloud<PointType> laser_cloud_corner_from_map = *in_laser_cloud_corner_from_map;
        pcl::PointCloud<PointType> laser_cloud_surf_from_map = *in_laser_cloud_surf_from_map;
        if (laser_cloud_corner_from_map.points.size() && laser_cloud_surf_from_map.points.size())
        {
            m_kdtree_corner_from_map.setInputCloud(laser_cloud_corner_from_map.makeShared());
            m_kdtree_surf_from_map.setInputCloud(laser_cloud_surf_from_map.makeShared());
        }
        else
            return 1;

        return find_out_incremental_transfrom(in_laser_cloud_corner_from_map, in_laser_cloud_surf_from_map, m_kdtree_corner_from_map, m_kdtree_surf_from_map, laserCloudCornerStack, laserCloudSurfStack);
    }

    void compute_interpolatation_rodrigue(const Eigen::Quaterniond &q_in, Eigen::Matrix<double, 3, 1> &angle_axis, double &angle_theta, Eigen::Matrix<double, 3, 3> &hat)
    {
        Eigen::AngleAxisd newAngleAxis(q_in);
        angle_axis = newAngleAxis.axis();
        angle_axis = angle_axis / angle_axis.norm();
        angle_theta = newAngleAxis.angle();
        hat.setZero();
        hat(0, 1) = -angle_axis(2);
        hat(1, 0) = angle_axis(2);
        hat(0, 2) = angle_axis(1);
        hat(2, 0) = -angle_axis(1);
        hat(1, 2) = -angle_axis(0);
        hat(2, 1) = angle_axis(0);
    };

    void pointAssociateToMap(PointType const *const pi,
                             PointType *const po,
                             double interpolate_s = 1.0,
                             int if_undistore = 0)
    {
        Eigen::Vector3d point_curr(pi->x, pi->y, pi->z);
        Eigen::Vector3d point_w;
        if (m_if_motion_deblur == 0 || if_undistore == 0 || interpolate_s == 1.0)
            point_w = m_q_w_curr * point_curr + m_t_w_curr;

        po->x = point_w.x();
        po->y = point_w.y();
        po->z = point_w.z();
        po->intensity = pi->intensity;
    }

    void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
    {
        Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
        Eigen::Vector3d point_curr = m_q_w_curr.inverse() * (point_w - m_t_w_curr);
        po->x = point_curr.x();
        po->y = point_curr.y();
        po->z = point_curr.z();
        po->intensity = pi->intensity;
    }

    unsigned int pointcloudAssociateToMap(pcl::PointCloud<PointType> const &pc_in, pcl::PointCloud<PointType> &pt_out,
                                            int if_undistore = 0)
    {
        unsigned int points_size = pc_in.points.size();
        pt_out.points.resize(points_size);

        for (unsigned int i = 0; i < points_size; i++)
            pointAssociateToMap(&pc_in.points[i], &pt_out.points[i], refine_blur(pc_in.points[i].intensity, m_minimum_pt_time_stamp, m_maximum_pt_time_stamp), if_undistore);

        return points_size;
    }

    unsigned int pointcloudAssociateTbeoMapped(pcl::PointCloud<PointType> const &pc_in, pcl::PointCloud<PointType> &pt_out)
    {
        unsigned int points_size = pc_in.points.size();
        pt_out.points.resize(points_size);

        for (unsigned int i = 0; i < points_size; i++)
            pointAssociateTobeMapped(&pc_in.points[i], &pt_out.points[i]);

        return points_size;
    }

};


#endif // POINT_CLOUD_REGISTRATION_HPP
