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
#include "logger.hpp"
#include "pcl_tools.hpp"
#include "timer.hpp"
#include "cell_map_hash.hpp"

#define CORNER_MIN_MAP_NUM 0
#define SURFACE_MIN_MAP_NUM 50

#define BLUR_SCALE 1.0

using namespace PCL_TOOLS;
using namespace COMMON_TOOLS;

int g_export_full_count = 0;

class PointCloudRegistration
{
public:
    ceres::LinearSolverType slover_type = ceres::SPARSE_NORMAL_CHOLESKY;
    int line_search_num = 5;
    int IF_LINE_FEATURE_CHECK = 0;
    int plane_search_num = 5;
    int IF_PLANE_FEATURE_CHECK = 0;
    int ICP_PLANE = 1;
    int ICP_LINE= 1;
    double para_buffer_RT[7] = {0, 0, 0, 1, 0, 0, 0};
    double para_buffer_RT_last[7] = {0, 0, 0, 1, 0, 0, 0};
    double para_buffer_incremental[7] = {0, 0, 0, 1, 0, 0, 0};

    double m_interpolatation_theta;

    int if_motion_deblur = 0;

    double angular_diff = 0;
    double t_diff = 0;
    double maximum_dis_plane_for_match = 50.0;
    double maximum_dis_line_for_match = 2.0;
    Eigen::Matrix<double, 3, 1> m_interpolatation_omega;
    Eigen::Matrix<double, 3, 3> m_interpolatation_omega_hat;
    Eigen::Matrix<double, 3, 3> m_interpolatation_omega_hat_sq2;

    const Eigen::Quaterniond m_q_I = Eigen::Quaterniond(1, 0, 0, 0);

    pcl::KdTreeFLANN<PointType> kdtree_corner_from_map_;
    pcl::KdTreeFLANN<PointType> kdtree_surf_from_map_;

    Eigen::Quaterniond q_w_curr, q_w_last;
    Eigen::Vector3d t_w_curr, t_w_last;

    Eigen::Map<Eigen::Quaterniond> m_q_w_incre = Eigen::Map<Eigen::Quaterniond>(para_buffer_incremental);
    Eigen::Map<Eigen::Vector3d> m_t_w_incre = Eigen::Map<Eigen::Vector3d>(para_buffer_incremental + 4);

    COMMON_TOOLS::File_logger *logger_common;
    COMMON_TOOLS::File_logger *logger_pcd;
    COMMON_TOOLS::File_logger *m_logger_timer;

    COMMON_TOOLS::Timer *m_timer;
    int current_frame_index;
    int mapping_init_accumulate_frames = 100;
    float  last_time_stamp = 0;
    float  para_max_angular_rate = 200.0 / 50.0; // max angular rate = 90.0 /50.0 deg/s
    float  para_max_speed = 100.0 / 50.0;        // max speed = 10 m/s
    float  m_max_final_cost = 100.0;
    int para_icp_max_iterations = 20;
    int    para_cere_max_iterations = 100;
    float minimum_pt_time_stamp = 0;
    float maximum_pt_time_stamp = 1.0;
    double m_minimum_icp_R_diff = 0.01;
    double m_minimum_icp_T_diff = 0.01;

    double m_inliner_dis = 0.02;
    double m_inlier_ratio = 0.80;

    double m_inlier_final_threshold;
    ceres::Solver::Summary summary;
    ceres::Solver::Summary m_final_opt_summary;

    ~PointCloudRegistration() {};

    PointCloudRegistration()
    {
        q_w_last.setIdentity();
        t_w_last.setZero();
        q_w_curr.setIdentity();
        t_w_curr.setZero();
    };

    void reset_incremtal_parameter()
    {
        m_interpolatation_theta = 0;
        m_interpolatation_omega_hat.setZero();
        m_interpolatation_omega_hat_sq2.setZero();
    };

    float refine_blur(float in_blur, const float &min_blur, const float &max_blur)
    {
        float res = 1.0;
        if (if_motion_deblur)
        {
            res = (in_blur - min_blur) / (max_blur - min_blur);
            if (!std::isfinite(res) || res > 1.0)
                return 1.0;
            else
                return res;
        }
        return res;
    };

    void set_ceres_solver_bound(ceres::Problem &problem, double *para_buffer_RT)
    {
        for (unsigned int i = 0; i < 3; i++)
        {
            problem.SetParameterLowerBound(para_buffer_RT + 4, i, - para_max_speed);
            problem.SetParameterUpperBound(para_buffer_RT + 4, i, + para_max_speed);
        }
    };

    double compute_inlier_residual_threshold(std::vector<double> residuals, double ratio)
    {
        std::set<double> dis_vec;
        for (size_t i = 0; i < (size_t)(residuals.size() / 3); i++)
            dis_vec.insert(fabs(residuals[3 * i + 0]) + fabs(residuals[3 * i + 1]) + fabs(residuals[3 * i + 2]));
        return *std::next(dis_vec.begin(), (int)((ratio) * dis_vec.size()));
    }

    int find_out_incremental_transfrom(pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map,
                                       pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map,
                                       pcl::KdTreeFLANN<PointType> &kdtree_corner_from_map,
                                       pcl::KdTreeFLANN<PointType> &kdtree_surf_from_map,
                                       pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map_filtered,
                                       pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map_filtered)
    {
        Eigen::Map<Eigen::Quaterniond> q_w_incre = Eigen::Map<Eigen::Quaterniond>(para_buffer_incremental);
        Eigen::Map<Eigen::Vector3d> t_w_incre = Eigen::Map<Eigen::Vector3d>(para_buffer_incremental + 4);

        kdtree_corner_from_map_ = kdtree_corner_from_map;
        kdtree_surf_from_map_ = kdtree_surf_from_map;

        pcl::PointCloud<PointType> _laser_cloud_corner_from_map = *laser_cloud_corner_from_map;
        pcl::PointCloud<PointType> _laser_cloud_surf_from_map = *laser_cloud_surf_from_map;

        size_t laser_corner_pt_num = _laser_cloud_corner_from_map.points.size();
        size_t laser_surface_pt_num = _laser_cloud_surf_from_map.points.size();
        size_t laser_corner_pt_num_filtered = laser_cloud_corner_from_map_filtered->points.size();
        size_t laser_surface_pt_num_filtered = laser_cloud_surf_from_map_filtered->points.size();

        *(m_logger_timer->get_ostream())<< m_timer->toc_string("Query points for match") << std::endl;
        m_timer->tic("Pose optimization");

        int surf_avail_num = 0;
        int corner_avail_num = 0;
        float minimize_cost = summary.final_cost;
        PointType pointOri, pointSel;
        int corner_rejection_num = 0;
        int surface_rejecetion_num = 0;
        int if_undistore_in_matching = 1;

        if (laser_corner_pt_num > CORNER_MIN_MAP_NUM &&
            laser_surface_pt_num > SURFACE_MIN_MAP_NUM &&
            current_frame_index > mapping_init_accumulate_frames)
        {
            m_timer->tic("Build kdtree");
            *(m_logger_timer->get_ostream()) << m_timer->toc_string("Build kdtree") << std::endl;
            Eigen::Quaterniond q_last_optimize(1.f, 0.f, 0.f, 0.f);
            Eigen::Vector3d t_last_optimize(0.f, 0.f, 0.f);
            int iterCount = 0;

            std::vector<int> point_search_Idx;
            std::vector<float> point_search_sq_dis;
            
            for (iterCount = 0; iterCount < para_icp_max_iterations; iterCount++)
            {
                point_search_Idx.clear();
                point_search_sq_dis.clear();
                corner_avail_num = 0;
                surf_avail_num = 0;
                corner_rejection_num = 0;
                surface_rejecetion_num = 0;

                ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
                ceres::LocalParameterization *q_parameterization = new ceres::EigenQuaternionParameterization();
                ceres::Problem::Options problem_options;
                ceres::ResidualBlockId block_id;
                ceres::Problem problem(problem_options);
                std::vector<ceres::ResidualBlockId> residual_block_ids;
                
                problem.AddParameterBlock(para_buffer_incremental, 4, q_parameterization);
                problem.AddParameterBlock(para_buffer_incremental + 4, 3);

                for (size_t i = 0; i < laser_corner_pt_num_filtered; i++)
                {
                    pointOri = laser_cloud_corner_from_map_filtered->points[i];
                    
                    if ((!std::isfinite(pointOri.x)) || (!std::isfinite(pointOri.y)) || (!std::isfinite(pointOri.z)))
                        continue;

                    pointAssociateToMap(&pointOri,
                                        &pointSel,
                                        refine_blur(pointOri.intensity, minimum_pt_time_stamp, maximum_pt_time_stamp),
                                        if_undistore_in_matching);
                    
                    if (kdtree_corner_from_map_.nearestKSearch(pointSel, line_search_num, point_search_Idx, point_search_sq_dis) != line_search_num)
                        continue;
                                        
                    if (point_search_sq_dis[line_search_num - 1] < maximum_dis_line_for_match)
                    {
                        bool line_is_avail = true;
                        std::vector<Eigen::Vector3d> nearCorners;
                        Eigen::Vector3d center(0, 0, 0);
                        if (IF_LINE_FEATURE_CHECK)
                        {
                            for (int j = 0; j < line_search_num; j++)
                            {
                                Eigen::Vector3d tmp(_laser_cloud_corner_from_map.points[point_search_Idx[j]].x,
                                                    _laser_cloud_corner_from_map.points[point_search_Idx[j]].y,
                                                    _laser_cloud_corner_from_map.points[point_search_Idx[j]].z);
                                center = center + tmp;
                                nearCorners.push_back(tmp);
                            }

                            center = center / ((float)line_search_num);

                            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();

                            for (int j = 0; j < line_search_num; j++)
                            {
                                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                            }

                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

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
                                if (point_search_Idx[0] < 0 || point_search_Idx[0] >= _laser_cloud_corner_from_map.size() ||
                                    point_search_Idx[1] < 0 || point_search_Idx[1] >= _laser_cloud_corner_from_map.size())
                                    continue;
                                
                                auto pt_1 = pcl_pt_to_eigend(_laser_cloud_corner_from_map.points[point_search_Idx[0]]);
                                auto pt_2 = pcl_pt_to_eigend(_laser_cloud_corner_from_map.points[point_search_Idx[1]]);
                                
                                if ((pt_1 - pt_2).norm() < 0.0001)
                                    continue;

                                if (if_motion_deblur)
                                    cost_function =
                                        ceres_icp_point2line_mb<double>::Create(
                                            curr_point,
                                            pt_1,
                                            pt_2,
                                            refine_blur(pointOri.intensity, minimum_pt_time_stamp, maximum_pt_time_stamp) * 1.0,
                                            Eigen::Matrix<double, 4, 1>(q_w_last.w(), q_w_last.x(), q_w_last.y(), q_w_last.z()),
                                            t_w_last); //pointOri.intensity);
                                else
                                    cost_function =
                                        ceres_icp_point2line<double>::Create(
                                            curr_point,
                                            pt_1,
                                            pt_2,
                                            Eigen::Matrix<double, 4, 1>(q_w_last.w(), q_w_last.x(), q_w_last.y(), q_w_last.z()),
                                            t_w_last);
                                
                                block_id = problem.AddResidualBlock(cost_function, loss_function, para_buffer_incremental, para_buffer_incremental + 4);
                                residual_block_ids.push_back(block_id);
                                corner_avail_num++;
                            }
                        }
                        else
                            corner_rejection_num++;
                    }
                }

                for (size_t i = 0; i < laser_surface_pt_num_filtered; i++)
                {
                    pointOri = laser_cloud_surf_from_map_filtered->points[i];
                    int planeValid = true;
                    pointAssociateToMap(&pointOri,
                                        &pointSel,
                                        refine_blur(pointOri.intensity, minimum_pt_time_stamp, maximum_pt_time_stamp),
                                        if_undistore_in_matching);
                    
                    kdtree_surf_from_map_.nearestKSearch(pointSel, plane_search_num, point_search_Idx, point_search_sq_dis);

                    if (point_search_Idx[0] < 0 || point_search_Idx[0] >= _laser_cloud_corner_from_map.size() ||
                        point_search_Idx[plane_search_num / 2] < 0 || point_search_Idx[plane_search_num / 2] >= _laser_cloud_corner_from_map.size() ||
                        point_search_Idx[plane_search_num - 1] < 0 || point_search_Idx[plane_search_num - 1] >= _laser_cloud_corner_from_map.size())
                        continue;

                    if (point_search_sq_dis[plane_search_num - 1] < maximum_dis_plane_for_match)
                    {
                        std::vector<Eigen::Vector3d> nearCorners;
                        Eigen::Vector3d center(0, 0, 0);

                        if (IF_PLANE_FEATURE_CHECK)
                        {
                            for (int j = 0; j < plane_search_num; j++)
                            {
                                Eigen::Vector3d tmp(_laser_cloud_corner_from_map.points[point_search_Idx[j]].x,
                                                    _laser_cloud_corner_from_map.points[point_search_Idx[j]].y,
                                                    _laser_cloud_corner_from_map.points[point_search_Idx[j]].z);
                                center = center + tmp;
                                nearCorners.push_back(tmp);
                            }

                            center = center / (float)(plane_search_num);

                            Eigen::Matrix3d covMat = Eigen::Matrix3d::Zero();

                            for (int j = 0; j < plane_search_num; j++)
                            {
                                Eigen::Matrix<double, 3, 1> tmpZeroMean = nearCorners[j] - center;
                                covMat = covMat + tmpZeroMean * tmpZeroMean.transpose();
                            }

                            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(covMat);

                            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[0] &&
                                saes.eigenvalues()[2] < 10 * saes.eigenvalues()[1])
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

                                if (if_motion_deblur)
                                    cost_function = ceres_icp_point2plane_mb<double>::Create(
                                        curr_point,
                                        pcl_pt_to_eigend(_laser_cloud_surf_from_map.points[point_search_Idx[0]]),
                                        pcl_pt_to_eigend(_laser_cloud_surf_from_map.points[point_search_Idx[plane_search_num / 2]]),
                                        pcl_pt_to_eigend(_laser_cloud_surf_from_map.points[point_search_Idx[plane_search_num - 1]]),
                                        refine_blur(pointOri.intensity, minimum_pt_time_stamp, maximum_pt_time_stamp) * 1.0,
                                        Eigen::Matrix<double, 4, 1>(q_w_last.w(), q_w_last.x(), q_w_last.y(), q_w_last.z()),
                                        t_w_last);
                                else
                                    cost_function = ceres_icp_point2plane<double>::Create(
                                        curr_point,
                                        pcl_pt_to_eigend(_laser_cloud_surf_from_map.points[point_search_Idx[0]]),
                                        pcl_pt_to_eigend(_laser_cloud_surf_from_map.points[point_search_Idx[plane_search_num / 2]]),
                                        pcl_pt_to_eigend(_laser_cloud_surf_from_map.points[point_search_Idx[plane_search_num - 1]]),
                                        Eigen::Matrix<double, 4, 1>(q_w_last.w(), q_w_last.x(), q_w_last.y(), q_w_last.z()),
                                        t_w_last); //pointOri.intensity);
                                
                                block_id = problem.AddResidualBlock(cost_function, loss_function, para_buffer_incremental, para_buffer_incremental + 4);
                                residual_block_ids.push_back(block_id);
                            }
                            surf_avail_num++;
                        }
                        else
                            surface_rejecetion_num++;
                    }
                }

                ceres::Solver::Options options;
                std::vector<ceres::ResidualBlockId> residual_block_ids_bak;
                residual_block_ids_bak = residual_block_ids;

                for (size_t ii = 0; ii < 1; ii++)
                {
                    options.linear_solver_type = slover_type;
                    options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
                    options.num_threads = 8;
                    options.max_num_iterations = para_cere_max_iterations;
                    options.max_num_iterations = 2;
                    options.minimizer_progress_to_stdout = false;
                    options.check_gradients = false;

                    set_ceres_solver_bound(problem, para_buffer_incremental);
                    ceres::Solve(options, &problem, &summary);

                    // Remove outliers
                    residual_block_ids_bak.clear();

                    ceres::Problem::EvaluateOptions eval_options;
                    eval_options.residual_blocks = residual_block_ids;
                    double total_cost = 0.0;
                    std::vector<double> residuals;
                    problem.Evaluate(eval_options, &total_cost, &residuals, nullptr, nullptr);

                    double inliner_ratio_threshold = compute_inlier_residual_threshold(residuals, m_inlier_ratio);
                    m_inlier_final_threshold = std::max(m_inliner_dis, inliner_ratio_threshold);

                    for (unsigned int i = 0; i < residual_block_ids.size(); i++)
                    {   
                        if ((fabs(residuals[3 * i + 0]) + fabs(residuals[3 * i + 1]) + fabs(residuals[3 * i + 2])) >  m_inlier_final_threshold) // std::min(1.0, 10 * avr_cost)
                            problem.RemoveResidualBlock(residual_block_ids[i]);
                        else
                            residual_block_ids_bak.push_back(residual_block_ids[i]);
                    }

                    residual_block_ids = residual_block_ids_bak;
                }
                options.linear_solver_type = slover_type;
                options.max_num_iterations = para_cere_max_iterations;
                set_ceres_solver_bound(problem, para_buffer_incremental);
                ceres::Solve(options, &problem, &summary);
                if (if_motion_deblur)
                {
                    compute_interpolatation_rodrigue(q_w_incre, m_interpolatation_omega, m_interpolatation_theta, m_interpolatation_omega_hat);
                    m_interpolatation_omega_hat_sq2 = m_interpolatation_omega_hat * m_interpolatation_omega_hat;
                }

                t_w_curr = q_w_last * t_w_incre + t_w_last;
                q_w_curr = q_w_last * q_w_incre;

                angular_diff = (float)q_w_curr.angularDistance(q_w_last) * 57.3;
                t_diff = (t_w_curr - t_w_last).norm();
                minimize_cost = summary.final_cost;
            
                if (q_last_optimize.angularDistance(q_w_incre) < 57.3 * m_minimum_icp_R_diff &&
                    (t_last_optimize - t_w_incre).norm() < m_minimum_icp_T_diff)
                    break;
                else
                {
                    q_last_optimize = q_w_incre;
                    t_last_optimize = t_w_incre;
                }
            }
            
            if (laser_corner_pt_num_filtered != 0 && laser_surface_pt_num_filtered != 0)
            {
                logger_common->printf("Corner  total num %d |  use %d | rate = %d %% \r\n", laser_corner_pt_num_filtered, corner_avail_num, (corner_avail_num) *100 / laser_corner_pt_num_filtered);
                logger_common->printf("Surface total num %d |  use %d | rate = %d %% \r\n", laser_surface_pt_num_filtered, surf_avail_num, (surf_avail_num) *100 / laser_surface_pt_num_filtered);
            }
            *(m_logger_timer->get_ostream()) << m_timer->toc_string("Pose optimization") << std::endl;
            if (g_export_full_count < 5)
            {
                *(logger_common->get_ostream()) << summary.FullReport() << endl;
                g_export_full_count++;
            }
            else
                *(logger_common->get_ostream()) << summary.BriefReport() << endl;

            *(logger_common->get_ostream()) << "Last R:" << q_w_last.toRotationMatrix().eulerAngles(0, 1, 2).transpose() * 57.3 << " ,T = " << t_w_last.transpose() << endl;
            *(logger_common->get_ostream()) << "Curr R:" << q_w_curr.toRotationMatrix().eulerAngles(0, 1, 2).transpose() * 57.3 << " ,T = " << t_w_curr.transpose() << endl;
            *(logger_common->get_ostream()) << "Iteration time: " << iterCount << endl;

            logger_common->printf("Motion blur = %d | ", if_motion_deblur);
            logger_common->printf("Cost = %.5f| inlier_thr = %.2f |blk_size = %d | corner_num = %d | surf_num = %d | angle dis = %.2f | T dis = %.2f \r\n",
                                     minimize_cost, m_inlier_final_threshold, summary.num_residual_blocks, corner_avail_num, surf_avail_num, angular_diff, t_diff);
            if (angular_diff > para_max_angular_rate || minimize_cost > m_max_final_cost)
            {
                ROS_WARN("angular_diff > para_max_angular_rate OR minimize_cost > m_max_final_cost");
                *(logger_common->get_ostream()) << "**** Reject update **** " << endl;
                *(logger_common->get_ostream()) << summary.FullReport() << endl;
                for (int i = 0; i < 7; i++)
                    para_buffer_RT[i] = para_buffer_RT_last[i];
                last_time_stamp = minimum_pt_time_stamp;
                q_w_curr = q_w_last;
                t_w_curr = t_w_last;
                return 0;
            }
            m_final_opt_summary = summary;
        }
        else
            std::cout<<"need more surf, corner num; current_frame_index: "<<current_frame_index<<std::endl;

        return 1;
    }

    int find_out_incremental_transfrom(pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map,
                                       pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map,
                                       pcl::PointCloud<PointType>::Ptr laser_cloud_corner_from_map_filtered,
                                       pcl::PointCloud<PointType>::Ptr laser_cloud_surf_from_map_filtered)
    {
        pcl::PointCloud<PointType> _laser_cloud_corner_from_map = *laser_cloud_corner_from_map;
        pcl::PointCloud<PointType> _laser_cloud_surf_from_map = *laser_cloud_surf_from_map;
        if (_laser_cloud_corner_from_map.points.size() && _laser_cloud_surf_from_map.points.size())
        {
            kdtree_corner_from_map_.setInputCloud(_laser_cloud_corner_from_map.makeShared());
            kdtree_surf_from_map_.setInputCloud(_laser_cloud_surf_from_map.makeShared());
        }
        else
            return 1;

        return find_out_incremental_transfrom(laser_cloud_corner_from_map,
                                              laser_cloud_surf_from_map,
                                              kdtree_corner_from_map_,
                                              kdtree_surf_from_map_,
                                              laser_cloud_corner_from_map_filtered,
                                              laser_cloud_surf_from_map_filtered);
    }

    void compute_interpolatation_rodrigue(const Eigen::Quaterniond &q_in,
                                          Eigen::Matrix<double, 3, 1> &angle_axis,
                                          double &angle_theta,
                                          Eigen::Matrix<double, 3, 3> &hat)
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
        if (if_motion_deblur == 0 || if_undistore == 0 || interpolate_s == 1.0)
            point_w = q_w_curr * point_curr + t_w_curr;
        else
        {
            if (interpolate_s > 1.0 || interpolate_s < 0.0)
                printf("Input interpolate_s = %.5f\r\n", interpolate_s);
            Eigen::Quaterniond interpolate_q = m_q_I.slerp(interpolate_s * BLUR_SCALE, m_q_w_incre);
            Eigen::Vector3d interpolate_T = m_t_w_incre * (interpolate_s * BLUR_SCALE);
            point_w = q_w_last * (interpolate_q * point_curr + interpolate_T) + t_w_last;
        }
        po->x = point_w.x();
        po->y = point_w.y();
        po->z = point_w.z();
        po->intensity = pi->intensity;
    }

    void pointAssociateTobeMapped(PointType const *const pi, PointType *const po)
    {
        Eigen::Vector3d point_w(pi->x, pi->y, pi->z);
        Eigen::Vector3d point_curr = q_w_curr.inverse() * (point_w - t_w_curr);
        po->x = point_curr.x();
        po->y = point_curr.y();
        po->z = point_curr.z();
        po->intensity = pi->intensity;
    }

    unsigned int pointcloudAssociateToMap(pcl::PointCloud<PointType> const &pc_in,
                                          pcl::PointCloud<PointType> &pt_out,
                                          int if_undistore = 0)
    {
        unsigned int points_size = pc_in.points.size();
        pt_out.points.resize(points_size);
        for (unsigned int i = 0; i < points_size; i++)
            pointAssociateToMap(&pc_in.points[i],
                                &pt_out.points[i],
                                refine_blur(pc_in.points[i].intensity, minimum_pt_time_stamp, maximum_pt_time_stamp),
                                if_undistore);
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
