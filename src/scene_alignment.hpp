#ifndef __SCENCE_ALIGNMENT_HPP__
#define __SCENCE_ALIGNMENT_HPP__
#include "cell_map_hash.hpp"
#include "common_tools.h"
#include <iostream>
#include <pcl/registration/ndt.h>
#include <stdio.h>
#include <vector>

#include "point_cloud_registration.hpp"
#include <pcl/registration/icp.h>

#include "json_tools.hpp"
#include "pcl_tools.hpp"
#include "ceres/ceres.h"
#include "ceres_pose_graph_3d.hpp"

template <typename PT_DATA_TYPE>
class SceneAlignment
{
public:
    COMMON_TOOLS::File_logger file_logger_commond, file_logger_timer;
    COMMON_TOOLS::Timer timer;

    float line_res_ = 0.4;
    float plane_res_ = 0.4;
    pcl::VoxelGrid<PointType> down_sample_filter_corner;
    pcl::VoxelGrid<PointType> down_sample_filter_surface;
    int pair_idx = 0;
    PointCloudRegistration pc_reg;
    std::string save_path;

    SceneAlignment()
    {
        pair_idx = 0;
        set_downsample_resolution(line_res_, plane_res_);
    }

    SceneAlignment(std::string path)
    {
        init(path);
    };

    ~SceneAlignment(){};

    static int load_pose_and_regerror(std::string file_name,
                                      Eigen::Quaterniond &q_curr,
                                      Eigen::Vector3d &t_curr,
                                      Eigen::Matrix<double, Eigen::Dynamic, 1> &mat_reg_err)
    {
        FILE *fp = fopen(file_name.c_str(), "r");
        if (fp == nullptr)
        {
            std::cout << "load_mapping_from_file: " << file_name << " fail!" << std::endl;
            return 0;
        }
        else
        {
            char readBuffer[1 << 16];
            rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
            rapidjson::Document       doc;

            doc.ParseStream(is);
            if (doc.HasParseError())
            {
                printf("GetParseError, err_code =  %d\n", doc.GetParseError());
                return 0;
            }
            auto json_arrray = COMMON_TOOLS::get_json_array<double>(doc["Q"].GetArray());
            q_curr.w() = json_arrray[0];
            q_curr.x() = json_arrray[1];
            q_curr.y() = json_arrray[2];
            q_curr.z() = json_arrray[3];

            t_curr = Eigen::Vector3d(get_json_array<double>(doc["T"].GetArray()));

            rapidjson::Document::Array json_array = doc["Reg_err"].GetArray();
            size_t reg_err_size = json_array.Size();
            mat_reg_err.resize(reg_err_size, 1);
            for (size_t i = 0; i < reg_err_size; i++)
                mat_reg_err(i) = json_array[i].GetDouble();
            return 1;
        }
    }

    static Ceres_pose_graph_3d::Constraint3D add_constrain_of_loop(int s_idx, int t_idx,
                                                                   Eigen::Quaterniond q_a, Eigen::Vector3d t_a,
                                                                   Eigen::Quaterniond q_b, Eigen::Vector3d t_b,
                                                                   Eigen::Quaterniond icp_q, Eigen::Vector3d icp_t,
                                                                   int if_verbose = 1)
    {
        Ceres_pose_graph_3d::Constraint3D pose_constrain;
        auto q_res = q_b.inverse() * icp_q.inverse() * q_a;
        auto t_res = q_b.inverse() * (icp_q.inverse() * (t_a - icp_t) - t_b);
        if (if_verbose == 0)
        {
            cout << "=== Add_constrain_of_loop ====" << endl;
            cout << q_a.coeffs().transpose() << endl;
            cout << q_b.coeffs().transpose() << endl;
            cout << icp_q.coeffs().transpose() << endl;
            cout << t_a.transpose() << endl;
            cout << t_b.transpose() << endl;
            cout << icp_t.transpose() << endl;
            cout << "Result: " << endl;
            cout << q_res.coeffs().transpose() << endl;
            cout << t_res.transpose() << endl;
        }
        pose_constrain.id_begin = s_idx;
        pose_constrain.id_end = t_idx;
        pose_constrain.t_be.p = t_res;
        pose_constrain.t_be.q = q_res;

        return pose_constrain;
    }

    static void save_edge_and_vertex_to_g2o(std::string file_name,
                                            Ceres_pose_graph_3d::MapOfPoses &pose3d_map,
                                            Ceres_pose_graph_3d::VectorOfConstraints &pose_csn_vec)
    {
        FILE *fp = fopen(file_name.c_str(), "w+");
        if (fp != NULL)
        {
            std::cout << "Dump to g2o files:" << file_name << std::endl;
            for (auto it = pose3d_map.begin(); it != pose3d_map.end(); it++)
            {
                Ceres_pose_graph_3d::Pose3D pose3d = it->second;
                fprintf(fp, "VERTEX_SE3:QUAT %d %f %f %f %f %f %f %f\n", (int) it->first,
                        pose3d.p(0), pose3d.p(1), pose3d.p(2),
                        pose3d.q.x(), pose3d.q.y(), pose3d.q.z(), pose3d.q.w());
            }
            for (size_t i = 0; i < pose_csn_vec.size(); i++)
            {
                auto csn = pose_csn_vec[i];
                fprintf(fp, "EDGE_SE3:QUAT %d %d %f %f %f %f %f %f %f", csn.id_begin, csn.id_end,
                        csn.t_be.p(0), csn.t_be.p(1), csn.t_be.p(2),
                        csn.t_be.q.x(), csn.t_be.q.y(), csn.t_be.q.z(), csn.t_be.q.w());
                Eigen::Matrix<double, 6, 6> info_mat;
                info_mat.setIdentity();
                for (size_t c = 0; c < 6; c++)
                    for (size_t r = c; r < 6; r++)
                        fprintf(fp, " %f", info_mat(c, r));
                fprintf(fp, "\n");
            }
            fclose(fp);
            std::cout << "Dump to g2o file OK, file name: " << file_name << std::endl;
        }
        else
            std::cout << "Open file name " << file_name << " error, please check" << endl;
    }

    static void save_edge_and_vertex_to_g2o(std::string file_name,
                                            Ceres_pose_graph_3d::VectorOfPose pose3d_vec,
                                            Ceres_pose_graph_3d::VectorOfConstraints &pose_csn_vec)
    {
        FILE *fp = fopen(file_name.c_str(), "w+");
        if (fp != NULL)
        {
            std::cout << "Dump to g2o files:" << file_name << std::endl;
            for (size_t i = 0; i < pose3d_vec.size(); i++)
                fprintf(fp, "VERTEX_SE3:QUAT %d %f %f %f %f %f %f %f\n", (int)i,
                        pose3d_vec[i].p(0), pose3d_vec[i].p(1), pose3d_vec[i].p(2),
                        pose3d_vec[i].q.x(), pose3d_vec[i].q.y(), pose3d_vec[i].q.z(), pose3d_vec[i].q.w());

            for (size_t i = 0; i < pose_csn_vec.size(); i++)
            {
                auto csn = pose_csn_vec[i];
                fprintf(fp, "EDGE_SE3:QUAT %d %d %f %f %f %f %f %f %f", csn.id_begin, csn.id_end,
                        csn.t_be.p(0), csn.t_be.p(1), csn.t_be.p(2),
                        csn.t_be.q.x(), csn.t_be.q.y(), csn.t_be.q.z(), csn.t_be.q.w());
                Eigen::Matrix<double, 6, 6> info_mat;
                info_mat.setIdentity();
                for (size_t c = 0; c < 6; c++)
                    for (size_t r = c; r < 6; r++)
                        fprintf(fp, " %f", info_mat(c, r));
                fprintf(fp, "\n");
            }
            fclose(fp);
            std::cout << "Dump to g2o file OK, file name: " << file_name << std::endl;
        }
        else
            std::cout << "Open file name " << file_name << " error, please check" << endl;
    }

    void set_downsample_resolution(const float &line_res, const float &plane_res)
    {
        line_res_ = line_res;
        plane_res_ = plane_res;
        down_sample_filter_corner.setLeafSize(line_res_, line_res_, line_res_);
        down_sample_filter_surface.setLeafSize(plane_res_, plane_res_, plane_res_);
    }

    void init(std::string path)
    {
        save_path = path.append("/scene_align");
        COMMON_TOOLS::create_dir(save_path);
        file_logger_commond.set_log_dir(save_path);
        file_logger_timer.set_log_dir(save_path);
        file_logger_commond.init("common.log");
        file_logger_timer.init("timer.log");

        pc_reg.ICP_LINE = 0;
        pc_reg.logger_common = &file_logger_commond;
        pc_reg.m_logger_timer = &file_logger_timer;
        pc_reg.m_timer = &timer;
        pc_reg.para_max_speed = 1000.0;
        pc_reg.para_max_angular_rate = 360 * 57.3;
    }

    void dump_file_name(std::string save_file_name, std::map<int, std::string> &map_filename)
    {
        FILE *fp = fopen(save_file_name.c_str(), "w+");
        if (fp != NULL)
        {
            for (auto it = map_filename.begin(); it != map_filename.end(); it++)
                fprintf(fp, "%d %s\r\n", it->first, it->second.c_str());
            fclose(fp);
        }
    }

    int find_tranfrom_of_two_mappings(PointCloudMap<PT_DATA_TYPE> *pt_cell_map_a,
                                      PointCloudMap<PT_DATA_TYPE> *pt_cell_map_b,
                                      int if_save = 1,
                                      std::string mapping_save_path = std::string(" "))
    {
        pcl::PointCloud<PointType> source_pt_line = pt_cell_map_a->extract_specify_points(FeatureType::e_feature_line);
        pcl::PointCloud<PointType> source_pt_plane = pt_cell_map_a->extract_specify_points(FeatureType::e_feature_plane);
        pcl::PointCloud<PointType> all_pt_a = pt_cell_map_a->get_all_pointcloud();

        down_sample_filter_corner.setInputCloud(source_pt_line.makeShared());
        down_sample_filter_corner.filter(source_pt_line);
        down_sample_filter_surface.setInputCloud(source_pt_plane.makeShared());
        down_sample_filter_surface.filter(source_pt_plane);

        pcl::PointCloud<PointType> target_pt_line = pt_cell_map_b->extract_specify_points(FeatureType::e_feature_line);
        pcl::PointCloud<PointType> target_pt_plane = pt_cell_map_b->extract_specify_points(FeatureType::e_feature_plane);
        pcl::PointCloud<PointType> all_pt_b = pt_cell_map_b->get_all_pointcloud();

        down_sample_filter_corner.setInputCloud(target_pt_line.makeShared());
        down_sample_filter_corner.filter(target_pt_line);
        down_sample_filter_surface.setInputCloud(target_pt_plane.makeShared());
        down_sample_filter_surface.filter(target_pt_plane);

        Eigen::Matrix<double, 3, 1> transform_T = (pt_cell_map_a->get_center() - pt_cell_map_b->get_center()).template cast<double>();
        Eigen::Quaterniond transform_R = Eigen::Quaterniond::Identity();

        pc_reg.current_frame_index = 10000;
        pc_reg.q_w_curr.setIdentity();
        pc_reg.q_w_last.setIdentity();
        pc_reg.t_w_last.setZero();

        pc_reg.t_w_incre = transform_T;
        pc_reg.t_w_curr = transform_T;

        pc_reg.find_out_incremental_transfrom(source_pt_line.makeShared(), source_pt_plane.makeShared(),
                                              target_pt_line.makeShared(), target_pt_plane.makeShared());

        if (if_save)
        {
            PointCloudMap<PT_DATA_TYPE> temp_a, temp_b, temp_res;
            auto eigen_pt_a = PCL_TOOLS::pcl_pts_to_eigen_pts<PT_DATA_TYPE, PointType>(all_pt_a.makeShared());
            down_sample_filter_surface.setInputCloud(all_pt_b.makeShared());
            down_sample_filter_surface.filter(all_pt_b);
            down_sample_filter_surface.setInputCloud(all_pt_a.makeShared());
            down_sample_filter_surface.filter(all_pt_a);

            auto all_pt_temp = pointcloud_transfrom<double, PointType>(all_pt_b, transform_R.toRotationMatrix(), transform_T);

            temp_a.set_point_cloud(PCL_TOOLS::pcl_pts_to_eigen_pts<PT_DATA_TYPE, PointType>(all_pt_a.makeShared()));
            temp_b.set_point_cloud(PCL_TOOLS::pcl_pts_to_eigen_pts<PT_DATA_TYPE, PointType>(all_pt_temp.makeShared()));

            //all_pt_temp = pointcloud_transfrom<double, PointType>(all_pt_b, pc_reg.q_w_curr.toRotationMatrix(), pc_reg.t_w_curr);
            //temp_res.set_point_cloud(PCL_TOOLS::pcl_pts_to_eigen_pts<PT_DATA_TYPE, PointType>(all_pt_temp.makeShared()));

            std::string save_path;
            if (mapping_save_path.compare(std::string(" ")) == 0)
                save_path = save_path;
            else
                save_path = mapping_save_path;

            temp_a.save_to_file(save_path, std::to_string(pair_idx).append("_a.json"));
            temp_b.save_to_file(save_path, std::to_string(pair_idx).append("_b.json"));
            temp_b.save_to_file(save_path, std::to_string(pair_idx).append("_c.json"));
            pair_idx++;
        }
        return pc_reg.inlier_final_threshold;
    };
};

#endif