#ifndef LIVOX_LASER_SCAN_HANDLER_HPP
#define LIVOX_LASER_SCAN_HANDLER_HPP

#include <cmath>
#include <vector>

#define USE_HASH 1
#define SHOW_OPENCV_VIS 0

#if USE_HASH
#include <unordered_map>
#endif

#include <Eigen/Eigen>
#include <Eigen/Eigen>

#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <ros/ros.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include "eigen_math.hpp"
#include "tools/common.h"
#include "tools/pcl_tools.hpp"

#define PCL_DATA_SAVE_DIR "/home/ziv/data/loam_pc"

#define IF_LIVOX_HANDLER_REMOVE 0
#define IF_APPEND 0

using namespace std;
using namespace PCL_TOOLS;
using namespace COMMON_TOOLS;

class LivoxLaser
{
public:
    string SOFT_WARE_VERSION = string("V_0.1_beta");

    enum E_point_type
    {
        e_pt_normal = 0,                      // normal points
        e_pt_000 = 0x0001 << 0,               // points [0,0,0]
        e_pt_too_near = 0x0001 << 1,          // points in short range
        e_pt_reflectivity_low = 0x0001 << 2,  // low reflectivity
        e_pt_reflectivity_high = 0x0001 << 3, // high reflectivity
        e_pt_circle_edge = 0x0001 << 4,       // points near the edge of circle
        e_pt_nan = 0x0001 << 5,               // points with infinite value
        e_pt_small_view_angle = 0x0001 << 6,  // points with large viewed angle
    };

    enum E_feature_type // if and only if normal point can be labeled
    {
        e_label_invalid = -1,
        e_label_unlabeled = 0,
        e_label_corner = 0x0001 << 0,
        e_label_surface = 0x0001 << 1,
        e_label_near_nan = 0x0001 << 2,
        e_label_near_zero = 0x0001 << 3,
        e_label_hight_intensity = 0x0001 << 4,
    };

    // Encode point infos using points intensity, which is more convenient for debugging.
    enum E_intensity_type
    {
        e_I_raw = 0,
        e_I_motion_blur,
        e_I_motion_mix,
        e_I_sigma,
        e_I_scan_angle,
        e_I_curvature,
        e_I_view_angle,
        e_I_time_stamp
    };

    struct PtInfo
    {
        int pt_type = e_pt_normal;
        int pt_label = e_label_unlabeled;
        int idx = 0.f;
        float raw_intensity = 0.f;
        float time_stamp = 0.0;
        float polar_angle = 0.f;
        int polar_direction = 0;
        float polar_dis_sq2 = 0.f; // 点投影到2D平面，距原点距离平方 y_^2+z_^2
        float depth_sq2 = 0.f; // 点３D坐标的平方x^2+y^2+z^2
        float curvature = 0.0;
        float view_angle = 0.0;
        float sigma = 0.0;
        Eigen::Matrix<float, 2, 1> pt_2d_img; // project to X==1 plane
    };

    E_intensity_type default_return_intensity_type = e_I_motion_blur;

    int   pcl_data_save_index = 0;

    float max_fov = 17; // Edge of circle to main axis
    float max_edge_polar_pos = 0;
    float time_internal_pts = 1.0e-5; // 10us = 1e-5
    float m_cx = 0;
    float m_cy = 0;
    int   m_if_save_pcd_file = 0;
    int   input_pts_size;
    double first_receive_time = -1;
    double current_time;
    double last_maximum_time_stamp;
    float thr_corner_curvature = 0.05;
    float thr_surface_curvature = 0.01;
    float minimum_view_angle = 10;
    std::vector<PtInfo> pts_info_vec;
    std::vector<PointType> raw_pts_vec;

#if USE_HASH
    std::unordered_map<PointType, PtInfo *, Pt_hasher, Pt_compare> map_pt_idx; // using hash_map
    std::unordered_map<PointType, PtInfo *, Pt_hasher, Pt_compare>::iterator map_pt_idx_it;
#else
    std::map<PointType, PtInfo *, Pt_compare> map_pt_idx;
    std::map<PointType, PtInfo *, Pt_compare>::iterator map_pt_idx_it;
#endif

    float livox_min_allow_dis = 1.0;
    float livox_min_sigma = 7e-3;

    std::vector<pcl::PointCloud<pcl::PointXYZI>> m_last_laser_scan;

    int     m_img_width = 800;
    int     m_img_heigh = 800;

    ~LivoxLaser() {}

    LivoxLaser()
    {
        cout << "========= Hello, this is livox laser ========" << endl;
        cout << "Compile time:  " << __TIME__ << endl;
        cout << "Softward version: " << SOFT_WARE_VERSION << endl;
        cout << "========= End ========" << endl;
        max_edge_polar_pos = std::pow(tan(max_fov / 57.3) * 1, 2); // 0.093457
    }

    template <typename T>
    T dis2_xy(T x, T y)
    {
        return x * x + y * y;
    }

    template <typename T>
    T depth2_xyz(T x, T y, T z)
    {
        return x * x + y * y + z * z;
    }

    template <typename T>
    T depth_xyz(T x, T y, T z)
    {
        return sqrt(depth2_xyz(x, y, z));
    }

    template <typename T>
    PtInfo *find_pt_info(const T &pt)
    {
        map_pt_idx_it = map_pt_idx.find(pt);
        if (map_pt_idx_it == map_pt_idx.end())
        {
            printf("Input pt is [%lf, %lf, %lf]\r\n", pt.x, pt.y, pt.z);
            printf("Error!!!!\r\n");
            assert(map_pt_idx_it != map_pt_idx.end()); // else, there must be something error happened before.
        }
        return map_pt_idx_it->second;
    }

    void get_features(pcl::PointCloud<PointType> &pc_corners,
                      pcl::PointCloud<PointType> &pc_surface,
                      pcl::PointCloud<PointType> &pc_full,
                      float minimum_blur = 0.0,
                      float maximum_blur = 0.3)
    {
        int corner_num = 0;
        int surface_num = 0;
        int full_num = 0;
        pc_corners.resize(pts_info_vec.size());
        pc_surface.resize(pts_info_vec.size());
        pc_full.resize(pts_info_vec.size());
        float maximum_idx = maximum_blur * pts_info_vec.size();
        float minimum_idx = minimum_blur * pts_info_vec.size();
        int pt_critical_rm_mask = e_pt_000 | e_pt_nan;

        for (size_t i = 0; i < pts_info_vec.size(); i++)
        {
            if (pts_info_vec[i].idx > maximum_idx || pts_info_vec[i].idx < minimum_idx)
                continue;

            if ((pts_info_vec[i].pt_type & pt_critical_rm_mask) == 0)
            {
                if (pts_info_vec[i].pt_label & e_label_corner)
                {
                    if (pts_info_vec[i].pt_type != e_pt_normal)
                        continue;
                    if (pts_info_vec[i].depth_sq2 < std::pow(30, 2))
                    {
                        pc_corners.points[corner_num] = raw_pts_vec[i];
                        pc_corners.points[corner_num].intensity = pts_info_vec[i].time_stamp;
                        corner_num++;
                    }
                }
                else if (pts_info_vec[i].pt_label & e_label_surface)
                {
                    if (pts_info_vec[i].depth_sq2 < std::pow(1000, 2))
                    {
                        pc_surface.points[surface_num] = raw_pts_vec[i];
                        pc_surface.points[surface_num].intensity = pts_info_vec[i].time_stamp;
                        surface_num++;
                    }
                }

                pc_full.points[full_num] = raw_pts_vec[i];
                pc_full.points[full_num].intensity = pts_info_vec[i].time_stamp;
                full_num++;
            }
        }

        pc_corners.resize(corner_num);
        pc_surface.resize(surface_num);
        pc_full.resize(full_num);
    }

    template <typename T>
    void set_intensity(T &pt, const E_intensity_type &i_type = e_I_motion_blur)
    {
        PtInfo *pt_info = find_pt_info(pt);
        switch (i_type)
        {
        case (e_I_raw):
            pt.intensity = pt_info->raw_intensity;
            break;
        case (e_I_motion_blur):
            pt.intensity = (float)(pt_info->idx / input_pts_size);
            assert(pt.intensity <= 1.0 && pt.intensity >= 0.0);
            break;
        case (e_I_motion_mix):
            pt.intensity = 0.1 * ((float) pt_info->idx + 1) / (float) input_pts_size + (int) (pt_info->raw_intensity);
            break;
        case (e_I_scan_angle):
            pt.intensity = pt_info->polar_angle;
            break;
        case (e_I_curvature):
            pt.intensity = pt_info->curvature;
            break;
        case (e_I_view_angle):
            pt.intensity = pt_info->view_angle;
            break;
        case (e_I_time_stamp):
            pt.intensity = pt_info->time_stamp;
        default:
            pt.intensity = ((float) pt_info->idx + 1) / (float) input_pts_size;
        }
        return;
    }

    void add_mask_of_point(PtInfo *pt_infos, const E_point_type &pt_type, int neighbor_count = 0)
    {
        int idx = pt_infos->idx;
        pt_infos->pt_type |= pt_type; // bitwise OR

        if (neighbor_count > 0) // 为附近neighbor点赋予pt_type
        {
            for (int i = -neighbor_count; i < neighbor_count; i++)
            {
                idx = pt_infos->idx + i;

                if (i != 0 && idx >= 0 && idx < (int)pts_info_vec.size())
                    pts_info_vec[idx].pt_type |= pt_type;
            }
        }
    }

    void eval_point(PtInfo *pt_info)
    {
        if (pt_info->depth_sq2 < livox_min_allow_dis * livox_min_allow_dis)
            add_mask_of_point(pt_info, e_pt_too_near);

        pt_info->sigma = pt_info->raw_intensity / pt_info->polar_dis_sq2;

        if (pt_info->sigma < livox_min_sigma)
            add_mask_of_point(pt_info, e_pt_reflectivity_low);
    }

    void compute_features()
    {
        unsigned int pts_size = raw_pts_vec.size();
        size_t curvature_ssd_size = 2;
        int critical_rm_point = e_pt_000 | e_pt_nan;
        float neighbor_accumulate_xyz[3] = {0.0, 0.0, 0.0};

        for (size_t idx = curvature_ssd_size; idx < pts_size - curvature_ssd_size; idx++)
        {
            if (pts_info_vec[idx].pt_type & critical_rm_point)
                continue;

            neighbor_accumulate_xyz[0] = 0.0;
            neighbor_accumulate_xyz[1] = 0.0;
            neighbor_accumulate_xyz[2] = 0.0;

            for (size_t i = 1; i <= curvature_ssd_size; i++)
            {
                if ((pts_info_vec[idx + i].pt_type & e_pt_000) || (pts_info_vec[idx - i].pt_type & e_pt_000))
                {
                    if (i == 1)
                        pts_info_vec[idx].pt_label |= e_label_near_zero;
                    else
                        pts_info_vec[idx].pt_label = e_label_invalid;
                    break;
                }
                else if ((pts_info_vec[idx + i].pt_type & e_pt_nan) || (pts_info_vec[idx - i].pt_type & e_pt_nan))
                {
                    if (i == 1)
                        pts_info_vec[idx].pt_label |= e_label_near_nan;
                    else
                        pts_info_vec[idx].pt_label = e_label_invalid;
                    break;
                }
                else
                {
                    neighbor_accumulate_xyz[0] += raw_pts_vec[idx + i].x + raw_pts_vec[idx - i].x;
                    neighbor_accumulate_xyz[1] += raw_pts_vec[idx + i].y + raw_pts_vec[idx - i].y;
                    neighbor_accumulate_xyz[2] += raw_pts_vec[idx + i].z + raw_pts_vec[idx - i].z;
                }
            }

            if(pts_info_vec[idx].pt_label == e_label_invalid)
                continue;

            neighbor_accumulate_xyz[0] -= curvature_ssd_size * 2 * raw_pts_vec[idx].x;
            neighbor_accumulate_xyz[1] -= curvature_ssd_size * 2 * raw_pts_vec[idx].y;
            neighbor_accumulate_xyz[2] -= curvature_ssd_size * 2 * raw_pts_vec[idx].z;
            pts_info_vec[idx].curvature = neighbor_accumulate_xyz[0] * neighbor_accumulate_xyz[0] +
                                          neighbor_accumulate_xyz[1] * neighbor_accumulate_xyz[1] +
                                          neighbor_accumulate_xyz[2] * neighbor_accumulate_xyz[2];

            Eigen::Matrix<float, 3, 1> vec_a(raw_pts_vec[idx].x, raw_pts_vec[idx].y, raw_pts_vec[idx].z);
            Eigen::Matrix<float, 3, 1> vec_b(raw_pts_vec[idx + curvature_ssd_size].x - raw_pts_vec[idx - curvature_ssd_size].x,
                                             raw_pts_vec[idx + curvature_ssd_size].y - raw_pts_vec[idx - curvature_ssd_size].y,
                                             raw_pts_vec[idx + curvature_ssd_size].z - raw_pts_vec[idx - curvature_ssd_size].z);
            
            pts_info_vec[idx].view_angle = Eigen_math::vector_angle(vec_a  , vec_b, 1) * 57.3;

            if (pts_info_vec[idx].view_angle > minimum_view_angle)
            {
                if(pts_info_vec[idx].curvature < thr_surface_curvature)
                    pts_info_vec[idx].pt_label |= e_label_surface;

                float sq2_diff = 0.1;

                if (pts_info_vec[idx].curvature > thr_corner_curvature)
                    if (pts_info_vec[idx].depth_sq2 <= pts_info_vec[idx - curvature_ssd_size].depth_sq2 &&
                        pts_info_vec[idx].depth_sq2 <= pts_info_vec[idx + curvature_ssd_size].depth_sq2)
                        if (abs(pts_info_vec[idx].depth_sq2 - pts_info_vec[idx - curvature_ssd_size].depth_sq2) < sq2_diff * pts_info_vec[idx].depth_sq2 ||
                            abs(pts_info_vec[idx].depth_sq2 - pts_info_vec[idx + curvature_ssd_size].depth_sq2) < sq2_diff * pts_info_vec[idx].depth_sq2)
                            pts_info_vec[idx].pt_label |= e_label_corner;
            }
        }
    }

    template <typename T>
    int projection_scan_3d_2d(pcl::PointCloud<T> &laserCloudIn, std::vector<float> &scan_id_index)
    {
        unsigned int pts_size = laserCloudIn.size();
        pts_info_vec.clear();
        pts_info_vec.resize(pts_size);
        raw_pts_vec.resize(pts_size);
        map_pt_idx.clear();
        map_pt_idx.reserve(pts_size);
        scan_id_index.resize(pts_size);
        std::vector<int> edge_idx;
        edge_idx.clear();
        std::vector<int> split_idx;
        std::vector<int> zero_idx;
        input_pts_size = 0;

        for (unsigned int idx = 0; idx < pts_size; idx++)
        {
            raw_pts_vec[idx] = laserCloudIn.points[idx];
            PtInfo *pt_info = &pts_info_vec[idx];
            map_pt_idx.insert(std::make_pair(laserCloudIn.points[idx], pt_info));
            pt_info->raw_intensity = laserCloudIn.points[idx].intensity;
            pt_info->idx = idx;
            pt_info->time_stamp = current_time + (float)idx * time_internal_pts;
            last_maximum_time_stamp = pt_info->time_stamp;
            input_pts_size++;

            if (!std::isfinite(laserCloudIn.points[idx].x) ||
                !std::isfinite(laserCloudIn.points[idx].y) ||
                !std::isfinite(laserCloudIn.points[idx].z))
            {
                add_mask_of_point(pt_info, e_pt_nan); // 如果点的xyz值不是有限的，则标记为pt_nan
                continue;
            }

            if (laserCloudIn.points[idx].x == 0)
            {
                if (idx == 0)
                    ROS_INFO_ONCE("First point should be normal!!!"); // TODO: handle this case.
                else
                {
                    pt_info->pt_2d_img = pts_info_vec[idx - 1].pt_2d_img;
                    pt_info->polar_dis_sq2 = pts_info_vec[idx - 1].polar_dis_sq2;
                    add_mask_of_point(pt_info, e_pt_000);
                    continue;
                }
            }

            map_pt_idx.insert(std::make_pair(laserCloudIn.points[idx], pt_info));
            pt_info->depth_sq2 = depth2_xyz(laserCloudIn.points[idx].x, laserCloudIn.points[idx].y, laserCloudIn.points[idx].z);
            pt_info->pt_2d_img << laserCloudIn.points[idx].y / laserCloudIn.points[idx].x, laserCloudIn.points[idx].z / laserCloudIn.points[idx].x;
            pt_info->polar_dis_sq2 = dis2_xy(pt_info->pt_2d_img(0), pt_info->pt_2d_img(1));
            eval_point(pt_info); // 检查点是否距离太远或intensity太低

            if (pt_info->polar_dis_sq2 > max_edge_polar_pos)
                add_mask_of_point(pt_info, e_pt_circle_edge, 2); // 靠近fov边缘

            if (idx >= 1) // 每两个split(edge或zero)间隔至少50个点
            {
                float dis_incre = pt_info->polar_dis_sq2 - pts_info_vec[idx - 1].polar_dis_sq2;

                if (dis_incre > 0) // far away from zero
                    pt_info->polar_direction = 1;

                if (dis_incre < 0) // move toward zero
                    pt_info->polar_direction = -1;

                if (pt_info->polar_direction == -1 && pts_info_vec[idx - 1].polar_direction == 1)
                {
                    if (edge_idx.size() == 0 || (idx - split_idx[split_idx.size() - 1]) > 50)
                    {
                        split_idx.push_back(idx);
                        edge_idx.push_back(idx);
                        continue;
                    }
                }

                if (pt_info->polar_direction == 1 && pts_info_vec[idx - 1].polar_direction == -1)
                {
                    if (zero_idx.size() == 0 || (idx - split_idx[split_idx.size() - 1]) > 50)
                    {
                        split_idx.push_back(idx);
                        zero_idx.push_back(idx);
                        continue;
                    }
                }
            }
        }

        split_idx.push_back(pts_size - 1);

        int val_index = 0;
        int pt_angle_index = 0;
        float scan_angle = 0;
        int internal_size = 0;

        for (unsigned int idx = 0; idx < pts_size; idx++)
        {
            if (idx == 0 || idx > (unsigned int)split_idx[val_index + 1])
            {
                if (idx > (unsigned int)split_idx[val_index + 1])
                    val_index++;

                internal_size = split_idx[val_index + 1] - split_idx[val_index]; // 当前idx所处的split区间长度

                if (pts_info_vec[split_idx[val_index + 1]].polar_dis_sq2 > 10000)
                {
                    pt_angle_index = split_idx[val_index + 1] - (int)(internal_size * 0.20);
                    scan_angle = atan2(pts_info_vec[pt_angle_index].pt_2d_img(1), pts_info_vec[pt_angle_index].pt_2d_img(0)) * 57.3;
                    scan_angle = scan_angle + 180.0;
                }
                else
                {
                    pt_angle_index = split_idx[val_index + 1] - (int) (internal_size * 0.80);
                    scan_angle = atan2(pts_info_vec[pt_angle_index].pt_2d_img(1), pts_info_vec[pt_angle_index].pt_2d_img(0)) * 57.3;
                    scan_angle = scan_angle + 180.0;
                }
            }
            pts_info_vec[idx].polar_angle = scan_angle;
            scan_id_index[idx] = scan_angle;
        }

        return split_idx.size() - 1;
    }

    template <typename T>
    void split_laser_scan(const int clutter_size, const pcl::PointCloud<T> &laserCloudIn,
                          const std::vector<float> &scan_id_index,
                          std::vector<pcl::PointCloud<PointType>> &laserCloudScans)
    {
        std::vector< std::vector<int> > pts_mask;
        laserCloudScans.resize(clutter_size);
        pts_mask.resize(clutter_size);
        PointType point;
        int scan_idx = 0;

        for (unsigned int i = 0; i < laserCloudIn.size(); i++)
        {
            point = laserCloudIn.points[i];

            if (i > 0 && ((scan_id_index[i]) != (scan_id_index[i - 1])))
            {
                scan_idx++;
                pts_mask[scan_idx].reserve(5000);
            }

            laserCloudScans[scan_idx].push_back(point);
            pts_mask[scan_idx].push_back(pts_info_vec[i].pt_type);
        }
        laserCloudScans.resize(scan_idx); // 舍弃最后一个split后的数据

        int remove_point_pt_type = e_pt_000 | e_pt_too_near | e_pt_nan;

        for (unsigned int i = 0; i < laserCloudScans.size(); i++)
        {
            int scan_avail_num = 0;
            for (unsigned int idx = 0; idx < laserCloudScans[i].size(); idx++)
            {
                if ((pts_mask[i][idx] & remove_point_pt_type) == 0)
                {
                    if (laserCloudScans[i].points[idx].x == 0)
                    {
                        ROS_WARN("Error!!! Mask = %d\r\n", pts_mask[i][idx]);
                        assert(laserCloudScans[i].points[idx].x != 0);
                        continue;
                    }
                    laserCloudScans[i].points[scan_avail_num] = laserCloudScans[i].points[idx];
                    set_intensity(laserCloudScans[i].points[scan_avail_num], default_return_intensity_type); // intensity = idx / pts_size
                    scan_avail_num++;
                }
            }
            laserCloudScans[i].resize(scan_avail_num);
        }
    }

    template <typename T>
    std::vector<pcl::PointCloud<pcl::PointXYZI>> extract_laser_features(pcl::PointCloud<T> &laserCloudIn, double time_stamp = -1)
    {
        assert(time_stamp >= 0.0);
        if(time_stamp <= 0.0000001 || (time_stamp < last_maximum_time_stamp)) // old firmware, without timestamp
            current_time = last_maximum_time_stamp;
        else
            current_time = time_stamp - first_receive_time;

        if (first_receive_time <= 0)
            first_receive_time = time_stamp;

        std::vector<pcl::PointCloud<PointType>> laserCloudScans, temp_laser_scans;
        std::vector<float> scan_id_index;
        laserCloudScans.clear();
        map_pt_idx.clear();

        if (m_if_save_pcd_file)
        {
            stringstream ss;
            ss << PCL_DATA_SAVE_DIR << "/pc_" << pcl_data_save_index << ".pcd" << endl;
            pcl_data_save_index = pcl_data_save_index + 1;
            std::cout << "Save file = " << ss.str() << endl;
            pcl::io::savePCDFileASCII(ss.str(), laserCloudIn);
        }

        int clutter_size = projection_scan_3d_2d(laserCloudIn, scan_id_index);
        compute_features();
        if (clutter_size == 0)
            return laserCloudScans;
        else
        {
            split_laser_scan(clutter_size, laserCloudIn, scan_id_index, laserCloudScans);
            return laserCloudScans;
        }
    }
};

#endif // LIVOX_LASER_SCAN_HANDLER_HPP
