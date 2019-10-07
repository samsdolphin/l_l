#pragma once
#define USE_HASH 1

#include "common_tools.h"
#include "pcl_tools.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/types_c.h>

#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/prettywriter.h"

#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>

#include <Eigen/Eigen>
#include <map>
#include <vector>
#include <mutex>
#include <thread>
#include <iomanip>
#include <boost/format.hpp>
#include <math.h>
#if USE_HASH
#include <unordered_map>
#else
#endif

#define IF_COV_INIT_IDENTITY 0
#define IF_EIGEN_REPLACE 1

typedef double COMP_TYPE;
typedef pcl::PointXYZI pcl_pt;

enum FeatureType
{
    e_feature_sphere = 0,
    e_feature_line = 1,
    e_feature_plane = 2
};

template <typename DATA_TYPE>
class PointCloudCell
{
public:
    typedef Eigen::Matrix<DATA_TYPE, 3, 1> Eigen_Point;
    typedef Eigen::Matrix<DATA_TYPE, 3, 1> PT_TYPE;
    DATA_TYPE resolution_;
    Eigen::Matrix<DATA_TYPE, 3, 1> center;

    //private:
    Eigen::Matrix<COMP_TYPE, 3, 1> xyz_sum;
    Eigen::Matrix<COMP_TYPE, 3, 1> m_mean;
    Eigen::Matrix<COMP_TYPE, 3, 3> m_cov_mat;
    Eigen::Matrix<COMP_TYPE, 3, 3> m_icov_mat;

    /** \brief Eigen vectors of voxel covariance matrix */
    Eigen::Matrix<COMP_TYPE, 3, 3> m_eigen_vec; // Eigen vector of covariance matrix
    Eigen::Matrix<COMP_TYPE, 3, 1> m_eigen_val; // Eigen value of covariance values

    FeatureType feature_type = e_feature_sphere;
    double m_feature_determine_threshold = 1.0 / 3.0;
    Eigen::Matrix<COMP_TYPE, 3, 1> feature_vector;

public:
    pcl::VoxelGridCovariance<pcl_pt> m_pcl_voxel_cell;
    std::vector<PT_TYPE> point_vec;
    pcl::PointCloud<pcl_pt> pcl_pc;
    DATA_TYPE m_cov_det_sqrt;
    int m_if_compute_using_pcl = false;
    bool mean_need_update = true;
    bool covmat_need_update = true;
    bool icovmat_need_update = true;
    bool pcl_voxelgrid_need_update = true;
    size_t maximum_points_size = (size_t)1e4;
    std::mutex *m_mutex_cell;

    PointCloudCell()
    {
        m_mutex_cell = new std::mutex();
    };

    ~PointCloudCell()
    {
        m_mutex_cell->try_lock();
        m_mutex_cell->unlock();
        clear_data();
    };

    template <typename T1, typename T2>
    static void save_mat_to_jason_writter(T1 &writer, const std::string &name, const T2 &eigen_mat)
    {
        writer.Key(name.c_str()); // output a key,
        writer.StartArray(); // Between StartArray()/EndArray(),
        for (size_t i = 0; i < (size_t)(eigen_mat.cols() * eigen_mat.rows()); i++)
            writer.Double(eigen_mat(i));
        writer.EndArray();
    }

    std::string to_json_string()
    {
        get_icovmat(); // update data
        rapidjson::Document     document;
        rapidjson::StringBuffer sb;
        // See more detail in https://github.com/Tencent/rapidjson/blob/master/example/simplewriter/simplewriter.cpp

        rapidjson::Writer< rapidjson::StringBuffer > writer(sb);

        writer.StartObject(); // Between StartObject()/EndObject(),
        writer.SetMaxDecimalPlaces(1000); // like set_precision

        writer.Key("Pt_num");
        writer.Int(point_vec.size());
        writer.Key("Res"); // output a key
        writer.Double(resolution_);
        save_mat_to_jason_writter(writer, "Center", center);
        save_mat_to_jason_writter(writer, "Mean", m_mean);
        if (point_vec.size() > 5)
        {
            save_mat_to_jason_writter(writer, "Cov", m_cov_mat);
            save_mat_to_jason_writter(writer, "Icov", m_icov_mat);
            save_mat_to_jason_writter(writer, "Eig_vec", m_eigen_vec);
            save_mat_to_jason_writter(writer, "Eig_val", m_eigen_val);
        }
        else
        {
            Eigen::Matrix<COMP_TYPE, 3, 3> I;
            Eigen::Matrix<COMP_TYPE, 3, 1> Vec3d;
            I.setIdentity();
            Vec3d << 1.0, 1.0, 1.0;
            save_mat_to_jason_writter(writer, "Cov", I);
            save_mat_to_jason_writter(writer, "Icov", I);
            save_mat_to_jason_writter(writer, "Eig_vec", I);
            save_mat_to_jason_writter(writer, "Eig_val", Vec3d);
        }
        writer.Key("Pt_vec");
        writer.SetMaxDecimalPlaces(3);
        writer.StartArray();
        for (unsigned i = 0; i < point_vec.size(); i++)
        {
            writer.Double(point_vec[i](0));
            writer.Double(point_vec[i](1));
            writer.Double(point_vec[i](2));
        }
        writer.EndArray();
        writer.SetMaxDecimalPlaces(1000);

        writer.EndObject();

        return std::string(sb.GetString());
    }

    void save_to_file(const std::string &path = std::string("./"), const std::string &file_name = std::string(""))
    {
        std::stringstream str_ss;
        COMMON_TOOLS::create_dir(path);
        if (file_name.compare("") == 0)
        {
            str_ss << path << "/" << std::setprecision(3)
                   << center(0) << "_"
                   << center(1) << "_"
                   << center(2) << ".json";
        }
        else
            str_ss << path << "/" << file_name.c_str();

        std::fstream ofs;
        ofs.open(str_ss.str().c_str(), std::ios_base::out);
        //std::cout << "Save to " << str_ss.str();
        if (ofs.is_open())
        {
            ofs << to_json_string();
            ofs.close();
            //std::cout << " Successful. Number of points = " << point_vec.size() << std::endl;
        }
        else
            std::cout << " Fail !!!" << std::endl;
    }

    void pcl_voxelgrid_update()
    {
        if (pcl_voxelgrid_need_update)
        {
            m_pcl_voxel_cell.setLeafSize(200.0, 200.0, 200.0);
            m_pcl_voxel_cell.setInputCloud(pcl_pc.makeShared());
            m_pcl_voxel_cell.filter(true);
        }
        pcl_voxelgrid_need_update = false;
    }

    void set_data_need_update(int if_update_sum = 0)
    {
        mean_need_update = true;
        covmat_need_update = true;
        icovmat_need_update = true;
        pcl_voxelgrid_need_update = true;
        if (if_update_sum)
        {
            xyz_sum.setZero();
            for (size_t i = 0; i< point_vec.size(); i++)
                xyz_sum += point_vec[i].template cast<COMP_TYPE>();
        }
    }

    int get_points_count()
    {
        return point_vec.size();
    }

    Eigen::Matrix<DATA_TYPE, 3, 1> get_center()
    {
        return center.template cast<DATA_TYPE>();
    }

    Eigen::Matrix<DATA_TYPE, 3, 1> get_mean()
    {
        if (mean_need_update)
        {
            if (m_if_compute_using_pcl)
            {
                pcl_voxelgrid_update();
                mean_need_update = false;
                auto leaf = m_pcl_voxel_cell.getLeaf(center);
                assert(leaf != nullptr);
                return leaf->getMean().template cast<DATA_TYPE>();
            }
            set_data_need_update();
            m_mean = xyz_sum / ((DATA_TYPE)(point_vec.size()));
        }
        mean_need_update = false;
        return m_mean.template cast<DATA_TYPE>();
    }

    Eigen::Matrix<DATA_TYPE, 3, 3> robust_covmat()
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<COMP_TYPE, 3, 3>> eigensolver;
        Eigen::Matrix<COMP_TYPE, 3, 3> eigen_val;
        // Eigen values less than a threshold of max eigen value are inflated to a set fraction of the max eigen value.
        COMP_TYPE min_covar_eigvalue;
        COMP_TYPE min_covar_eigvalue_mult_ = 0.01; // pcl: 0.01
        if (!IF_EIGEN_REPLACE)
            min_covar_eigvalue_mult_ = 0;
        int pt_num = point_vec.size();
        m_cov_mat = (m_cov_mat - 2 * pt_num * m_mean * m_mean.transpose()) / pt_num + m_mean * m_mean.transpose();
        m_cov_mat *= (pt_num - 1.0) / pt_num;

        // std::cout << m_cov_mat << std::endl;
        eigensolver.compute(m_cov_mat);
        eigen_val = eigensolver.eigenvalues().asDiagonal();
        m_eigen_val = eigensolver.eigenvalues();
        m_eigen_vec = eigensolver.eigenvectors();

        // Avoids matrices near singularities (eq 6.11)[Magnusson 2009]
        min_covar_eigvalue = min_covar_eigvalue_mult_ * eigen_val(2, 2);
        if (eigen_val(0, 0) < min_covar_eigvalue)
        {
            eigen_val(0, 0) = min_covar_eigvalue;

            if (eigen_val(1, 1) < min_covar_eigvalue)
                eigen_val(1, 1) = min_covar_eigvalue;

            m_cov_mat = m_eigen_vec * eigen_val * m_eigen_vec.inverse();
            if (!std::isfinite(m_cov_mat(0, 0)))
                m_cov_mat.setIdentity();
        }
        return m_cov_mat.template cast<DATA_TYPE>();
    }

    Eigen::Matrix<DATA_TYPE, 3, 3> get_covmat()
    {
        if (covmat_need_update)
        {
            get_mean();
            if (m_if_compute_using_pcl)
            {
                pcl_voxelgrid_update();
                auto leaf = m_pcl_voxel_cell.getLeaf(center);
                assert(leaf != nullptr);
                return leaf->getCov().template cast<DATA_TYPE>();
            }
            size_t pt_size = point_vec.size();
            if (IF_COV_INIT_IDENTITY)
                m_cov_mat.setIdentity();
            else
                m_cov_mat.setZero();
            for (size_t i = 0; i < pt_size; i++)
                m_cov_mat = m_cov_mat + (point_vec[i] * point_vec[i].transpose()).template cast<COMP_TYPE>();
            robust_covmat();
        }
        covmat_need_update = false;
        return m_cov_mat.template cast<DATA_TYPE>();
    }

    Eigen::Matrix<DATA_TYPE, 3, 3> get_icovmat()
    {
        if (icovmat_need_update)
        {
            get_covmat();
            if (m_if_compute_using_pcl)
            {
                pcl_voxelgrid_update();
                auto leaf = m_pcl_voxel_cell.getLeaf(center);
                assert(leaf != nullptr);
                return leaf->getInverseCov().template cast<DATA_TYPE>();
            }
            m_icov_mat = m_cov_mat.inverse();
            if (!std::isfinite(m_icov_mat(0, 0)))
                m_icov_mat.setIdentity();
        }
        icovmat_need_update = false;
        return m_icov_mat.template cast<DATA_TYPE>();
    }

    pcl::PointCloud<pcl_pt> get_pointcloud()
    {
        std::unique_lock<std::mutex> lock(*m_mutex_cell);
        pcl::PointCloud<pcl_pt> pt_temp = pcl_pc;
        return pt_temp;
    }

    std::vector<PT_TYPE> get_pointcloud_eigen()
    {
        std::unique_lock<std::mutex> lock(*m_mutex_cell);
        return point_vec;
    }

    void set_pointcloud(pcl::PointCloud<pcl_pt>& pc_in)
    {
        std::unique_lock<std::mutex> lock(*m_mutex_cell);
        pcl_pc = pc_in;
        point_vec = PCL_TOOLS::pcl_pts_to_eigen_pts<float, pcl_pt >(pc_in.makeShared());
    }

    void clear_data()
    {
        std::unique_lock<std::mutex> lock(*m_mutex_cell);
        point_vec.clear();
        pcl_pc.clear();
        m_mean.setZero();
        xyz_sum.setZero();
        m_cov_mat.setZero();
    }

    PointCloudCell(const PT_TYPE &cell_center, const DATA_TYPE &res = 1.0)
    {
        m_mutex_cell = new std::mutex();
        clear_data();
        resolution_ = res;
        maximum_points_size = (int)res * 100.0;
        point_vec.reserve(maximum_points_size);
        if (m_if_compute_using_pcl)
            pcl_pc.reserve(maximum_points_size);
        center = cell_center;
    }

    void append_pt(const PT_TYPE &pt)
    {
        std::unique_lock<std::mutex> lock(*m_mutex_cell);
        pcl_pc.push_back(PCL_TOOLS::eigen_to_pcl_pt<pcl_pt>(pt));
        point_vec.push_back(pt);
        if (point_vec.size() > maximum_points_size)
        {
            maximum_points_size *= 10;
            point_vec.reserve(maximum_points_size);
        }
        xyz_sum = xyz_sum + pt.template cast<COMP_TYPE>();

        set_data_need_update();
    }

    void set_target_pc(const std::vector<PT_TYPE> &pt_vec)
    {
        std::unique_lock<std::mutex> lock(*m_mutex_cell);
        // "The three-dimensional normal-distributions transform: an efficient representation for registration, surface analysis, and loop detection"
        int pt_size = pt_vec.size();
        clear_data();
        point_vec.reserve(maximum_points_size);
        for (int i = 0; i < pt_size; i++)
            append_pt(pt_vec[i]);
        set_data_need_update();
    };

    FeatureType determine_feature(int if_recompute)
    {
        if (if_recompute)
            set_data_need_update(1);

        get_covmat();
        feature_type = e_feature_sphere;

        if (point_vec.size() < 10)
        {
            feature_type = e_feature_sphere;
            feature_vector << 0, 0, 0;
            return e_feature_sphere;
        }

        if ((center.template cast<float>() - m_mean.template cast<float>()).norm() > resolution_ * 0.75)
        {
            feature_type = e_feature_sphere;
            feature_vector << 0, 0, 0;
            return e_feature_sphere;
        }

        if (m_eigen_val[2] * 0.1 > m_eigen_val[1])
        {
            feature_type = e_feature_line;
            feature_vector = m_eigen_vec.block<3, 1>(0, 2);
        }
        if ((m_eigen_val[0] < m_feature_determine_threshold * m_eigen_val[1]) && (m_eigen_val[1] > 0.5 * m_eigen_val[2]))
        {
            feature_type = e_feature_plane;
            feature_vector = m_eigen_vec.block<3, 1>(0, 0);
        }
        return feature_type;
    }
};

template <typename DATA_TYPE>
class PointCloudMap
{
public:
    typedef Eigen::Matrix<DATA_TYPE, 3, 1> PT_TYPE;
    typedef Eigen::Matrix<DATA_TYPE, 3, 1> Eigen_Point;
    typedef PointCloudCell<DATA_TYPE> PC_CELL;

    DATA_TYPE m_x_min, m_x_max;
    DATA_TYPE m_y_min, m_y_max;
    DATA_TYPE m_z_min, m_z_max;
    DATA_TYPE resolution_; // resolution mean the distance of a cute to its bound.
    COMMON_TOOLS::Timer m_timer;
    
    std::vector<PC_CELL *> pc_cell_vec;
    int scale = 10;
    int THETA_RES = (int)(12 * scale);
    int BETA_RES = (int)(6 * scale);
    std::mutex *m_mapping_mutex;
    std::mutex *octotree_mutex;
    std::mutex *m_mutex_addcell;
    std::string json_file_name;
    float m_ratio_nonzero_line, m_ratio_nonzero_plane;

#if USE_HASH
    typedef std::unordered_map<PT_TYPE, PC_CELL *, PCL_TOOLS::Eigen_pt_hasher, PCL_TOOLS::Eigen_pt_compare> MAP_PT_CELL;
    typedef typename std::unordered_map<PT_TYPE, PC_CELL *, PCL_TOOLS::Eigen_pt_hasher, PCL_TOOLS::Eigen_pt_compare>::iterator MAP_PT_CELL_IT;
#else
    typedef std::map<PT_TYPE, PC_CELL *, PCL_TOOLS::Pt_compare> MAP_PT_CELL;
    typedef typename std::map<PT_TYPE, PC_CELL *, PCL_TOOLS::Pt_compare>::iterator MAP_PT_CELL_IT;
#endif

    MAP_PT_CELL map_pt_cell; // using hash_map
    MAP_PT_CELL_IT map_pt_cell_it;

    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> feature_img_line, feature_img_plane;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> feature_img_line_roi, feature_img_plane_roi;
    Eigen::Matrix<float, 3, 3> m_eigen_R, m_eigen_R_roi;
    float roi_range;

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree = pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(0.0001);
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cells_center = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    int m_initialized = false;

    PointCloudMap()
    {
        m_mapping_mutex = new std::mutex();
        m_mutex_addcell = new std::mutex();
        octotree_mutex = new std::mutex();
        m_x_min = std::numeric_limits<DATA_TYPE>::max();
        m_y_min = std::numeric_limits<DATA_TYPE>::max();
        m_z_min = std::numeric_limits<DATA_TYPE>::max();

        m_x_max = std::numeric_limits<DATA_TYPE>::min();
        m_y_max = std::numeric_limits<DATA_TYPE>::min();
        m_z_max = std::numeric_limits<DATA_TYPE>::min();
        m_cells_center->reserve(1e5);
        set_resolution(1.0);
    };

    ~PointCloudMap()
    {
        m_mapping_mutex->try_lock();
        m_mapping_mutex->unlock();

        m_mutex_addcell->try_lock();
        m_mutex_addcell->unlock();

        octotree_mutex->try_lock();
        octotree_mutex->unlock();
    }

    int get_cells_size()
    {
        return map_pt_cell.size();
    }

    PT_TYPE find_cell_center(const PT_TYPE &pt)
    {
        PT_TYPE cell_center;
        DATA_TYPE GRID_SIZE = resolution_ * 1.0;
        DATA_TYPE HALF_GRID_SIZE = resolution_ * 0.5; // ?

        cell_center(0) = (std::round((pt(0) - m_x_min - HALF_GRID_SIZE) / GRID_SIZE)) * GRID_SIZE + m_x_min + HALF_GRID_SIZE;
        cell_center(1) = (std::round((pt(1) - m_y_min - HALF_GRID_SIZE) / GRID_SIZE)) * GRID_SIZE + m_y_min + HALF_GRID_SIZE;
        cell_center(2) = (std::round((pt(2) - m_z_min - HALF_GRID_SIZE) / GRID_SIZE)) * GRID_SIZE + m_z_min + HALF_GRID_SIZE;
        return cell_center;
    }

    void clear_data()
    {
        for (MAP_PT_CELL_IT it = map_pt_cell.begin(); it != map_pt_cell.end(); it++)
        {
            it->second->clear_data();
            delete it->second;
        }
        map_pt_cell.clear();
        pc_cell_vec.clear();
        m_cells_center->clear();
        octree.deleteTree();
    }

    void set_point_cloud(const std::vector<PT_TYPE> &input_pt_vec)
    {
        clear_data();
        for (size_t i = 0; i < input_pt_vec.size(); i++)
        {
            m_x_min = std::min(input_pt_vec[i](0), m_x_min);
            m_y_min = std::min(input_pt_vec[i](1), m_y_min);
            m_z_min = std::min(input_pt_vec[i](2), m_z_min);

            m_x_max = std::max(input_pt_vec[i](0), m_x_max);
            m_y_max = std::max(input_pt_vec[i](1), m_y_max);
            m_z_max = std::max(input_pt_vec[i](2), m_z_max);
        }

        for (size_t i = 0; i < input_pt_vec.size(); i++)
        {
            PC_CELL *cell = find_cell(input_pt_vec[i]);
            cell->append_pt(input_pt_vec[i]);
        }

        octree.setInputCloud(m_cells_center);
        octree.addPointsFromInputCloud();
        //std::cout << "*** set_point_cloud octree initialization finish ***" << std::endl;
        m_initialized = true;
    }

    void append_cloud(const std::vector<PT_TYPE> &input_pt_vec, int if_vervose = false)
    {
        m_timer.tic(__FUNCTION__);
        m_mapping_mutex->lock();
        int current_size = get_cells_size();
        if (current_size == 0)
        {
            set_point_cloud(input_pt_vec);
            m_mapping_mutex->unlock();
        }
        else
        {
            m_mapping_mutex->unlock();
            for (size_t i = 0; i < input_pt_vec.size(); i++)
            {
                PointCloudCell<DATA_TYPE> *cell = find_cell(input_pt_vec[i]);
                cell->append_pt(input_pt_vec[i]);
            }
        }
    }

    template <typename T>
    void set_resolution(T resolution)
    {
        resolution_ = DATA_TYPE(resolution * 0.5);
        octree.setResolution(resolution_);
    };

    DATA_TYPE get_resolution()
    {
        return resolution_ * 2;
    }

    PC_CELL *add_cell(const PT_TYPE &cell_center)
    {
        std::unique_lock<std::mutex> lock(*m_mutex_addcell);
        MAP_PT_CELL_IT it = map_pt_cell.find(cell_center);
        if (it != map_pt_cell.end())
            return it->second;

        PC_CELL *cell = new PC_CELL(cell_center, (DATA_TYPE)resolution_);
        map_pt_cell.insert(std::make_pair(cell_center, cell));
        if (m_initialized == false)
            m_cells_center->push_back(pcl::PointXYZ(cell->center(0), cell->center(1), cell->center(2)));
        else
        {
            std::unique_lock<std::mutex> lock(*octotree_mutex);
            octree.addPointToCloud(pcl::PointXYZ(cell->center(0), cell->center(1), cell->center(2)), m_cells_center);
        }
        pc_cell_vec.push_back(cell);
        return cell;
    }

    PC_CELL *find_cell(const PT_TYPE &pt, int if_add = 1)
    {
        PT_TYPE cell_center = find_cell_center(pt);
        MAP_PT_CELL_IT it = map_pt_cell.find(cell_center);
        if (it == map_pt_cell.end())
        {
            if (if_add)
            {
                auto cell_ptr =  add_cell(cell_center);
                return cell_ptr;
            }
            else
                return nullptr;
        }
        else
            return it->second;
    }

    PT_TYPE get_center()
    {
        PT_TYPE cell_center;
        cell_center.setZero();
        for (size_t i = 0 ; i < pc_cell_vec.size(); i++)
            cell_center += pc_cell_vec[i]->center;
        cell_center *= 1.0 / (float)pc_cell_vec.size();
        return cell_center;
    }

    float distributions_of_cell(PT_TYPE & cell_center = PT_TYPE(0,0,0),  float ratio =  0.8, std::vector<PT_TYPE> * err_vec = nullptr)
    {
        cell_center = get_center();
        std::set<float> dis_vec;
        for (size_t i = 0; i < pc_cell_vec.size(); i++)
        {
            auto err = pc_cell_vec[i]->center - cell_center;
            if (err_vec != nullptr)
                err_vec->push_back(err);
            dis_vec.insert((float)err.norm());
        }
        // https://stackoverflow.com/questions/1033089/can-i-increment-an-iterator-by-just-adding-a-number
        return *std::next(dis_vec.begin(), (int)(dis_vec.size() * ratio));
    }

    template <typename T>
    std::vector<PC_CELL *> find_cells_in_radius(T pt, float searchRadius = 0)
    {
        std::unique_lock<std::mutex> lock(*octotree_mutex);
        std::vector<PC_CELL *> cells_vec;
        pcl::PointXYZ searchPoint = PCL_TOOLS::eigen_to_pcl_pt<pcl::PointXYZ>(pt);
        std::vector<int> cloudNWRSearch;
        std::vector<float> cloudNWRRadius;
        // execute octree radius search
        if (searchRadius == 0)
            octree.radiusSearch(searchPoint, resolution_, cloudNWRSearch, cloudNWRRadius);
        else
            octree.radiusSearch(searchPoint, searchRadius, cloudNWRSearch, cloudNWRRadius);

        PT_TYPE eigen_pt;
        for (size_t i = 0; i < cloudNWRSearch.size(); i++)
        {
            eigen_pt = PCL_TOOLS::pcl_pt_to_eigen<DATA_TYPE>(octree.getInputCloud()->points[cloudNWRSearch[i]]);
            cells_vec.push_back(find_cell(eigen_pt));
        }
        return cells_vec;
    }

    std::string to_json_string(int &avail_cell_size = 0)
    {
        std::string str;
        str.reserve(map_pt_cell.size() * 1e4);
        std::stringstream str_s(str);
        str_s << "[";
        avail_cell_size = 0;
        for (MAP_PT_CELL_IT it = map_pt_cell.begin(); it != map_pt_cell.end();)
        {
            PointCloudCell<DATA_TYPE> *cell = it->second;

            if (avail_cell_size != 0)
                str_s << ",";
            str_s << cell->to_json_string();
            avail_cell_size++;

            it++;
            if (it == map_pt_cell.end())
                break;
        }
        str_s << "]";
        return str_s.str();
    }

    void save_to_file(const std::string &path = std::string("./"), const std::string &file_name = std::string(""))
    {
        std::stringstream str_ss;
        COMMON_TOOLS::create_dir(path);
        if (file_name.compare("") == 0)
            str_ss << path << "/" << std::setprecision(3) << "mapping.json";
        else
            str_ss << path << "/" << file_name.c_str();

        std::fstream ofs;
        ofs.open(str_ss.str().c_str(), std::ios_base::out);
        //std::cout << "Save to " << str_ss.str();
        if (ofs.is_open())
        {
            int avail_cell_size = 0;
            ofs << to_json_string(avail_cell_size);
            ofs.close();
            //std::cout << " Successful. Number of cell = " << avail_cell_size << std::endl;
        }
        else
            std::cout << " Fail !!!" << std::endl;
    }

    template <typename T>
    T *get_json_array(const rapidjson::Document::Array &json_array)
    {
        T *res_mat = new T[json_array.Size()];
        for (size_t i = 0; i < json_array.Size(); i++)
            res_mat[i] = (T) json_array[i].GetDouble();
        return res_mat;
    }

    int load_mapping_from_file(const std::string &file_name = std::string("./mapping.json"))
    {
        COMMON_TOOLS::Timer timer;
        timer.tic("Load mapping from json file");
        FILE *fp = fopen(file_name.c_str(), "r");
        if (fp == nullptr)
        {
            std::cout << "load_mapping_from_file: " << file_name << " fail!" << std::endl;
            return 0;
        }
        else
        {
            json_file_name = file_name;
            char readBuffer[1 << 16];
            rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));

            rapidjson::Document doc;
            doc.ParseStream(is);
            if (doc.HasParseError())
            {
                printf("GetParseError, err_code =  %d\n", doc.GetParseError());
                return 0;
            }

            DATA_TYPE *pt_vec_data;
            size_t pt_num;
            for (unsigned int i = 0; i < doc.Size(); ++i)
            {
                if (i == 0)
                    set_resolution(doc[i]["Res"].GetDouble() * 2.0);
                PC_CELL *cell = add_cell(Eigen::Matrix<DATA_TYPE, 3, 1>(get_json_array<DATA_TYPE>(doc[i]["Center"].GetArray())));

                cell->m_mean = Eigen::Matrix<COMP_TYPE, 3, 1>(get_json_array<COMP_TYPE>(doc[i]["Mean"].GetArray()));
                cell->m_cov_mat = Eigen::Matrix<COMP_TYPE, 3, 3>(get_json_array<COMP_TYPE>(doc[i]["Cov"].GetArray()));
                cell->m_icov_mat = Eigen::Matrix<COMP_TYPE, 3, 3>(get_json_array<COMP_TYPE>(doc[i]["Icov"].GetArray()));
                cell->m_eigen_vec = Eigen::Matrix<COMP_TYPE, 3, 3>(get_json_array<COMP_TYPE>(doc[i]["Eig_vec"].GetArray()));
                cell->m_eigen_val = Eigen::Matrix<COMP_TYPE, 3, 1>(get_json_array<COMP_TYPE>(doc[i]["Eig_val"].GetArray()));

                pt_num = doc[i]["Pt_num"].GetInt();
                cell->point_vec.resize(pt_num);
                pt_vec_data = get_json_array<DATA_TYPE>(doc[i]["Pt_vec"].GetArray());
                for (size_t pt_idx = 0; pt_idx < pt_num; pt_idx++)
                {
                    cell->point_vec[pt_idx] << pt_vec_data[pt_idx * 3 + 0], pt_vec_data[pt_idx * 3 + 1], pt_vec_data[pt_idx * 3 + 2];
                    cell->xyz_sum = cell->xyz_sum + cell->point_vec[pt_idx].template cast<COMP_TYPE>();
                }
                delete pt_vec_data;
            }
            fclose(fp);

            std::cout << timer.toc_string("Load mapping from json file") << std::endl;
            return map_pt_cell.size();
        }
    }

    std::vector<Eigen::Matrix<DATA_TYPE, 3, 1>> load_pts_from_file(const std::string &file_name = std::string("./mapping.json"))
    {
        COMMON_TOOLS::Timer timer;
        timer.tic("Load points from json file");
        FILE *fp = fopen(file_name.c_str(), "r");
        std::vector<Eigen::Matrix<DATA_TYPE, 3, 1>> res_vec;
        if (fp == nullptr)
            return res_vec;
        else
        {
            json_file_name = file_name;
            char readBuffer[1 << 16];
            rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));

            rapidjson::Document doc;
            doc.ParseStream(is);
            if (doc.HasParseError())
            {
                printf("GetParseError, error code = %d\n", doc.GetParseError());
                return res_vec;
            }

            DATA_TYPE *pt_vec_data;
            size_t pt_num;

            for (unsigned int i = 0; i < doc.Size(); ++i)
            {
                std::vector<Eigen::Matrix<DATA_TYPE, 3, 1>> pt_vec_cell;
                pt_num = doc[i]["Pt_num"].GetInt();
                pt_vec_cell.resize(pt_num);
                pt_vec_data = get_json_array<DATA_TYPE>(doc[i]["Pt_vec"].GetArray());
                for (size_t pt_idx = 0; pt_idx < pt_num; pt_idx++)
                    pt_vec_cell[pt_idx] << pt_vec_data[pt_idx * 3 + 0], pt_vec_data[pt_idx * 3 + 1], pt_vec_data[pt_idx * 3 + 2];
                res_vec.insert(res_vec.end(), pt_vec_cell.begin(), pt_vec_cell.end());
            }
            fclose(fp);
            std::cout << timer.toc_string("Load points from json file") << std::endl;
        }
        return res_vec;
    }

    template <typename T>
    void eigen_decompose_of_featurevector(std::vector<Eigen::Matrix<T, 3, 1>> &feature_vectors,
                                          Eigen::Matrix< T, 3, 3 > &eigen_vector,
                                          Eigen::Matrix<T, 3, 1> &eigen_val)
    {
        Eigen::Matrix<double, 3, 3> mat_cov;
        mat_cov.setZero();
        for (size_t i = 0; i < feature_vectors.size(); i++)
            mat_cov = mat_cov + (feature_vectors[i] * feature_vectors[i].transpose()).template cast<double>();

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 3, 3>> eigensolver;
        eigensolver.compute(mat_cov);
        eigen_val = eigensolver.eigenvalues().template cast<T>();
        eigen_vector = eigensolver.eigenvectors().template cast<T>();
    }

    template <typename T>
    void feature_direction(Eigen::Matrix<T, 3, 1> & vec_3d, int & theta_idx, int & beta_idx)
    {
        int theta_res = THETA_RES;
        int beta_res = BETA_RES;
        double theta_step = 360.0 / theta_res;
        double beta_step = 180.0 / beta_res;
        double theta = atan2(vec_3d[1], vec_3d[0]) * 57.3 + 180.0;
        double vec_norm = sqrt(vec_3d[1] * vec_3d[1] + vec_3d[0] * vec_3d[0]);
        double beta = atan2(vec_3d[2], vec_norm) * 57.3 + 90.0;
        theta_idx = ((int) (std::floor(theta / theta_step))) % theta_res;
        if (theta_idx < 0)
            theta_idx += theta_res;
        beta_idx = (std::floor(beta / beta_step));

        #if ENABLE_DEBUG
            cout << vec_3d.transpose() << endl;
            printf("Theta = %.2f, beta = %.2f, idx = [%d, %d], res = %d \r\n" , theta, beta, theta_idx, beta_idx);
        #endif
    }

    template <typename T>
    static void refine_feature_img(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &feature_img)
    {
        int rows = feature_img.rows();
        int cols = feature_img.cols();

        if (feature_img.row(0).maxCoeff() < feature_img.row(rows - 1).maxCoeff())
            feature_img = feature_img.colwise().reverse().eval();

        if ((feature_img.block(0, 0, 2, round(cols / 2))).maxCoeff() < (feature_img.block(0, round(cols / 2), 2, round(cols / 2))).maxCoeff())
            feature_img = feature_img.rowwise().reverse().eval();
    }

    static float similiarity_of_two_image(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_a,
                                          const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_b)
    {
        assert(((img_a.rows() == img_b.rows()) && (img_a.cols() == img_b.cols())));

        auto  img_sub_mea_a = img_a.array() - img_a.mean();
        auto  img_sub_mea_b = img_b.array() - img_b.mean();

        float product = ((img_sub_mea_a).cwiseProduct(img_sub_mea_b)).mean();
        int devide_size = img_a.rows() * img_a.cols() - 1;
        float std_a = (img_sub_mea_a.array().pow(2)).sum() / devide_size ;
        float std_b = (img_sub_mea_b.array().pow(2)).sum() / devide_size;
        return sqrt(product * product / std_a / std_b);
    };

    static float ratio_of_nonzero_in_img(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img)
    {
        int count = 0;
        for (int i = 0; i < img.rows(); i++)
            for (int j = 0; j < img.cols(); j++)
                if (img(i, j) >= 1.0)
                    count++;

        return (float)(count) / (img.rows() * img.cols());
    }

    std::vector<PT_TYPE> query_point_cloud(std::vector< PC_CELL *> & cell_vec)
    {
        std::vector<std::vector<PT_TYPE>> pt_vec_vec;
        pt_vec_vec.reserve(1000);
        for (int i = 0; i < cell_vec.size(); i++)
        {
            pt_vec_vec.push_back(cell_vec[i]->get_pointcloud_eigen());
        }
        return COMMON_TOOLS::vector_2d_to_1d(pt_vec_vec);
    }

    static float max_similiarity_of_two_image(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_a,
                                              const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_b,
                                              float minimum_zero_ratio = 0.00)
    {
        if (ratio_of_nonzero_in_img(img_a) < minimum_zero_ratio)
            return 0;

        if (ratio_of_nonzero_in_img(img_b) < minimum_zero_ratio)
            return 0;

        size_t cols = img_a.cols();
        size_t rows = img_a.rows();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img_a_roi = img_a.block(0, 0, (int)std::round(rows / 2), cols);
        float max_res = -3e8;

        cv::Mat hist_a, hist_b;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img_b_roi;
        img_b_roi.resize(rows, cols);

        for (size_t i = 0; i < rows; i++)
        {
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img_b_roi_up = img_b.block(i, 0, rows - i, cols);
            Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img_b_roi_down = img_b.block(0, 0, i, cols);

            img_b_roi << img_b_roi_up , img_b_roi_down;
            float res = similiarity_of_two_image_cv(img_a, img_b_roi);

            if (res > max_res)
                max_res = res;
        }
        return max_res;
    }



    static float similiarity_of_two_image_cv(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_a,
                                             const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_b,
                                             int method = CV_COMP_CORREL)
    {
        cv::Mat hist_a, hist_b;
        cv::eigen2cv(img_a, hist_a);
        cv::eigen2cv(img_b, hist_b);
        return cv::compareHist(hist_a, hist_b, method); // https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=comparehist
    }

    static float max_similiarity_of_two_image_cv(const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_a,
                                                 const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &img_b,
                                                 int method = CV_COMP_CORREL)
    {
        cv::Mat hist_a, hist_b;
        int cols = img_a.cols();
        int rows = img_a.rows();
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> img_a_roi = img_a.block(0, 0, (int)std::round(rows / 2), cols);
        cv::eigen2cv(img_a_roi, hist_a);
        cv::eigen2cv(img_b, hist_b);
        cv::Mat result;
        cv::matchTemplate(hist_b, hist_a, result, CV_TM_CCORR_NORMED);
        double minVal;
        double maxVal;
        cv::Point minLoc;
        cv::Point maxLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
        return maxVal;
    }

    int extract_feature_mapping(std::vector<PC_CELL *> cell_vec,
                                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &feature_img_line,
                                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &feature_img_plane,
                                int if_recompute = 0)
    {
        MAP_PT_CELL_IT it;
        std::vector<Eigen::Matrix<DATA_TYPE, 3, 1>> feature_vecs_plane, feature_vecs_line;
        for (size_t i = 0; i < cell_vec.size(); i++)
        {
            PC_CELL *cell = cell_vec[i];
            auto feature_type = cell->determine_feature(if_recompute);
            if (feature_type == FeatureType::e_feature_line)
            {
                feature_vecs_line.push_back(cell->feature_vector.template cast<DATA_TYPE>());
                feature_vecs_line.push_back(-1.0 * cell->feature_vector.template cast<DATA_TYPE>());
            }
            else if (feature_type == FeatureType::e_feature_plane)
            {
                feature_vecs_plane.push_back(cell->feature_vector.template cast<DATA_TYPE>());
                feature_vecs_plane.push_back(-1.0 * cell->feature_vector.template cast<DATA_TYPE>());
            }
        }

        Eigen::Matrix<DATA_TYPE, 3, 3> eigen_vector;
        Eigen::Matrix<DATA_TYPE, 3, 1> eigen_val;

        eigen_decompose_of_featurevector(feature_vecs_plane, eigen_vector, eigen_val);
        eigen_vector = eigen_vector.rowwise().reverse().eval();
        eigen_val = eigen_val.colwise().reverse().eval();
        eigen_vector.col(2) = eigen_vector.col(0).cross(eigen_vector.col(1));

        feature_img_line.resize(THETA_RES, BETA_RES);
        feature_img_plane.resize(THETA_RES, BETA_RES);

        feature_img_line.setZero();
        feature_img_plane.setZero();

        int theta_idx = 0;
        int beta_idx = 0;
        Eigen::Matrix<DATA_TYPE, 3, 1> affined_vector;

        for (size_t i = 0; i < feature_vecs_plane.size(); i++)
        {
            affined_vector = eigen_vector.transpose() * feature_vecs_plane[i];
            feature_direction(affined_vector, theta_idx, beta_idx);
            feature_img_plane(theta_idx, beta_idx) += 1;

            affined_vector = affined_vector * -1.0;
            feature_direction((affined_vector), theta_idx, beta_idx);
            feature_img_plane(theta_idx, beta_idx) += 1;
        }

        for (size_t i = 0; i < feature_vecs_line.size(); i++)
        {
            affined_vector = eigen_vector.transpose() * feature_vecs_line[i];
            feature_direction(affined_vector, theta_idx, beta_idx);
            feature_img_line(theta_idx, beta_idx) += 1;

            affined_vector = affined_vector * -1.0;
            feature_direction((affined_vector), theta_idx, beta_idx);
            feature_img_line(theta_idx, beta_idx) += 1;
        }

        refine_feature_img(feature_img_plane);
        refine_feature_img(feature_img_line);

        cv::Mat feature_img_plane_cv, feature_img_line_cv;
        cv::eigen2cv(feature_img_line, feature_img_line_cv);
        cv::eigen2cv(feature_img_plane, feature_img_plane_cv);
        cv::Size kernel_size = cv::Size(3, 3);
        m_ratio_nonzero_line = ratio_of_nonzero_in_img(feature_img_line);
        m_ratio_nonzero_plane = ratio_of_nonzero_in_img(feature_img_plane);
        float sigma = 0.5;
        cv::GaussianBlur(feature_img_plane_cv, feature_img_plane_cv, kernel_size, sigma);
        cv::GaussianBlur(feature_img_line_cv, feature_img_line_cv, kernel_size, sigma);
        cv::cv2eigen(feature_img_plane_cv, feature_img_plane);
        cv::cv2eigen(feature_img_line_cv, feature_img_line);

        return 0;
    };

    int analyze_mapping(int if_recompute = 0)
    {
        float ratio = 0.90;
        Eigen_Point cell_center;
        std::vector<PC_CELL *> cell_vec;
        
        roi_range = distributions_of_cell(cell_center, ratio);
        cell_vec = find_cells_in_radius(get_center(), roi_range);
        extract_feature_mapping(cell_vec, feature_img_line_roi, feature_img_plane_roi, if_recompute);
        extract_feature_mapping(pc_cell_vec, feature_img_line, feature_img_plane, if_recompute);
        return 0;
    }
    
    pcl::PointCloud<pcl_pt> extract_specify_points(FeatureType select_type)
    {
        pcl::PointCloud<pcl_pt> res_pt;
        for (size_t i = 0; i < pc_cell_vec.size(); i++)
            if (pc_cell_vec[i]->feature_type == select_type)
                res_pt += pc_cell_vec[i]->get_pointcloud();
        return res_pt;
    }

    pcl::PointCloud<pcl_pt> get_all_pointcloud()
    {
        pcl::PointCloud<pcl_pt> res_pt;
        for (size_t i = 0; i < pc_cell_vec.size(); i++)
            res_pt += pc_cell_vec[i]->get_pointcloud();
        return res_pt;
    }
};
