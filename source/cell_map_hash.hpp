#pragma once
#define USE_HASH 1

#include "common_tools.h"
#include "pcl_tools.hpp"
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
//#include <pcl/filters/voxel_grid_covariance.h>
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/prettywriter.h"

//#include "voxel_grid_covariance.h"
//#include <pcl/octree/octree_pointcloud_changedetector.h>
#include <pcl/filters/voxel_grid_covariance.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>

#include <Eigen/Eigen>
#include <boost/format.hpp>
#include <iomanip>
#include <map>
#include <math.h>
#include <mutex>
#include <thread>
#include <vector>
#if USE_HASH
#include <unordered_map>
#else
#endif

#define IF_COV_INIT_IDENTITY 0
#define IF_EIGEN_REPLACE 1
#define IF_ENABLE_INCREMENTAL_UPDATE_MEAN_COV 1
typedef double         COMP_TYPE;
typedef pcl::PointXYZI pcl_pt;

enum Feature_type
{
    e_feature_sphere = 0,
    e_feature_line = 1,
    e_feature_plane = 2
};

template <typename DATA_TYPE>
class Points_cloud_cell
{
  public:
    typedef Eigen::Matrix< DATA_TYPE, 3, 1 > Eigen_Point;

    typedef Eigen::Matrix< DATA_TYPE, 3, 1 > PT_TYPE;
    DATA_TYPE                                m_resolution;
    Eigen::Matrix< DATA_TYPE, 3, 1 >         m_center;

    //private:
    Eigen::Matrix< COMP_TYPE, 3, 1 > m_xyz_sum;
    Eigen::Matrix< COMP_TYPE, 3, 1 > m_mean;
    Eigen::Matrix< COMP_TYPE, 3, 3 > m_cov_mat;
    Eigen::Matrix< COMP_TYPE, 3, 3 > m_icov_mat;

    /** \brief Eigen vectors of voxel covariance matrix */
    Eigen::Matrix< COMP_TYPE, 3, 3 > m_eigen_vec; // Eigen vector of covariance matrix
    Eigen::Matrix< COMP_TYPE, 3, 1 > m_eigen_val; // Eigen value of covariance values

    Feature_type                     m_feature_type = e_feature_sphere;
    double                           m_feature_determine_threshold_line = 1.0 / 3.0;
    double                           m_feature_determine_threshold_plane = 1.0 / 3.0;
    Eigen::Matrix< COMP_TYPE, 3, 1 > m_feature_vector;

  public:
    pcl::VoxelGridCovariance<pcl_pt> m_pcl_voxel_cell;
    std::vector< PT_TYPE >             m_points_vec;
    pcl::PointCloud<pcl_pt> m_pcl_pc_vec;
    DATA_TYPE                          m_cov_det_sqrt;
    int                                m_if_compute_using_pcl = false;
    bool                               m_mean_need_update = true;
    bool                               m_covmat_need_update = true;
    bool                               m_icovmat_need_update = true;
    bool                               m_pcl_voxelgrid_need_update = true;
    size_t                             m_maximum_points_size = (size_t) 1e4;
    int                                m_if_incremental_update_mean_and_cov = 0;
    std::mutex *                       mutex_cell;
    Points_cloud_cell()
    {
        mutex_cell = new std::mutex();
        clear_data();
   };

    ~Points_cloud_cell()
    {
        mutex_cell->try_lock();
        mutex_cell->unlock();
        clear_data();
   };
    ADD_SCREEN_PRINTF_OUT_METHOD;
    template < typename T, typename TT >
    void save_mat_to_jason_writter(T &writer, const std::string &name, const TT &eigen_mat)
    {
        writer.Key(name.c_str()); // output a key,
        writer.StartArray();        // Between StartArray()/EndArray(),
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
#if 0
        rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(sb);
#else
        //rapidjson::Writer<rapidjson::StringBuffer> writer(sb);
        rapidjson::Writer< rapidjson::StringBuffer > writer(sb);
#endif
        writer.StartObject();               // Between StartObject()/EndObject(),
        writer.SetMaxDecimalPlaces(1000); // like set_precision

        writer.Key("Pt_num");
        writer.Int(m_points_vec.size());
        writer.Key("Res"); // output a key
        writer.Double(m_resolution);
        save_mat_to_jason_writter(writer, "Center", m_center);
        save_mat_to_jason_writter(writer, "Mean", m_mean);
        if (m_points_vec.size() > 5)
        {
            save_mat_to_jason_writter(writer, "Cov", m_cov_mat);
            save_mat_to_jason_writter(writer, "Icov", m_icov_mat);
            save_mat_to_jason_writter(writer, "Eig_vec", m_eigen_vec);
            save_mat_to_jason_writter(writer, "Eig_val", m_eigen_val);
       }
        else
        {
            Eigen::Matrix< COMP_TYPE, 3, 3 > I;
            Eigen::Matrix< COMP_TYPE, 3, 1 > Vec3d;
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
        for (unsigned i = 0; i < m_points_vec.size(); i++)
        {
            writer.Double(m_points_vec[i](0));
            writer.Double(m_points_vec[i](1));
            writer.Double(m_points_vec[i](2));
       }
        writer.EndArray();
        writer.SetMaxDecimalPlaces(1000);

        writer.EndObject();

        //document.Accept(writer);    // Accept() traverses the DOM and generates Handler events.
        //puts(sb.GetString());
        return std::string(sb.GetString());
   }

    void save_to_file(const std::string &path = std::string("./"), const std::string &file_name = std::string(""))
    {
        std::stringstream str_ss;
        Common_tools::create_dir(path);
        if (file_name.compare("") == 0)
        {
            str_ss << path << "/" << std::setprecision(3)
                   << m_center(0) << "_"
                   << m_center(1) << "_"
                   << m_center(2) << ".json";
       }
        else
        {
            str_ss << path << "/" << file_name.c_str();
       }
        std::fstream ofs;
        ofs.open(str_ss.str().c_str(), std::ios_base::out);
        std::cout << "Save to " << str_ss.str();
        if (ofs.is_open())
        {
            ofs << to_json_string();
            ofs.close();
            std::cout << " Successful. Number of points = " << m_points_vec.size() << std::endl;
       }
        else
        {
            std::cout << " Fail !!!" << std::endl;
       }
   }

    void pcl_voxelgrid_update()
    {
        if (m_pcl_voxelgrid_need_update)
        {
            m_pcl_voxel_cell.setLeafSize(200.0, 200.0, 200.0);
            m_pcl_voxel_cell.setInputCloud(m_pcl_pc_vec.makeShared());
            m_pcl_voxel_cell.filter(true);
            //assert(m_pcl_voxel_cell.getLeafSize() == 1);
       }
        m_pcl_voxelgrid_need_update = false;
   }

    void set_data_need_update(int if_update_sum = 0)
    {
        m_mean_need_update = true;
        m_covmat_need_update = true;
        m_icovmat_need_update = true;
        m_pcl_voxelgrid_need_update = true;
        if (if_update_sum)
        {
            m_xyz_sum.setZero();
            for (size_t i = 0; i < m_points_vec.size(); i++)
            {
                m_xyz_sum += m_points_vec[i].template cast< COMP_TYPE >();
           }
       }
   }

    int get_points_count()
    {
        return m_points_vec.size();
   }

    Eigen::Matrix< DATA_TYPE, 3, 1 > get_center()
    {
        return m_center.template cast<DATA_TYPE>();
   }

    Eigen::Matrix< DATA_TYPE, 3, 1 > get_mean()
    {
        if (m_if_incremental_update_mean_and_cov == false)
        {
            if (m_mean_need_update)
            {
                if (m_if_compute_using_pcl)
                {
                    
                    pcl_voxelgrid_update();
                    
                    m_mean_need_update = false;
                    auto leaf = m_pcl_voxel_cell.getLeaf(m_center);
                    assert(leaf != nullptr);
                    return leaf->getMean().template cast<DATA_TYPE>();
               }
                set_data_need_update();
                m_mean = m_xyz_sum / ((DATA_TYPE)(m_points_vec.size()));
           }
            m_mean_need_update = false;
       }
        return m_mean.template cast<DATA_TYPE>();
   }

    Eigen::Matrix< DATA_TYPE, 3, 3 > robust_covmat()
    {
        Eigen::SelfAdjointEigenSolver< Eigen::Matrix< COMP_TYPE, 3, 3 > > eigensolver;
        Eigen::Matrix< COMP_TYPE, 3, 3 >                                  eigen_val;

        COMP_TYPE min_covar_eigvalue;
        COMP_TYPE min_covar_eigvalue_mult_ = 0.01; // pcl: 0.01
        if (!IF_EIGEN_REPLACE)
            min_covar_eigvalue_mult_ = 0;

        // Avoids matrices near singularities (eq 6.11)[Magnusson 2009]
        if (1)
        {
            eigensolver.compute(m_cov_mat);
            eigen_val = eigensolver.eigenvalues().asDiagonal();
            m_eigen_val = eigensolver.eigenvalues();
            m_eigen_vec = eigensolver.eigenvectors();
            min_covar_eigvalue = min_covar_eigvalue_mult_ * eigen_val(2, 2);
            if (eigen_val(0, 0) < min_covar_eigvalue)
            {
                eigen_val(0, 0) = min_covar_eigvalue;

                if (eigen_val(1, 1) < min_covar_eigvalue)
                {
                    eigen_val(1, 1) = min_covar_eigvalue;
               }

                m_cov_mat = m_eigen_vec * eigen_val * m_eigen_vec.inverse();
                if (!std::isfinite(m_cov_mat(0, 0)))
                {
                    m_cov_mat.setIdentity();
               }
           }
       }
        return m_cov_mat.template cast<DATA_TYPE>();
   }

    Eigen::Matrix< DATA_TYPE, 3, 3 > get_covmat()
    {
        if (m_covmat_need_update)
        {
            get_mean();
            size_t pt_size = m_points_vec.size();
            if (pt_size < 5 || m_if_incremental_update_mean_and_cov == false)
            {
                if (m_if_compute_using_pcl)
                {
                    
                    pcl_voxelgrid_update();
                    
                    auto leaf = m_pcl_voxel_cell.getLeaf(m_center);
                    assert(leaf != nullptr);
                    return leaf->getCov().template cast<DATA_TYPE>();
               }
                if (IF_COV_INIT_IDENTITY)
                {
                    m_cov_mat.setIdentity();
               }
                else
                {
                    m_cov_mat.setZero();
               }

                if (pt_size <= 2)
                {
                    return m_cov_mat.template cast<DATA_TYPE>();
               }

                for (size_t i = 0; i < pt_size; i++)
                {
                    //Eigen::Matrix< COMP_TYPE, 3, 1 > y_minus_mu;
                    //y_minus_mu = m_points_vec[i].template cast< COMP_TYPE >() - m_mean;
                    //m_cov_mat = (m_cov_mat  + (y_minus_mu * y_minus_mu.transpose())) ;
                    m_cov_mat = m_cov_mat + (m_points_vec[i] * m_points_vec[i].transpose()).template cast< COMP_TYPE >();
               }
                m_cov_mat -= pt_size * (m_mean * m_mean.transpose());
                m_cov_mat /= (pt_size - 1);
           }
            robust_covmat();
       }
        m_covmat_need_update = false;
        return m_cov_mat.template cast<DATA_TYPE>();
   }

    Eigen::Matrix< DATA_TYPE, 3, 3 > get_icovmat()
    {
        if (m_icovmat_need_update)
        {
            get_covmat();
            if (m_if_compute_using_pcl)
            {
                pcl_voxelgrid_update();
                auto leaf = m_pcl_voxel_cell.getLeaf(m_center);
                assert(leaf != nullptr);
                return leaf->getInverseCov().template cast<DATA_TYPE>();
           }
            m_icov_mat = m_cov_mat.inverse();
            if (!std::isfinite(m_icov_mat(0, 0)))
            {
                m_icov_mat.setIdentity();
           }
       }
        m_icovmat_need_update = false;
        return m_icov_mat.template cast<DATA_TYPE>();
   }

    pcl::PointCloud<pcl_pt> get_pointcloud()
    {
        std::unique_lock<std::mutex> lock(*mutex_cell);
        pcl::PointCloud<pcl_pt> pt_temp = m_pcl_pc_vec;
        return pt_temp;
   }

    std::vector< PT_TYPE > get_pointcloud_eigen()
    {
        std::unique_lock<std::mutex> lock(*mutex_cell);
        return m_points_vec;
   }

    void set_pointcloud(pcl::PointCloud<pcl_pt> &pc_in)
    {
        std::unique_lock<std::mutex> lock(*mutex_cell);
        m_pcl_pc_vec = pc_in;
        m_points_vec = PCL_TOOLS::pcl_pts_to_eigen_pts<float, pcl_pt>(pc_in.makeShared());
   }

    void clear_data()
    {
        std::unique_lock<std::mutex> lock(*mutex_cell);
        m_points_vec.clear();
        m_pcl_pc_vec.clear();
        m_mean.setZero();
        m_xyz_sum.setZero();
        m_cov_mat.setZero();
   }

    Points_cloud_cell(const PT_TYPE &cell_center, const DATA_TYPE &res = 1.0)
    {
        mutex_cell = new std::mutex();
        clear_data();
        m_resolution = res;
        m_maximum_points_size = (int) res * 100.0;
        m_points_vec.reserve(m_maximum_points_size);
        if (m_if_compute_using_pcl)
        {
            m_pcl_pc_vec.reserve(m_maximum_points_size);
       }
        m_center = cell_center;
        //append_pt(cell_center);
   }

    void append_pt(const PT_TYPE &pt)
    {
        std::unique_lock<std::mutex> lock(*mutex_cell);
        if (1)
        {
            m_pcl_pc_vec.push_back(PCL_TOOLS::eigen_to_pcl_pt<pcl_pt>(pt));
       }
        m_points_vec.push_back(pt);
        if (m_points_vec.size() > m_maximum_points_size)
        {
            m_maximum_points_size *= 10;
            m_points_vec.reserve(m_maximum_points_size);
       }

        m_xyz_sum = m_xyz_sum + pt.template cast< COMP_TYPE >();

        if (m_if_incremental_update_mean_and_cov)
        {
            auto   mean_old = m_mean;
            auto   cov_old = m_cov_mat;
            auto   P_new = pt.template cast< COMP_TYPE >();
            size_t N = m_points_vec.size() - 1;
            m_mean = (N * mean_old + P_new) / (N + 1);

            if (N > 5)
            {
                m_cov_mat = ((N - 1) * cov_old + (P_new - mean_old) * ((P_new - mean_old).transpose()) +
                              (N + 1) * (mean_old - m_mean) * ((mean_old - m_mean).transpose()) +
                              2 * (mean_old - m_mean) * ((P_new - mean_old).transpose())) /
                            N;
           }
            else
            {
                get_covmat();
           }

            //if (N < 10) //For comparision
            if (0)
            {
                screen_out << "***Incre N = " << N << " ***" << std::endl;
                screen_out << "Cov old: " << cov_old << std::endl;
                screen_out << "Cov new: " << m_cov_mat << std::endl;
                screen_out << "Cov raw: " << get_covmat() << std::endl;
           }
       }

        set_data_need_update();
        //m_points_vec.insert(pt);
   }

    void set_target_pc(const std::vector< PT_TYPE > &pt_vec)
    {
        std::unique_lock<std::mutex> lock(*mutex_cell);
        // "The three-dimensional normal-distributions transform: an efficient representation for registration, surface analysis, and loop detection"
        // Formulation 6.2 and 6.3
        int pt_size = pt_vec.size();
        //assert(pt_size > 5);
        clear_data();
        m_points_vec.reserve(m_maximum_points_size);
        for (int i = 0; i < pt_size; i++)
        {
            append_pt(pt_vec[i]);
       }
        set_data_need_update();
   };

    Feature_type determine_feature(int if_recompute)
    {
        if (if_recompute)
        {
            set_data_need_update(1);
       }

        get_covmat();
        m_feature_type = e_feature_sphere;
        if (m_points_vec.size() < 10)
        {
            m_feature_type = e_feature_sphere;
            m_feature_vector << 0, 0, 0;
            return e_feature_sphere;
       }

        if ((m_center.template cast<float>() - m_mean.template cast<float>()).norm() > m_resolution * 0.75)
        {
            m_feature_type = e_feature_sphere;
            m_feature_vector << 0, 0, 0;
            return e_feature_sphere;
       }

        if ((m_eigen_val[1] * m_feature_determine_threshold_plane > m_eigen_val[0]))
        {
            m_feature_type = e_feature_plane;
            //m_feature_vector = m_eigen_vec.block< 1, 3 >(0, 0).transpose();
            m_feature_vector = m_eigen_vec.block< 3, 1 >(0, 0);
            return m_feature_type;
       }
        if (m_eigen_val[2] * m_feature_determine_threshold_line > m_eigen_val[1])
        {
            m_feature_type = e_feature_line;
            //m_feature_vector = m_eigen_vec.block< 1, 3 >(2, 0).transpose();
            m_feature_vector = m_eigen_vec.block< 3, 1 >(0, 2);
       }
        return m_feature_type;
   }
};

template <typename DATA_TYPE>
class Points_cloud_map
{
  public:
    typedef Eigen::Matrix< DATA_TYPE, 3, 1 > PT_TYPE;
    typedef Eigen::Matrix< DATA_TYPE, 3, 1 > Eigen_Point;
    DATA_TYPE                                m_x_min, m_x_max;
    DATA_TYPE                                m_y_min, m_y_max;
    DATA_TYPE                                m_z_min, m_z_max;
    DATA_TYPE                                m_resolution; // resolution mean the distance of a cute to its bound.
    Common_tools::Timer                      m_timer;
    typedef Points_cloud_cell<DATA_TYPE>   Mapping_cell;
    std::vector<Mapping_cell *> m_cell_vec;
    int                                      scale = 10;
    int                                      THETA_RES = (int) (12 * scale);
    int                                      BETA_RES = (int) (6 * scale);
    int                                      m_if_incremental_update_mean_and_cov = IF_ENABLE_INCREMENTAL_UPDATE_MEAN_COV;
    std::mutex *                             m_mapping_mutex;
    std::mutex *                             m_octotree_mutex;
    std::mutex *                             mutex_addcell;
    std::string                              m_json_file_name;

    float m_ratio_nonzero_line, m_ratio_nonzero_plane;
#if USE_HASH
    typedef std::unordered_map< PT_TYPE, Mapping_cell *, PCL_TOOLS::Eigen_pt_hasher, PCL_TOOLS::Eigen_pt_compare >                    Map_pt_cell;
    typedef typename std::unordered_map< PT_TYPE, Mapping_cell *, PCL_TOOLS::Eigen_pt_hasher, PCL_TOOLS::Eigen_pt_compare >::iterator Map_pt_cell_it;
#else
    typedef std::map< PT_TYPE, Mapping_cell *, PCL_TOOLS::Pt_compare > Map_pt_cell;
    typedef typename std::map< PT_TYPE, Mapping_cell *, PCL_TOOLS::Pt_compare >::iterator Map_pt_cell_it;
#endif

    Map_pt_cell    m_map_pt_cell; // using hash_map
    Map_pt_cell_it m_map_pt_cell_it;

    Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > m_feature_img_line, m_feature_img_plane;
    Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > m_feature_img_line_roi, m_feature_img_plane_roi;
    Eigen::Matrix< float, 3, 3 >                           m_eigen_R, m_eigen_R_roi;
    float                                                  m_roi_range;

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> m_octree = pcl::octree::OctreePointCloudSearch<pcl::PointXYZ>(0.0001);
    pcl::PointCloud<pcl::PointXYZ>::Ptr                m_cells_center = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
    int                                                  m_initialized = false;
    Points_cloud_map()
    {
        m_mapping_mutex = new std::mutex();
        mutex_addcell = new std::mutex();
        m_octotree_mutex = new std::mutex();
        m_x_min = std::numeric_limits<DATA_TYPE>::max();
        m_y_min = std::numeric_limits<DATA_TYPE>::max();
        m_z_min = std::numeric_limits<DATA_TYPE>::max();

        m_x_max = std::numeric_limits<DATA_TYPE>::min();
        m_y_max = std::numeric_limits<DATA_TYPE>::min();
        m_z_max = std::numeric_limits<DATA_TYPE>::min();
        m_cells_center->reserve(1e5);
        set_resolution(1.0);
        m_if_verbose_screen_printf = 1;
   };
    ~Points_cloud_map()
    {
        m_mapping_mutex->try_lock();
        m_mapping_mutex->unlock();

        mutex_addcell->try_lock();
        mutex_addcell->unlock();

        m_octotree_mutex->try_lock();
        m_octotree_mutex->unlock();
   }

    ADD_SCREEN_PRINTF_OUT_METHOD;

    int get_cells_size()
    {
        return m_map_pt_cell.size();
   }

    PT_TYPE find_cell_center(const PT_TYPE &pt)
    {
        PT_TYPE   cell_center;
        DATA_TYPE GRID_SIZE = m_resolution * 1.0;
        DATA_TYPE HALF_GRID_SIZE = m_resolution * 0.5;

        // cell_center(0) = ((int) ((pt(0) - m_x_min - m_resolution*0.5) / m_resolution)) * m_resolution + m_x_min + m_resolution*0.5;
        // cell_center(1) = ((int) ((pt(1) - m_y_min - m_resolution*0.5) / m_resolution)) * m_resolution + m_y_min + m_resolution*0.5;
        // cell_center(2) = ((int) ((pt(2) - m_z_min - m_resolution*0.5) / m_resolution)) * m_resolution + m_z_min + m_resolution*0.5;

        m_x_min = 0;
        m_y_min = 0;
        m_z_min = 0;

        cell_center(0) = (std::round((pt(0) - m_x_min - HALF_GRID_SIZE) / GRID_SIZE)) * GRID_SIZE + m_x_min + HALF_GRID_SIZE;
        cell_center(1) = (std::round((pt(1) - m_y_min - HALF_GRID_SIZE) / GRID_SIZE)) * GRID_SIZE + m_y_min + HALF_GRID_SIZE;
        cell_center(2) = (std::round((pt(2) - m_z_min - HALF_GRID_SIZE) / GRID_SIZE)) * GRID_SIZE + m_z_min + HALF_GRID_SIZE;

        // cell_center(0) = (std::floor((pt(0) - m_x_min) / m_resolution)) * m_resolution + m_x_min + m_resolution*0.5;
        // cell_center(1) = (std::floor((pt(1) - m_y_min) / m_resolution)) * m_resolution + m_y_min + m_resolution*0.5;
        // cell_center(2) = (std::floor((pt(2) - m_z_min) / m_resolution)) * m_resolution + m_z_min + m_resolution*0.5;
        //std::cout << "In pt = " << pt.transpose() << " , its correspond cell center: " << cell_center.transpose() << std::endl;
        return cell_center;
   }

    void clear_data()
    {
        for (Map_pt_cell_it it = m_map_pt_cell.begin(); it != m_map_pt_cell.end(); it++)
        {
            it->second->clear_data();
            delete it->second;
       }
        m_map_pt_cell.clear();
        m_cell_vec.clear();
        m_cells_center->clear();
        m_octree.deleteTree();
   }

    void set_point_cloud(const std::vector< PT_TYPE > &input_pt_vec)
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
            Mapping_cell *cell = find_cell(input_pt_vec[i]);
            cell->append_pt(input_pt_vec[i]);
       }

        m_octree.setInputCloud(m_cells_center);
        m_octree.addPointsFromInputCloud();
        std::cout << "*** set_point_cloud octree initialization finish ***" << std::endl;
        m_initialized = true;
   }

    void append_cloud(const std::vector< PT_TYPE > &input_pt_vec, int if_vervose = false)
    {
        //std::unique_lock<std::mutex> lock(m_mapping_mutex);
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
                Points_cloud_cell<DATA_TYPE> *cell = find_cell(input_pt_vec[i]);
                cell->append_pt(input_pt_vec[i]);
                //octree.addPointToCloud(pcl::PointXYZ(cell->m_center(0), cell->m_center(1), cell->m_center(2)) , m_cells_center);
           }
       }
        if (if_vervose == false)
        {
            screen_out << "Input points size: " << input_pt_vec.size() << ", "
                       << "add cell number: " << get_cells_size() - current_size << ", "
                       << "curren cell number: " << m_map_pt_cell.size() << std::endl;
            screen_out << m_timer.toc_string(__FUNCTION__) << std::endl;
       }
   }

    template <typename T>
    void set_resolution(T resolution)
    {
        //m_resolution = DATA_TYPE(resolution);
        m_resolution = DATA_TYPE(resolution * 0.5);
        m_octree.setResolution(m_resolution);
   };

    DATA_TYPE get_resolution()
    {
        return m_resolution * 2;
   }

    Mapping_cell *add_cell(const PT_TYPE &cell_center)
    {
        std::unique_lock<std::mutex> lock(*mutex_addcell);
        Map_pt_cell_it it = m_map_pt_cell.find(cell_center);

        if (it != m_map_pt_cell.end())
            return it->second;

        Mapping_cell *cell = new Mapping_cell(cell_center, (DATA_TYPE)m_resolution);
        cell->m_if_incremental_update_mean_and_cov = m_if_incremental_update_mean_and_cov;
        m_map_pt_cell.insert(std::make_pair(cell_center, cell));
        if (m_initialized == false)
            m_cells_center->push_back(pcl::PointXYZ(cell->m_center(0), cell->m_center(1), cell->m_center(2)));
        else
        {
            std::unique_lock<std::mutex> lock(*m_octotree_mutex);
            m_octree.addPointToCloud(pcl::PointXYZ(cell->m_center(0), cell->m_center(1), cell->m_center(2)), m_cells_center);
       }

        m_cell_vec.push_back(cell);
        return cell;
   }

    Mapping_cell *find_cell(const PT_TYPE &pt, int if_add = 1)
    {
        PT_TYPE cell_center = find_cell_center(pt);
        Map_pt_cell_it it = m_map_pt_cell.find(cell_center);
        if (it == m_map_pt_cell.end())
        {
            if (if_add)
            {
                auto cell_ptr = add_cell(cell_center);
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
        for (size_t i = 0; i < m_cell_vec.size(); i++)
        {
            cell_center += m_cell_vec[i]->m_center;
       }
        cell_center *= (1.0 / (float) m_cell_vec.size());
        return cell_center;
   }

    float distributions_of_cell(PT_TYPE &cell_center = PT_TYPE(0, 0, 0), float ratio = 0.8, std::vector< PT_TYPE > *err_vec = nullptr)
    {
        cell_center = get_center();
        std::set<float> dis_vec;
        for (size_t i = 0; i < m_cell_vec.size(); i++)
        {
            auto err = m_cell_vec[i]->m_center - cell_center;
            if (err_vec != nullptr)
            {
                err_vec->push_back(err);
           }
            dis_vec.insert((float) err.norm());
       }
        // https://stackoverflow.com/questions/1033089/can-i-increment-an-iterator-by-just-adding-a-number
        return *std::next(dis_vec.begin(), (int) (dis_vec.size() * ratio));
   }

    template <typename T>
    std::vector<Mapping_cell *> find_cells_in_radius(T pt, float searchRadius = 0)
    {
        std::unique_lock<std::mutex> lock(*m_octotree_mutex);
        std::vector<Mapping_cell *> cells_vec;
        pcl::PointXYZ searchPoint = PCL_TOOLS::eigen_to_pcl_pt<pcl::PointXYZ>(pt);
        std::vector<int> cloudNWRSearch;
        std::vector<float> cloudNWRRadius;
        // execute octree radius search
        if (searchRadius == 0)
            m_octree.radiusSearch(searchPoint, m_resolution, cloudNWRSearch, cloudNWRRadius);
        else
            m_octree.radiusSearch(searchPoint, searchRadius, cloudNWRSearch, cloudNWRRadius);

        PT_TYPE eigen_pt;
        for (size_t i = 0; i < cloudNWRSearch.size(); i++)
        {
            eigen_pt = PCL_TOOLS::pcl_pt_to_eigen<DATA_TYPE>(m_octree.getInputCloud()->points[cloudNWRSearch[i]]);
            cells_vec.push_back(find_cell(eigen_pt));
       }

        return cells_vec;
   }

    std::string to_json_string(int &avail_cell_size = 0)
    {
        std::string str;
        str.reserve(m_map_pt_cell.size() * 1e4);
        std::stringstream str_s(str);
        str_s << "[";
        avail_cell_size = 0;
        for (Map_pt_cell_it it = m_map_pt_cell.begin(); it != m_map_pt_cell.end();)
        {
            Points_cloud_cell<DATA_TYPE> *cell = it->second;
            //if (cell->m_points_vec.size() > 5)
            if (1)
            {
                if (avail_cell_size != 0)
                {
                    str_s << ",";
               }
                str_s << cell->to_json_string();
                avail_cell_size++;
           }
            else
            {
                //std::cout << "Points_vec = " << cell->m_points_vec.size() << endl;
                //continue;
           }

            it++;
            if (it == m_map_pt_cell.end())
            {
                break;
           }
       }
        str_s << "]";
        return str_s.str();
   }

    void save_to_file(const std::string &path = std::string("./"), const std::string &file_name = std::string(""))
    {
        ENABLE_SCREEN_PRINTF;
        std::stringstream str_ss;
        Common_tools::create_dir(path);
        if (file_name.compare("") == 0)
        {
            str_ss << path << "/" << std::setprecision(3) << "mapping.json";
       }
        else
        {
            str_ss << path << "/" << file_name.c_str();
       }
        std::fstream ofs;
        ofs.open(str_ss.str().c_str(), std::ios_base::out);
        screen_out << "Save to " << str_ss.str();
        if (ofs.is_open())
        {
            int avail_cell_size = 0;
            ofs << to_json_string(avail_cell_size);
            ofs.close();
            screen_out << " Successful. Number of cell = " << avail_cell_size << std::endl;
       }
        else
        {
            screen_out << " Fail !!!" << std::endl;
       }
   }

    template <typename T>
    T *get_json_array(const rapidjson::Document::Array &json_array)
    {
        T *res_mat = new T[json_array.Size()];
        for (size_t i = 0; i < json_array.Size(); i++)
        {
            res_mat[i] = (T) json_array[i].GetDouble();
       }
        return res_mat;
   }

    int load_mapping_from_file(const std::string &file_name = std::string("./mapping.json"))
    {
        Common_tools::Timer timer;
        timer.tic("Load mapping from json file");
        FILE *fp = fopen(file_name.c_str(), "r");
        if (fp == nullptr)
        {
            std::cout << "load_mapping_from_file: " << file_name << " fail!" << std::endl;
            return 0;
       }
        else
        {
            m_json_file_name = file_name;
            char                      readBuffer[1 << 16];
            rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));

            rapidjson::Document doc;
            doc.ParseStream(is);
            if (doc.HasParseError())
            {
                printf("GetParseError, err_code =  %d\n", doc.GetParseError());
                return 0;
           }

            DATA_TYPE *pt_vec_data;
            size_t     pt_num;
            for (unsigned int i = 0; i < doc.Size(); ++i)
            {
                if (i == 0)
                {
                    set_resolution(doc[i]["Res"].GetDouble() * 2.0);
               }
                //cout << i << endl;
                Mapping_cell *cell = add_cell(Eigen::Matrix< DATA_TYPE, 3, 1 >(get_json_array<DATA_TYPE>(doc[i]["Center"].GetArray())));
                //Mapping_cell *cell = find_cell(Eigen::Matrix< DATA_TYPE, 3, 1 >(get_json_array<DATA_TYPE>(doc[i]["Center"].GetArray())));

                cell->m_mean = Eigen::Matrix< COMP_TYPE, 3, 1 >(get_json_array< COMP_TYPE >(doc[i]["Mean"].GetArray()));
                cell->m_cov_mat = Eigen::Matrix< COMP_TYPE, 3, 3 >(get_json_array< COMP_TYPE >(doc[i]["Cov"].GetArray()));
                cell->m_icov_mat = Eigen::Matrix< COMP_TYPE, 3, 3 >(get_json_array< COMP_TYPE >(doc[i]["Icov"].GetArray()));
                cell->m_eigen_vec = Eigen::Matrix< COMP_TYPE, 3, 3 >(get_json_array< COMP_TYPE >(doc[i]["Eig_vec"].GetArray()));
                cell->m_eigen_val = Eigen::Matrix< COMP_TYPE, 3, 1 >(get_json_array< COMP_TYPE >(doc[i]["Eig_val"].GetArray()));

                pt_num = doc[i]["Pt_num"].GetInt();
                cell->m_points_vec.resize(pt_num);
                pt_vec_data = get_json_array<DATA_TYPE>(doc[i]["Pt_vec"].GetArray());
                for (size_t pt_idx = 0; pt_idx < pt_num; pt_idx++)
                {
                    cell->m_points_vec[pt_idx] << pt_vec_data[pt_idx * 3 + 0], pt_vec_data[pt_idx * 3 + 1], pt_vec_data[pt_idx * 3 + 2];
                    cell->m_xyz_sum = cell->m_xyz_sum + cell->m_points_vec[pt_idx].template cast< COMP_TYPE >();
               }
                delete pt_vec_data;
                //cout << doc[i]["Mean"].GetArray().Size() << endl;
           }
            fclose(fp);
            //cout << "Cell number = "<< m_map_pt_cell.size() << endl;

            //m_cells_center->resize(m_map_pt_cell.size());
            //m_map_pt_cell.reserve(m_map_pt_cell.size());

            std::cout << timer.toc_string("Load mapping from json file") << std::endl;
            return m_map_pt_cell.size();
       }
   }

    std::vector< Eigen::Matrix< DATA_TYPE, 3, 1 > > load_pts_from_file(const std::string &file_name = std::string("./mapping.json"))
    {
        Common_tools::Timer timer;
        timer.tic("Load points from json file");
        FILE *                                          fp = fopen(file_name.c_str(), "r");
        std::vector< Eigen::Matrix< DATA_TYPE, 3, 1 > > res_vec;
        if (fp == nullptr)
        {
            return res_vec;
       }
        else
        {
            m_json_file_name = file_name;
            char                      readBuffer[1 << 16];
            rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));

            rapidjson::Document doc;
            doc.ParseStream(is);
            if (doc.HasParseError())
            {
                printf("GetParseError, error code = %d\n", doc.GetParseError());
                return res_vec;
           }

            DATA_TYPE *pt_vec_data;
            size_t     pt_num;

            for (unsigned int i = 0; i < doc.Size(); ++i)
            {
                std::vector< Eigen::Matrix< DATA_TYPE, 3, 1 > > pt_vec_cell;
                pt_num = doc[i]["Pt_num"].GetInt();
                pt_vec_cell.resize(pt_num);
                pt_vec_data = get_json_array<DATA_TYPE>(doc[i]["Pt_vec"].GetArray());
                for (size_t pt_idx = 0; pt_idx < pt_num; pt_idx++)
                {
                    pt_vec_cell[pt_idx] << pt_vec_data[pt_idx * 3 + 0], pt_vec_data[pt_idx * 3 + 1], pt_vec_data[pt_idx * 3 + 2];
               }
                res_vec.insert(res_vec.end(), pt_vec_cell.begin(), pt_vec_cell.end());
           }
            fclose(fp);
            std::cout << "****** Load point from:" << file_name << "  successful ****** " << std::endl;
            std::cout << timer.toc_string("Load points from json file") << std::endl;
       }
        return res_vec;
   }

    template <typename T>
    void eigen_decompose_of_featurevector(std::vector< Eigen::Matrix< T, 3, 1 > > &feature_vectors, Eigen::Matrix< T, 3, 3 > &eigen_vector, Eigen::Matrix< T, 3, 1 > &eigen_val)
    {
        Eigen::Matrix< double, 3, 3 > mat_cov;
        mat_cov.setIdentity();
        // mat_cov.setZero();
        for (size_t i = 0; i < feature_vectors.size(); i++)
        {
            mat_cov = mat_cov + (feature_vectors[i] * feature_vectors[i].transpose()).template cast< double >();
       }
        Eigen::SelfAdjointEigenSolver< Eigen::Matrix< double, 3, 3 > > eigensolver;
        eigensolver.compute(mat_cov);
        eigen_val = eigensolver.eigenvalues().template cast< T >();
        eigen_vector = eigensolver.eigenvectors().template cast< T >();
   }
    template <typename T>
    void feature_direction(Eigen::Matrix< T, 3, 1 > &vec_3d, int &theta_idx, int &beta_idx)
    {

        int    theta_res = THETA_RES;
        int    beta_res = BETA_RES;
        double theta_step = 360.0 / theta_res;
        double beta_step = 180.0 / beta_res;
        double theta = atan2(vec_3d[1], vec_3d[0]) * 57.3 + 180.0;
        //double beta = atan2(vec_3d[2], vec_3d.block< 2, 1 >(0, 0).norm()) * 57.3 + 90.0;
        double vec_norm = sqrt(vec_3d[1] * vec_3d[1] + vec_3d[0] * vec_3d[0]);
        double beta = atan2(vec_3d[2], vec_norm) * 57.3 + 90.0;
        theta_idx = ((int) (std::floor(theta / theta_step))) % theta_res;
        if (theta_idx < 0)
            theta_idx += theta_res;
        beta_idx = (std::floor(beta / beta_step));
#if ENABLE_DEBUG
        cout << vec_3d.transpose() << endl;
        printf("Theta = %.2f, beta = %.2f, idx = [%d, %d], res = %d \r\n", theta, beta, theta_idx, beta_idx);
#endif
   }

    template <typename T>
    static void refine_feature_img(Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic > &feature_img)
    {
        int rows = feature_img.rows();
        int cols = feature_img.cols();
        //printf("Rows = %d, cols = %d\r\n", rows, cols);
        if (feature_img.row(0).maxCoeff() < feature_img.row(rows - 1).maxCoeff())
        {
            feature_img = feature_img.colwise().reverse().eval();
       }
        //cout << feature_img.block(0, 0, 2, floor(cols / 2 - 1)) << endl;
        //cout << feature_img.block(0, floor(cols / 2), 2, floor(cols / 2-1)) << endl;

        //cout << feature_img.block(0, 0, 2, round(cols / 2))  << endl;
        //cout <<  feature_img.block(0, round(cols / 2), 2, round(cols / 2)) << endl;

        //if ((feature_img.block(0, 0, 2, floor(cols / 2 - 1))).maxCoeff() < (feature_img.block(0, floor(cols / 2), 2, floor(cols / 2 - 1))).maxCoeff())
        if ((feature_img.block(0, 0, 2, round(cols / 2))).maxCoeff() < (feature_img.block(0, round(cols / 2), 2, round(cols / 2))).maxCoeff())
        {
            feature_img = feature_img.rowwise().reverse().eval();
       }
        //cout << feature_img.block(0, 0, 2, round(cols / 2)) << endl;
        //cout << feature_img.block(0, round(cols / 2), 2, round(cols / 2)) << endl;
   }

    static float similiarity_of_two_image(const Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > &img_a, const Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > &img_b)
    {
        assert(((img_a.rows() == img_b.rows()) && (img_a.cols() == img_b.cols())));
        //cout << img_a.mean() << endl;
        //cout << (img_a.array() - img_a.mean()) << endl;
        auto img_sub_mea_a = img_a.array() - img_a.mean();
        auto img_sub_mea_b = img_b.array() - img_b.mean();

        float product = ((img_sub_mea_a).cwiseProduct(img_sub_mea_b)).mean();
        int   devide_size = img_a.rows() * img_a.cols() - 1;
        float std_a = (img_sub_mea_a.array().pow(2)).sum() / devide_size;
        float std_b = (img_sub_mea_b.array().pow(2)).sum() / devide_size;
        return sqrt(product * product / std_a / std_b);

        //return 0;
   };

    static float ratio_of_nonzero_in_img(const Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > &img)
    {
        int count = 0;
        for (int i = 0; i < img.rows(); i++)
        {
            for (int j = 0; j < img.cols(); j++)
                if (img(i, j) >= 1.0)
                    count++;
       }
        return (float) (count) / (img.rows() * img.cols());
   }

    //template <typename T>
    std::vector< PT_TYPE > query_point_cloud(std::vector<Mapping_cell *> &cell_vec)
    {
        std::vector< std::vector< PT_TYPE > > pt_vec_vec;
        pt_vec_vec.reserve(1000);
        for (int i = 0; i < cell_vec.size(); i++)
        {
            pt_vec_vec.push_back(cell_vec[i]->get_pointcloud_eigen());
       }
        return Common_tools::vector_2d_to_1d(pt_vec_vec);
   }

    static float max_similiarity_of_two_image(const Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > &img_a, const Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > &img_b, float minimum_zero_ratio = 0.00)
    {
        if (ratio_of_nonzero_in_img(img_a) < minimum_zero_ratio)
        {
            return 0;
       }

        if (ratio_of_nonzero_in_img(img_b) < minimum_zero_ratio)
        {
            return 0;
       }
        size_t                                                 cols = img_a.cols();
        size_t                                                 rows = img_a.rows();
        Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > img_a_roi = img_a.block(0, 0, (int) std::round(rows / 2), cols);
        float                                                  max_res = -0;

        cv::Mat                                                hist_a, hist_b;
        Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > img_b_roi;
        img_b_roi.resize(rows, cols);
        float res = 0;
        for (size_t i = 0; i < rows; i++)
        {
            //Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > img_b_roi = img_b.block(i, 0, (int) std::round(rows / 2), cols);
            Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > img_b_roi_up = img_b.block(i, 0, rows - i, cols);
            Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > img_b_roi_down = img_b.block(0, 0, i, cols);

            img_b_roi << img_b_roi_up, img_b_roi_down;
            //refine_feature_img(img_b_roi);
            //refine_feature_img(img_b_roi);

            /*
            cv::eigen2cv(img_a_roi, hist_a);
            cv::eigen2cv(img_b_roi, hist_b);*/

            //float res = similiarity_of_two_image(img_a.block(0, 0, (int)(rows / 2), cols), img_b_roi.block(0, 0, (int)(rows / 2), cols));

            //res = similiarity_of_two_image(img_a, img_b_roi);
            if (1)
            {
                auto temp_a = img_a;
                auto temp_b = img_b_roi;
                refine_feature_img(temp_a);
                refine_feature_img(temp_b);
                res = similiarity_of_two_image_cv(temp_a, temp_b);
           }
            else
            {
                res = similiarity_of_two_image_cv(img_a, img_b_roi);
           }
            //float res = similiarity_of_two_image_cv(img_a, img_b_roi);
            //std::cout << res << " -- " << max_res << std::endl;
            //std::cout << i << " --- " << res << std::endl;
            if (fabs(res) > fabs(max_res))
                max_res = res;
       }

        //std::cout << hist_a + hist_b << std::endl;
        return max_res;
   }

    static float similiarity_of_two_image_cv(const Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > &img_a, const Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > &img_b, int method = CV_COMP_CORREL)
    {
        cv::Mat hist_a, hist_b;
        cv::eigen2cv(img_a, hist_a);
        cv::eigen2cv(img_b, hist_b);
        return cv::compareHist(hist_a, hist_b, method); // https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=comparehist
   }

    static float max_similiarity_of_two_image_cv(const Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > &img_a, const Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > &img_b, int method = CV_COMP_CORREL)
    {
        cv::Mat                                                hist_a, hist_b;
        int                                                    cols = img_a.cols();
        int                                                    rows = img_a.rows();
        Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > img_a_roi = img_a.block(0, 0, (int) std::round(rows / 2), cols);
        cv::eigen2cv(img_a_roi, hist_a);
        cv::eigen2cv(img_b, hist_b);
        cv::Mat result;
        cv::matchTemplate(hist_b, hist_a, result, CV_TM_CCORR_NORMED);
        double    minVal;
        double    maxVal;
        cv::Point minLoc;
        cv::Point maxLoc;
        minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());
        return maxVal;
        //return cv::compareHist(hist_a, hist_b, method); // https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=comparehist
   }

    int extract_feature_mapping(std::vector<Mapping_cell *>                           cell_vec,
                                 Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > &feature_img_line,
                                 Eigen::Matrix< float, Eigen::Dynamic, Eigen::Dynamic > &feature_img_plane,
                                 int                                                     if_recompute = 0)
    {

        Map_pt_cell_it                                  it;
        std::vector< Eigen::Matrix< DATA_TYPE, 3, 1 > > m_feature_vecs_plane, m_feature_vecs_line;
        for (size_t i = 0; i < cell_vec.size(); i++)
        {
            Mapping_cell *cell = cell_vec[i];

            auto feature_type = cell->determine_feature(if_recompute);
            cell->robust_covmat();
            if (feature_type == Feature_type::e_feature_line)
            {
                m_feature_vecs_line.push_back(cell->m_feature_vector.template cast<DATA_TYPE>());
                m_feature_vecs_line.push_back(-1.0 * cell->m_feature_vector.template cast<DATA_TYPE>());
           }
            if (feature_type == Feature_type::e_feature_plane)
            {
                m_feature_vecs_plane.push_back(cell->m_feature_vector.template cast<DATA_TYPE>());
                m_feature_vecs_plane.push_back(-1.0 * cell->m_feature_vector.template cast<DATA_TYPE>());
           }
       }
        screen_out << "Total cell numbers: " << cell_vec.size();
        screen_out << " ,line feature have : " << m_feature_vecs_line.size();
        screen_out << " ,plane feature have : " << m_feature_vecs_plane.size() << std::endl;

        Eigen::Matrix< DATA_TYPE, 3, 3 > eigen_vector;
        Eigen::Matrix< DATA_TYPE, 3, 1 > eigen_val;

        eigen_decompose_of_featurevector(m_feature_vecs_plane, eigen_vector, eigen_val);
        eigen_vector = eigen_vector.rowwise().reverse().eval();
        eigen_val = eigen_val.colwise().reverse().eval();
        eigen_vector.col(2) = eigen_vector.col(0).cross(eigen_vector.col(1));
        //screen_out << "Eigen value = " << eigen_val.transpose() << std::endl;

        //cout << "Eigen values: \n" << eigen_val << endl;
        //cout << "Eigen vector: \n" << eigen_vector << endl;
        feature_img_line.resize(THETA_RES, BETA_RES);
        feature_img_plane.resize(THETA_RES, BETA_RES);

        feature_img_line.setZero();
        feature_img_plane.setZero();

        int                              theta_idx = 0;
        int                              beta_idx = 0;
        Eigen::Matrix< DATA_TYPE, 3, 1 > affined_vector;
        //cout << __FILE__ << " -- " << __LINE__ << endl;
        for (size_t i = 0; i < m_feature_vecs_plane.size(); i++)
        {
            affined_vector = eigen_vector.transpose() * m_feature_vecs_plane[i];
            feature_direction(affined_vector, theta_idx, beta_idx);
            feature_img_plane(theta_idx, beta_idx) = feature_img_plane(theta_idx, beta_idx) + 1;

            affined_vector = affined_vector * -1.0;
            feature_direction((affined_vector), theta_idx, beta_idx);
            feature_img_plane(theta_idx, beta_idx) = feature_img_plane(theta_idx, beta_idx) + 1;
            //std::cout << "theta_idx = " << theta_idx << ", beta_idx = " << beta_idx << std::endl;
       }

        //cout << __FILE__ << " -- " << __LINE__ << endl;
        for (size_t i = 0; i < m_feature_vecs_line.size(); i++)
        {
            affined_vector = eigen_vector.transpose() * m_feature_vecs_line[i];
            feature_direction(affined_vector, theta_idx, beta_idx);
            //std::cout << "theta_idx = " << theta_idx << ", beta_idx = " << beta_idx << std::endl;
            feature_img_line(theta_idx, beta_idx) = feature_img_line(theta_idx, beta_idx) + 1;

            affined_vector = affined_vector * -1.0;
            feature_direction((affined_vector), theta_idx, beta_idx);
            feature_img_line(theta_idx, beta_idx) = feature_img_line(theta_idx, beta_idx) + 1;
       }

        //cout << __FILE__ << " -- " << __LINE__ << endl;
        if (1)
        {
            refine_feature_img(feature_img_plane);
            refine_feature_img(feature_img_line);
       }
        if (1)
        {
            cv::Mat feature_img_plane_cv, feature_img_line_cv;
            cv::eigen2cv(feature_img_line, feature_img_line_cv);
            cv::eigen2cv(feature_img_plane, feature_img_plane_cv);
            cv::Size kernel_size = cv::Size(5, 5);
            m_ratio_nonzero_line = ratio_of_nonzero_in_img(feature_img_line);
            m_ratio_nonzero_plane = ratio_of_nonzero_in_img(feature_img_plane);
            if (1)
            {
                float sigma = 5;
                cv::GaussianBlur(feature_img_plane_cv, feature_img_plane_cv, kernel_size, sigma);
                cv::GaussianBlur(feature_img_line_cv, feature_img_line_cv, kernel_size, sigma);
                cv::cv2eigen(feature_img_plane_cv, feature_img_plane);
                cv::cv2eigen(feature_img_line_cv, feature_img_line);
           }
       }

        //cout << "---- plane feature ----\r\n" << m_feature_img_plane.cast<int>() <<endl;
        //cout << "---- line  feature ----\r\n" << m_feature_img_line.cast<int>() <<endl;

        return 0;
   };

    int analyze_mapping(int if_recompute = 0)
    {

        float                         ratio = 0.90;
        Eigen_Point                   cell_center;
        std::vector<Mapping_cell *> cell_vec;

        m_roi_range = distributions_of_cell(cell_center, ratio);

        cell_vec = find_cells_in_radius(get_center(), m_roi_range);
        extract_feature_mapping(cell_vec, m_feature_img_line_roi, m_feature_img_plane_roi, if_recompute);
        extract_feature_mapping(m_cell_vec, m_feature_img_line, m_feature_img_plane, if_recompute);
        return 0;
   }

    pcl::PointCloud<pcl_pt> extract_specify_points(Feature_type select_type)
    {
        pcl::PointCloud<pcl_pt> res_pt;
        for (size_t i = 0; i < m_cell_vec.size(); i++)
        {
            if (m_cell_vec[i]->m_feature_type == select_type)
            {
                res_pt += m_cell_vec[i]->get_pointcloud();
           }
       }
        return res_pt;
   }

    pcl::PointCloud<pcl_pt> get_all_pointcloud()
    {
        pcl::PointCloud<pcl_pt> res_pt;
        for (size_t i = 0; i < m_cell_vec.size(); i++)
        {

            res_pt += m_cell_vec[i]->get_pointcloud();
       }
        return res_pt;
   }
};
