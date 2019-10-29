// Author: Lin Jiarong          ziv.lin.ljr@gmail.com

#ifndef __ceres_icp_hpp__
#define __ceres_icp_hpp__
#define MAX_LOG_LEVEL -100
#include "eigen_math.hpp"
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <math.h>
#include <stdio.h>
#include <vector>
// Point to Point ICP
// Contour to contour ICP
// Plane to Plane ICP

//p2p with motion deblur
template <typename _T>
struct ceres_icp_point2point_mb
{
    Eigen::Matrix<_T, 3, 1> current_pt;
    Eigen::Matrix<_T, 3, 1> m_closest_pt;
    _T m_motion_blur_s;
    Eigen::Matrix<_T, 4, 1> q_last;
    Eigen::Matrix<_T, 3, 1> t_last;
    _T weigh;
    ceres_icp_point2point_mb(const Eigen::Matrix<_T, 3, 1> current_pt,
                           const Eigen::Matrix<_T, 3, 1> closest_pt,
                           const _T &motion_blur_s = 1.0,
                           Eigen::Matrix<_T, 4, 1> q_s = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                           Eigen::Matrix<_T, 3, 1> t_s = Eigen::Matrix<_T, 3, 1>(0, 0, 0)) : current_pt(current_pt),
                                                                                                m_closest_pt(closest_pt),
                                                                                                m_motion_blur_s(motion_blur_s),
                                                                                                q_last(q_s),
                                                                                                t_last(t_s)

    {
        weigh = 1.0;
   };

    template <typename T>
    bool operator()(const T *_q, const T *_t, T *residual) const
    {

        Eigen::Quaternion<T> q_last_{(T)q_last(0), (T)q_last(1), (T)q_last(2), (T)q_last(3)};
        Eigen::Matrix<T, 3, 1> t_last_ = t_last.template cast<T>();

        Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
        Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};

        Eigen::Quaternion<T> q_interpolate = Eigen::Quaternion<T>::Identity().slerp((T)m_motion_blur_s, q_incre);
        Eigen::Matrix<T, 3, 1> t_interpolate = t_incre * T(m_motion_blur_s);

        Eigen::Matrix<T, 3, 1> pt{T(current_pt(0)), T(current_pt(1)), T(current_pt(2))};
        Eigen::Matrix<T, 3, 1> pt_transfromed;
        pt_transfromed = q_last_ * (q_interpolate * pt + t_interpolate) + t_last_;

        //Eigen::Matrix< T, 3, 1 > vec_err = (pt_transfromed - m_closest_pt);

        residual[0] = (pt_transfromed(0) - T(m_closest_pt(0))) * T(weigh);
        residual[1] = (pt_transfromed(1) - T(m_closest_pt(1))) * T(weigh);
        residual[2] = (pt_transfromed(2) - T(m_closest_pt(2))) * T(weigh);
        return true;
   };

    static ceres::CostFunction *Create(const Eigen::Matrix<_T, 3, 1> current_pt,
                                        const Eigen::Matrix<_T, 3, 1> closest_pt,
                                        const _T motion_blur_s = 1.0,
                                        Eigen::Matrix<_T, 4, 1> q_s = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                                        Eigen::Matrix<_T, 3, 1> t_s = Eigen::Matrix<_T, 3, 1>(0, 0, 0))
    {
        return (new ceres::AutoDiffCostFunction<
                 ceres_icp_point2point_mb, 3, 4, 3>(
            new ceres_icp_point2point_mb(current_pt, closest_pt, motion_blur_s)));
   }
};

//point-to-line
template <typename _T>
struct ceres_icp_point2line_mb
{
    Eigen::Matrix<_T, 3, 1> current_pt;
    Eigen::Matrix<_T, 3, 1> target_line_a, target_line_b;
    Eigen::Matrix<_T, 3, 1> unit_vec_ab;
    _T m_motion_blur_s;
    Eigen::Matrix<_T, 4, 1> q_last;
    Eigen::Matrix<_T, 3, 1> t_last;
    _T weigh;
    ceres_icp_point2line_mb(const Eigen::Matrix<_T, 3, 1> &current_pt,
                          const Eigen::Matrix<_T, 3, 1> &target_line_a,
                          const Eigen::Matrix<_T, 3, 1> &target_line_b,
                          const _T motion_blur_s = 1.0,
                          Eigen::Matrix<_T, 4, 1> q_s = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                          Eigen::Matrix<_T, 3, 1> t_s = Eigen::Matrix<_T, 3, 1>(0, 0, 0)) : current_pt(current_pt), target_line_a(target_line_a),
                                                                                               target_line_b(target_line_b),
                                                                                               m_motion_blur_s(motion_blur_s),
                                                                                               q_last(q_s),
                                                                                               t_last(t_s)
    {
        unit_vec_ab = target_line_b - target_line_a;
        unit_vec_ab = unit_vec_ab / unit_vec_ab.norm();
        // weigh = 1/current_pt.norm();
        weigh = 1.0;
        //cout << unit_vec_ab.transpose() <<endl;
   };

    template <typename T>
    bool operator()(const T *_q, const T *_t, T *residual) const
    {

        Eigen::Quaternion<T> q_last_{(T)q_last(0), (T)q_last(1), (T)q_last(2), (T)q_last(3)};
        Eigen::Matrix<T, 3, 1> t_last_ = t_last.template cast<T>();

        Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
        Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};

        Eigen::Quaternion<T> q_interpolate = Eigen::Quaternion<T>::Identity().slerp((T)m_motion_blur_s, q_incre);
        Eigen::Matrix<T, 3, 1> t_interpolate = t_incre * T(m_motion_blur_s);

        Eigen::Matrix<T, 3, 1> pt = current_pt.template cast<T>();
        Eigen::Matrix<T, 3, 1> pt_transfromed;
        pt_transfromed = q_last_ * (q_interpolate * pt + t_interpolate) + t_last_;

        Eigen::Matrix<T, 3, 1> tar_line_pt_a = target_line_a.template cast<T>();
        Eigen::Matrix<T, 3, 1> vec_line_ab_unit = unit_vec_ab.template cast<T>();

        Eigen::Matrix<T, 3, 1> vec_ac = pt_transfromed - tar_line_pt_a;
        Eigen::Matrix<T, 3, 1> residual_vec = vec_ac - Eigen_math::vector_project_on_unit_vector(vec_ac, vec_line_ab_unit);

        residual[0] = residual_vec(0) * T(weigh);
        residual[1] = residual_vec(1) * T(weigh);
        residual[2] = residual_vec(2) * T(weigh);

        return true;
   };

    static ceres::CostFunction *Create(const Eigen::Matrix<_T, 3, 1> &current_pt,
                                        const Eigen::Matrix<_T, 3, 1> &target_line_a,
                                        const Eigen::Matrix<_T, 3, 1> &target_line_b,
                                        const _T motion_blur_s = 1.0,
                                        Eigen::Matrix<_T, 4, 1> q_last_ = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                                        Eigen::Matrix<_T, 3, 1> t_last_ = Eigen::Matrix<_T, 3, 1>(0, 0, 0))
    {
        // TODO: can be vector or distance
        return (new ceres::AutoDiffCostFunction<
                 ceres_icp_point2line_mb, 3, 4, 3>(
            new ceres_icp_point2line_mb(current_pt, target_line_a, target_line_b, motion_blur_s, q_last_, t_last_)));
   }
};

// point to plane with motion deblur
template <typename _T>
struct ceres_icp_point2plane_mb
{
    Eigen::Matrix<_T, 3, 1> current_pt;
    Eigen::Matrix<_T, 3, 1> target_line_a, target_line_b, target_line_c;
    Eigen::Matrix<_T, 3, 1> unit_vec_ab, unit_vec_ac, unit_vec_n;
    _T m_motion_blur_s;
    _T weigh;
    Eigen::Matrix<_T, 4, 1> q_last;
    Eigen::Matrix<_T, 3, 1> t_last;
    ceres_icp_point2plane_mb(const Eigen::Matrix<_T, 3, 1> &current_pt,
                           const Eigen::Matrix<_T, 3, 1> &target_line_a,
                           const Eigen::Matrix<_T, 3, 1> &target_line_b,
                           const Eigen::Matrix<_T, 3, 1> &target_line_c,
                           const _T motion_blur_s = 1.0,
                           Eigen::Matrix<_T, 4, 1> q_s = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                           Eigen::Matrix<_T, 3, 1> t_s = Eigen::Matrix<_T, 3, 1>(0, 0, 0)) : current_pt(current_pt), target_line_a(target_line_a),
                                                                                                target_line_b(target_line_b),
                                                                                                target_line_c(target_line_c),
                                                                                                m_motion_blur_s(motion_blur_s),
                                                                                                q_last(q_s),
                                                                                                t_last(t_s)

    {
        //assert(motion_blur_s <= 1.5 && motion_blur_s >= 0.0);
        //assert(motion_blur_s <= 1.01 && motion_blur_s >= 0.0);
        unit_vec_ab = target_line_b - target_line_a;
        unit_vec_ab = unit_vec_ab / unit_vec_ab.norm();

        unit_vec_ac = target_line_c - target_line_a;
        unit_vec_ac = unit_vec_ac / unit_vec_ac.norm();

        unit_vec_n = unit_vec_ab.cross(unit_vec_ac);
        weigh = 1.0;
   };

    template <typename T>
    bool operator()(const T *_q, const T *_t, T *residual) const
    {

        Eigen::Quaternion<T> q_last_{(T)q_last(0), (T)q_last(1), (T)q_last(2), (T)q_last(3)};
        Eigen::Matrix<T, 3, 1> t_last_ = t_last.template cast<T>();

        Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
        Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};

        Eigen::Quaternion<T> q_interpolate = Eigen::Quaternion<T>::Identity().slerp((T)m_motion_blur_s, q_incre);
        Eigen::Matrix<T, 3, 1> t_interpolate = t_incre * T(m_motion_blur_s);

        Eigen::Matrix<T, 3, 1> pt = current_pt.template cast<T>();
        Eigen::Matrix<T, 3, 1> pt_transfromed;
        pt_transfromed = q_last_ * (q_interpolate * pt + t_interpolate) + t_last_;

        Eigen::Matrix<T, 3, 1> tar_line_pt_a = target_line_a.template cast<T>();
        Eigen::Matrix<T, 3, 1> vec_line_plane_norm = unit_vec_n.template cast<T>();

        Eigen::Matrix<T, 3, 1> vec_ad = pt_transfromed - tar_line_pt_a;
        Eigen::Matrix<T, 3, 1> residual_vec = Eigen_math::vector_project_on_unit_vector(vec_ad, vec_line_plane_norm) * T(weigh);

        residual[0] = residual_vec(0) * T(weigh);
        residual[1] = residual_vec(1) * T(weigh);
        residual[2] = residual_vec(2) * T(weigh);
        //cout << residual_vec.rows() << "  " <<residual_vec.cols()  <<endl;

        //cout << " *** " << residual_vec[0] << "  " << residual_vec[1] << "  " << residual_vec[2] ;
        //cout << " *** " << residual[0] << "  " << residual[1] << "  " << residual[2] << " *** " << endl;
        return true;
   };

    static ceres::CostFunction *Create(const Eigen::Matrix<_T, 3, 1> &current_pt,
                                        const Eigen::Matrix<_T, 3, 1> &target_line_a,
                                        const Eigen::Matrix<_T, 3, 1> &target_line_b,
                                        const Eigen::Matrix<_T, 3, 1> &target_line_c,
                                        const _T motion_blur_s = 1.0,
                                        Eigen::Matrix<_T, 4, 1> q_last_ = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                                        Eigen::Matrix<_T, 3, 1> t_last_ = Eigen::Matrix<_T, 3, 1>(0, 0, 0))
    {
        // TODO: can be vector or distance
        return (new ceres::AutoDiffCostFunction<
                 ceres_icp_point2plane_mb, 3, 4, 3>(
            new ceres_icp_point2plane_mb(current_pt, target_line_a, target_line_b, target_line_c, motion_blur_s, q_last_, t_last_)));
   }
};


//point-to-line
template <typename _T>
struct ceres_icp_point2line
{
    Eigen::Matrix<_T, 3, 1> current_pt;
    Eigen::Matrix<_T, 3, 1> target_line_a, target_line_b;
    Eigen::Matrix<_T, 3, 1> unit_vec_ab;
    Eigen::Matrix<_T, 4, 1> q_last;
    Eigen::Matrix<_T, 3, 1> t_last;
    _T weigh;
    ceres_icp_point2line(const Eigen::Matrix<_T, 3, 1> &current_pt,
                         const Eigen::Matrix<_T, 3, 1> &target_line_a,
                         const Eigen::Matrix<_T, 3, 1> &target_line_b,
                         Eigen::Matrix<_T, 4, 1> q_s = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                         Eigen::Matrix<_T, 3, 1> t_s = Eigen::Matrix<_T, 3, 1>(0, 0, 0)) : current_pt(current_pt),
                                                                                           target_line_a(target_line_a),
                                                                                           target_line_b(target_line_b),
                                                                                           q_last(q_s),
                                                                                           t_last(t_s)
    {
        unit_vec_ab = target_line_b - target_line_a;
        unit_vec_ab = unit_vec_ab / unit_vec_ab.norm();
        weigh = 1.0;
    };

    template <typename T>
    bool operator()(const T *_q, const T *_t, T *residual) const
    {
        Eigen::Quaternion<T> q_last_{(T)q_last(0), (T)q_last(1), (T)q_last(2), (T)q_last(3)};
        Eigen::Matrix<T, 3, 1> t_last_ = t_last.template cast<T>();

        Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
        Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};

        Eigen::Matrix<T, 3, 1> pt = current_pt.template cast<T>();
        Eigen::Matrix<T, 3, 1> pt_transfromed;
        pt_transfromed = q_last_ * (q_incre * pt + t_incre) + t_last_;

        Eigen::Matrix<T, 3, 1> tar_line_pt_a = target_line_a.template cast<T>();
        Eigen::Matrix<T, 3, 1> vec_line_ab_unit = unit_vec_ab.template cast<T>();

        Eigen::Matrix<T, 3, 1> vec_ac = pt_transfromed - tar_line_pt_a;
        Eigen::Matrix<T, 3, 1> residual_vec = vec_ac - Eigen_math::vector_project_on_unit_vector(vec_ac, vec_line_ab_unit);

        residual[0] = residual_vec(0) * T(weigh);
        residual[1] = residual_vec(1) * T(weigh);
        residual[2] = residual_vec(2) * T(weigh);

        return true;
    };

    static ceres::CostFunction *Create(const Eigen::Matrix<_T, 3, 1> &current_pt,
                                       const Eigen::Matrix<_T, 3, 1> &target_line_a,
                                       const Eigen::Matrix<_T, 3, 1> &target_line_b,
                                       Eigen::Matrix<_T, 4, 1> q_last_ = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                                       Eigen::Matrix<_T, 3, 1> t_last_ = Eigen::Matrix<_T, 3, 1>(0, 0, 0))
    {
        // TODO: can be vector or distance
        return (new ceres::AutoDiffCostFunction<ceres_icp_point2line, 3, 4, 3>(
            new ceres_icp_point2line(current_pt, target_line_a, target_line_b, q_last_, t_last_)));
    }
};


// point to plane
template <typename _T>
struct ceres_icp_point2plane
{
    Eigen::Matrix<_T, 3, 1> current_pt;
    Eigen::Matrix<_T, 3, 1> target_line_a, target_line_b, target_line_c;
    Eigen::Matrix<_T, 3, 1> unit_vec_ab, unit_vec_ac, unit_vec_n;
    _T weigh;
    Eigen::Matrix<_T, 4, 1> q_last;
    Eigen::Matrix<_T, 3, 1> t_last;
    ceres_icp_point2plane(const Eigen::Matrix<_T, 3, 1> &current_pt,
                          const Eigen::Matrix<_T, 3, 1> &target_line_a,
                          const Eigen::Matrix<_T, 3, 1> &target_line_b,
                          const Eigen::Matrix<_T, 3, 1> &target_line_c,
                          Eigen::Matrix<_T, 4, 1> q_s = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                          Eigen::Matrix<_T, 3, 1> t_s = Eigen::Matrix<_T, 3, 1>(0, 0, 0)) : current_pt(current_pt),
                                                                                            target_line_a(target_line_a),
                                                                                            target_line_b(target_line_b),
                                                                                            target_line_c(target_line_c),
                                                                                            q_last(q_s),
                                                                                            t_last(t_s)
    {
        unit_vec_ab = target_line_b - target_line_a;
        unit_vec_ab = unit_vec_ab / unit_vec_ab.norm();

        unit_vec_ac = target_line_c - target_line_a;
        unit_vec_ac = unit_vec_ac / unit_vec_ac.norm();

        unit_vec_n = unit_vec_ab.cross(unit_vec_ac);
        weigh = 1.0;
    };

    template <typename T>
    bool operator()(const T *_q, const T *_t, T *residual) const
    {
        Eigen::Quaternion<T> q_last_{(T)q_last(0), (T)q_last(1), (T)q_last(2), (T)q_last(3)};
        Eigen::Matrix<T, 3, 1> t_last_ = t_last.template cast<T>();

        Eigen::Quaternion<T> q_incre{_q[3], _q[0], _q[1], _q[2]};
        Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};

        Eigen::Matrix<T, 3, 1> pt = current_pt.template cast<T>();
        Eigen::Matrix<T, 3, 1> pt_transfromed;
        pt_transfromed = q_last_ * (q_incre * pt + t_incre) + t_last_;

        Eigen::Matrix<T, 3, 1> tar_line_pt_a = target_line_a.template cast<T>();
        Eigen::Matrix<T, 3, 1> vec_line_plane_norm = unit_vec_n.template cast<T>();

        Eigen::Matrix<T, 3, 1> vec_ad = pt_transfromed - tar_line_pt_a;
        Eigen::Matrix<T, 3, 1> residual_vec = Eigen_math::vector_project_on_unit_vector(vec_ad, vec_line_plane_norm) * T(weigh);

        residual[0] = residual_vec(0) * T(weigh);
        residual[1] = residual_vec(1) * T(weigh);
        residual[2] = residual_vec(2) * T(weigh);

        return true;
    };

    static ceres::CostFunction *Create(const Eigen::Matrix<_T, 3, 1> &current_pt,
                                        const Eigen::Matrix<_T, 3, 1> &target_line_a,
                                        const Eigen::Matrix<_T, 3, 1> &target_line_b,
                                        const Eigen::Matrix<_T, 3, 1> &target_line_c,
                                        Eigen::Matrix<_T, 4, 1> q_last_ = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                                        Eigen::Matrix<_T, 3, 1> t_last_ = Eigen::Matrix<_T, 3, 1>(0, 0, 0))
    {
        // TODO: can be vector or distance
        return (new ceres::AutoDiffCostFunction<ceres_icp_point2plane, 3, 4, 3>(
            new ceres_icp_point2plane(current_pt, target_line_a, target_line_b, target_line_c,  q_last_, t_last_)));
    }
};


#endif
