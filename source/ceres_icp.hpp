// Author: Jiarong Lin          ziv.lin.ljr@gmail.com
// Modified: Xiyuan Liu         liuxiyuan95@gmail.com

#ifndef CERES_ICP_HPP
#define CERES_ICP_HPP

#define MAX_LOG_LEVEL -100

#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <Eigen/Eigen>
#include <math.h>
#include <stdio.h>
#include <ceres/ceres.h>
#include "tools_eigen_math.hpp"
#include "utils_math.hpp"

template <typename _T>
struct ceres_icp_point2line
{
    Eigen::Matrix<_T, 3, 1> current_pt_;
    Eigen::Matrix<_T, 3, 1> target_a, target_b;
    Eigen::Matrix<_T, 3, 1> unit_vec_ab;
    Eigen::Matrix<_T, 4, 1> q_last_;
    Eigen::Matrix<_T, 3, 1> t_last_;
    _T weigh;
    ceres_icp_point2line(const Eigen::Matrix<_T, 3, 1> &current_pt,
                         const Eigen::Matrix<_T, 3, 1> &target_line_a,
                         const Eigen::Matrix<_T, 3, 1> &target_line_b,
                         Eigen::Matrix<_T, 4, 1> q_s = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                         Eigen::Matrix<_T, 3, 1> t_s = Eigen::Matrix<_T, 3, 1>(0, 0, 0)) : current_pt_(current_pt),
                                                                                           target_a(target_line_a),
                                                                                           target_b(target_line_b),
                                                                                           q_last_(q_s),
                                                                                           t_last_(t_s)
    {
        unit_vec_ab = target_line_b - target_line_a;
        unit_vec_ab = unit_vec_ab / unit_vec_ab.norm();
        weigh = 1.0;
    };

    template <typename T>
    bool operator()(const T *_a, const T *_t, T *residual) const
    {
        Eigen::Quaternion<T> q_last{(T)q_last_(0), (T)q_last_(1), (T)q_last_(2), (T)q_last_(3)};
        Eigen::Matrix<T, 3, 1> t_last = t_last_.template cast<T>();
        Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
        Eigen::Matrix<T, 3, 3> R_incre;

        R_incre << cos(_a[2])*cos(_a[1]), -sin(_a[2])*cos(_a[0])+cos(_a[2])*sin(_a[1])*sin(_a[0]), sin(_a[2])*sin(_a[0])+cos(_a[2])*sin(_a[1])*cos(_a[0]),
                   sin(_a[2])*cos(_a[1]), cos(_a[2])*cos(_a[0])+sin(_a[2])*sin(_a[1])*sin(_a[0]), -cos(_a[2])*sin(_a[0])+sin(_a[2])*sin(_a[1])*cos(_a[0]),
                   -sin(_a[1]), cos(_a[1])*sin(_a[0]), cos(_a[1])*cos(_a[0]);

        Eigen::Matrix<T, 3, 1> pt = current_pt_.template cast<T>();
        Eigen::Matrix<T, 3, 1> pt_transfromed;
        pt_transfromed = q_last * (R_incre * pt + t_incre) + t_last;

        Eigen::Matrix<T, 3, 1> tar_pt_a = target_a.template cast<T>();
        Eigen::Matrix<T, 3, 1> vec_line_ab_unit = unit_vec_ab.template cast<T>();

        Eigen::Matrix<T, 3, 1> vec_ac = pt_transfromed - tar_pt_a;
        Eigen::Matrix<T, 3, 1> residual_vec = vec_ac - Eigen_math::vector_project_on_unit_vector(vec_ac, vec_line_ab_unit);

        residual[0] = residual_vec(0) * T(weigh);
        residual[1] = residual_vec(1) * T(weigh);
        residual[2] = residual_vec(2) * T(weigh);

        return true;
    };

    static ceres::CostFunction *Create(const Eigen::Matrix<_T, 3, 1> &current_pt,
                                       const Eigen::Matrix<_T, 3, 1> &target_line_a,
                                       const Eigen::Matrix<_T, 3, 1> &target_line_b,
                                       Eigen::Matrix<_T, 4, 1> q_last = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                                       Eigen::Matrix<_T, 3, 1> t_last = Eigen::Matrix<_T, 3, 1>(0, 0, 0))
    {
        ROS_INFO_ONCE("AUTO_DIFF_FUNCTION_USED");
        return (new ceres::AutoDiffCostFunction<ceres_icp_point2line, 3, 3, 3>(
            new ceres_icp_point2line(current_pt, target_line_a, target_line_b, q_last, t_last)));
    }
};

template <typename _T>
struct ceres_icp_point2plane
{
    Eigen::Matrix<_T, 3, 1> current_pt_;
    Eigen::Matrix<_T, 3, 1> target_a, target_b, target_c;
    Eigen::Matrix<_T, 3, 1> unit_vec_ab, unit_vec_ac, unit_vec_n;
    _T weigh;
    Eigen::Matrix<_T, 4, 1> q_last_;
    Eigen::Matrix<_T, 3, 1> t_last_;
    ceres_icp_point2plane(const Eigen::Matrix<_T, 3, 1> &current_pt,
                          const Eigen::Matrix<_T, 3, 1> &target_line_a,
                          const Eigen::Matrix<_T, 3, 1> &target_line_b,
                          const Eigen::Matrix<_T, 3, 1> &target_line_c,
                          Eigen::Matrix<_T, 4, 1> q_s = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                          Eigen::Matrix<_T, 3, 1> t_s = Eigen::Matrix<_T, 3, 1>(0, 0, 0)) : current_pt_(current_pt), target_a(target_line_a),
                                                                                            target_b(target_line_b),
                                                                                            target_c(target_line_c),
                                                                                            q_last_(q_s),
                                                                                            t_last_(t_s)
    {
        unit_vec_ab = target_line_b - target_line_a;
        unit_vec_ab = unit_vec_ab / unit_vec_ab.norm();

        unit_vec_ac = target_line_c - target_line_a;
        unit_vec_ac = unit_vec_ac / unit_vec_ac.norm();

        unit_vec_n = unit_vec_ab.cross(unit_vec_ac);
        weigh = 1.0;
    };

    template <typename T>
    bool operator()(const T *_a, const T *_t, T *residual) const
    {
        Eigen::Quaternion<T> q_last{(T)q_last_(0), (T)q_last_(1), (T)q_last_(2), (T)q_last_(3)};
        Eigen::Matrix<T, 3, 1> t_last = t_last_.template cast<T>();
        Eigen::Matrix<T, 3, 1> t_incre{_t[0], _t[1], _t[2]};
        Eigen::Matrix<T, 3, 3> R_incre;

        R_incre << cos(_a[2])*cos(_a[1]), -sin(_a[2])*cos(_a[0])+cos(_a[2])*sin(_a[1])*sin(_a[0]), sin(_a[2])*sin(_a[0])+cos(_a[2])*sin(_a[1])*cos(_a[0]),
                   sin(_a[2])*cos(_a[1]), cos(_a[2])*cos(_a[0])+sin(_a[2])*sin(_a[1])*sin(_a[0]), -cos(_a[2])*sin(_a[0])+sin(_a[2])*sin(_a[1])*cos(_a[0]),
                   -sin(_a[1]), cos(_a[1])*sin(_a[0]), cos(_a[1])*cos(_a[0]);

        Eigen::Matrix<T, 3, 1> pt = current_pt_.template cast<T>();
        Eigen::Matrix<T, 3, 1> pt_transfromed;
        pt_transfromed = q_last * (R_incre * pt + t_incre) + t_last;

        Eigen::Matrix<T, 3, 1> tar_pt_a = target_a.template cast<T>();
        Eigen::Matrix<T, 3, 1> vec_line_plane_norm = unit_vec_n.template cast<T>();

        Eigen::Matrix<T, 3, 1> vec_ad = pt_transfromed - tar_pt_a;
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
                                       Eigen::Matrix<_T, 4, 1> q_last = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                                       Eigen::Matrix<_T, 3, 1> t_last = Eigen::Matrix<_T, 3, 1>(0, 0, 0))
    {
        return (new ceres::AutoDiffCostFunction<ceres_icp_point2plane, 3, 3, 3>(
            new ceres_icp_point2plane(current_pt, target_line_a, target_line_b, target_line_c, q_last, t_last)));
    }
};

template <typename _T>
class Point2Line : public ceres::SizedCostFunction<3, 3, 3> 
{
public:
    Point2Line(const Eigen::Matrix<_T, 3, 1> &current_pt,
               const Eigen::Matrix<_T, 3, 1> &target_line_a,
               const Eigen::Matrix<_T, 3, 1> &target_line_b,
               Eigen::Matrix<_T, 4, 1> q_last = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
               Eigen::Matrix<_T, 3, 1> t_last = Eigen::Matrix<_T, 3, 1>(0, 0, 0));
    virtual ~Point2Line(){}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

public:
    Eigen::Matrix<_T, 3, 1> current_pt_;
    Eigen::Matrix<_T, 3, 1> target_a, target_b;
    Eigen::Matrix<_T, 3, 1> unit_vec_ab;
    _T weigh;
    Eigen::Matrix<_T, 4, 1> q_last_;
    Eigen::Matrix<_T, 3, 1> t_last_;
};

template <typename _T>
Point2Line<_T>::Point2Line(const Eigen::Matrix<_T, 3, 1>& current_pt,
                           const Eigen::Matrix<_T, 3, 1>& target_line_a,
                           const Eigen::Matrix<_T, 3, 1>& target_line_b,
                           Eigen::Matrix<_T, 4, 1> q_last,
                           Eigen::Matrix<_T, 3, 1> t_last) : current_pt_(current_pt),
                                                             target_a(target_line_a),
                                                             target_b(target_line_b),
                                                             q_last_(q_last),
                                                             t_last_(t_last)
{
    ROS_INFO_ONCE("SIZED_COST_FUNCTION_USED");
    unit_vec_ab = target_line_b - target_line_a;
    unit_vec_ab = unit_vec_ab / unit_vec_ab.norm();
    weigh = 1.0;
}

template <typename _T>
bool Point2Line<_T>::Evaluate(double const* const* parameters,
                              double* residuals,
                              double** jacobians) const
{
    double a = parameters[0][0];
    double b = parameters[0][1];
    double g = parameters[0][2];
    double x = parameters[1][0];
    double y = parameters[1][1];
    double z = parameters[1][2];

    Eigen::Quaternion<_T> q_last{q_last_(0), q_last_(1), q_last_(2), q_last_(3)};
    Eigen::Matrix<_T, 3, 1> t_incre{x, y, z};
    Eigen::Matrix<_T, 3, 3> R_incre;

    R_incre << cos(g)*cos(b), -sin(g)*cos(a)+cos(g)*sin(b)*sin(a), sin(g)*sin(a)+cos(g)*sin(b)*cos(a),
               sin(g)*cos(b), cos(g)*cos(a)+sin(g)*sin(b)*sin(a), -cos(g)*sin(a)+sin(g)*sin(b)*cos(a),
               -sin(b), cos(b)*sin(a), cos(b)*cos(a);

    Eigen::Matrix<_T, 3, 1> pt_transfromed;
    pt_transfromed = q_last * (R_incre * current_pt_ + t_incre) + t_last_;

    Eigen::Matrix<_T, 3, 1> vec_ac = pt_transfromed - target_a;
    Eigen::Matrix<_T, 3, 1> residual_vec = vec_ac - Eigen_math::vector_project_on_unit_vector(vec_ac, unit_vec_ab);

    residuals[0] = residual_vec(0) * weigh;
    residuals[1] = residual_vec(1) * weigh;
    residuals[2] = residual_vec(2) * weigh;

    if (jacobians)
    {
        Eigen::Matrix<_T, 3, 3> Rx, Ry, Rz, parx, pary, parz, L;
        Rx << 1, 0, 0, 0, cos(a), -sin(a), 0, sin(a), cos(a);
        Ry << cos(b), 0, sin(b), 0, 1, 0, -sin(b), 0, cos(b);
        Rz << cos(g), -sin(g), 0, sin(g), cos(g), 0, 0, 0, 1;
        parx << 0, 0, 0, 0, -sin(a), -cos(a), 0, cos(a), -sin(a);
        pary << -sin(b), 0, cos(b), 0, 0, 0, -cos(b), 0, -sin(b);
        parz << -sin(g), -cos(g), 0, cos(g), -sin(g), 0, 0, 0, 0;
        L = Eigen::Matrix<_T, 3, 3>::Identity(3, 3) - unit_vec_ab * unit_vec_ab.transpose() / unit_vec_ab.norm() / unit_vec_ab.norm();
        if (jacobians[0])
        {
            Eigen::Matrix<_T, 3, 1> j_a = L * q_last * Rz * Ry * parx * current_pt_;
            Eigen::Matrix<_T, 3, 1> j_b = L * q_last * Rz * pary * Rx * current_pt_;
            Eigen::Matrix<_T, 3, 1> j_g = L * q_last * parz * Ry * Rx * current_pt_;
            for (int i = 0; i < 3; i++)
            {
                jacobians[0][0+i*3] = j_a(i);
                jacobians[0][1+i*3] = j_b(i);
                jacobians[0][2+i*3] = j_g(i);
            }
        }
        if (jacobians[1])
        {
            Eigen::Matrix<_T, 3, 1> p_x{1.0, 0.0, 0.0};
            Eigen::Matrix<_T, 3, 1> p_y{0.0, 1.0, 0.0};
            Eigen::Matrix<_T, 3, 1> p_z{0.0, 0.0, 1.0};
            Eigen::Matrix<_T, 3, 1> j_x = L * q_last * p_x;
            Eigen::Matrix<_T, 3, 1> j_y = L * q_last * p_y;
            Eigen::Matrix<_T, 3, 1> j_z = L * q_last * p_z;
            for (int i = 0; i < 3; i++)
            {
                jacobians[1][0+i*3] = j_x(i);
                jacobians[1][1+i*3] = j_y(i);
                jacobians[1][2+i*3] = j_z(i);
            }
        }
    }

    return true;
}

template <typename _T>
class Point2Plane : public ceres::SizedCostFunction<3, 3, 3> 
{
public:
    Point2Plane(const Eigen::Matrix<_T, 3, 1>& current_pt,
                const Eigen::Matrix<_T, 3, 1>& target_line_a,
                const Eigen::Matrix<_T, 3, 1>& target_line_b,
                const Eigen::Matrix<_T, 3, 1>& target_line_c,
                Eigen::Matrix<_T, 4, 1> q_last = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                Eigen::Matrix<_T, 3, 1> t_last = Eigen::Matrix<_T, 3, 1>(0, 0, 0));
    virtual ~Point2Plane(){}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;

public:
    Eigen::Matrix<_T, 3, 1> current_pt_;
    Eigen::Matrix<_T, 3, 1> target_a, target_b, target_c;
    Eigen::Matrix<_T, 3, 1> unit_vec_ab, unit_vec_ac, unit_vec_n;
    _T weigh;
    Eigen::Matrix<_T, 4, 1> q_last_;
    Eigen::Matrix<_T, 3, 1> t_last_;
};

template <typename _T>
Point2Plane<_T>::Point2Plane(const Eigen::Matrix<_T, 3, 1>& current_pt,
                             const Eigen::Matrix<_T, 3, 1>& target_line_a,
                             const Eigen::Matrix<_T, 3, 1>& target_line_b,
                             const Eigen::Matrix<_T, 3, 1>& target_line_c,
                             Eigen::Matrix<_T, 4, 1> q_last,
                             Eigen::Matrix<_T, 3, 1> t_last) : current_pt_(current_pt),
                                                               target_a(target_line_a),
                                                               target_b(target_line_b),
                                                               target_c(target_line_c),
                                                               q_last_(q_last),
                                                               t_last_(t_last)
{
    unit_vec_ab = target_line_b - target_line_a;
    unit_vec_ab = unit_vec_ab / unit_vec_ab.norm();

    unit_vec_ac = target_line_c - target_line_a;
    unit_vec_ac = unit_vec_ac / unit_vec_ac.norm();

    unit_vec_n = unit_vec_ab.cross(unit_vec_ac);
    weigh = 1.0;
}

template <typename _T>
bool Point2Plane<_T>::Evaluate(double const* const* parameters,
                               double* residuals,
                               double** jacobians) const
{
    double a = parameters[0][0];
    double b = parameters[0][1];
    double g = parameters[0][2];
    double x = parameters[1][0];
    double y = parameters[1][1];
    double z = parameters[1][2];

    Eigen::Quaternion<_T> q_last{q_last_(0), q_last_(1), q_last_(2), q_last_(3)};
    Eigen::Matrix<_T, 3, 1> t_incre{x, y, z};
    Eigen::Matrix<_T, 3, 3> R_incre;

    R_incre << cos(g)*cos(b), -sin(g)*cos(a)+cos(g)*sin(b)*sin(a), sin(g)*sin(a)+cos(g)*sin(b)*cos(a),
               sin(g)*cos(b), cos(g)*cos(a)+sin(g)*sin(b)*sin(a), -cos(g)*sin(a)+sin(g)*sin(b)*cos(a),
               -sin(b), cos(b)*sin(a), cos(b)*cos(a);

    Eigen::Matrix<_T, 3, 1> pt_transfromed;
    pt_transfromed = q_last * (R_incre * current_pt_ + t_incre) + t_last_;

    Eigen::Matrix<_T, 3, 1> vec_ad = pt_transfromed - target_a;
    Eigen::Matrix<_T, 3, 1> residual_vec = Eigen_math::vector_project_on_unit_vector(vec_ad, unit_vec_n);

    residuals[0] = residual_vec(0) * weigh;
    residuals[1] = residual_vec(1) * weigh;
    residuals[2] = residual_vec(2) * weigh;

    if (jacobians)
    {
        Eigen::Matrix<_T, 3, 3> Rx, Ry, Rz, parx, pary, parz, L;
        Rx << 1, 0, 0, 0, cos(a), -sin(a), 0, sin(a), cos(a);
        Ry << cos(b), 0, sin(b), 0, 1, 0, -sin(b), 0, cos(b);
        Rz << cos(g), -sin(g), 0, sin(g), cos(g), 0, 0, 0, 1;
        parx << 0, 0, 0, 0, -sin(a), -cos(a), 0, cos(a), -sin(a);
        pary << -sin(b), 0, cos(b), 0, 0, 0, -cos(b), 0, -sin(b);
        parz << -sin(g), -cos(g), 0, cos(g), -sin(g), 0, 0, 0, 0;
        L = unit_vec_n * unit_vec_n.transpose() / unit_vec_n.norm() / unit_vec_n.norm();
        if (jacobians[0])
        {
            Eigen::Matrix<_T, 3, 1> j_a = L * q_last * Rz * Ry * parx * current_pt_;
            Eigen::Matrix<_T, 3, 1> j_b = L * q_last * Rz * pary * Rx * current_pt_;
            Eigen::Matrix<_T, 3, 1> j_g = L * q_last * parz * Ry * Rx * current_pt_;
            for (int i = 0; i < 3; i++)
            {
                jacobians[0][0+i*3] = j_a(i);
                jacobians[0][1+i*3] = j_b(i);
                jacobians[0][2+i*3] = j_g(i);
            }
        }
        if (jacobians[1])
        {
            Eigen::Matrix<_T, 3, 1> p_x{1.0, 0.0, 0.0};
            Eigen::Matrix<_T, 3, 1> p_y{0.0, 1.0, 0.0};
            Eigen::Matrix<_T, 3, 1> p_z{0.0, 0.0, 1.0};
            Eigen::Matrix<_T, 3, 1> j_x = L * q_last * p_x;
            Eigen::Matrix<_T, 3, 1> j_y = L * q_last * p_y;
            Eigen::Matrix<_T, 3, 1> j_z = L * q_last * p_z;
            for (int i = 0; i < 3; i++)
            {
                jacobians[1][0+i*3] = j_x(i);
                jacobians[1][1+i*3] = j_y(i);
                jacobians[1][2+i*3] = j_z(i);
            }
        }
    }

    return true;
}

#endif // CERES_ICP_HPP
