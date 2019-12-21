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
#include <sophus/se3.hpp>

template <typename _T>
struct quaternion_point2line
{
    Eigen::Matrix<_T, 3, 1> m_current_pt;
    Eigen::Matrix<_T, 3, 1> m_target_line_a, m_target_line_b;
    Eigen::Matrix<_T, 3, 1> m_unit_vec_ab;
    Eigen::Matrix<_T, 4, 1> m_q_last;
    Eigen::Matrix<_T, 3, 1> m_t_last;
    _T m_weigh;
    quaternion_point2line(const Eigen::Matrix<_T, 3, 1> &current_pt,
                          const Eigen::Matrix<_T, 3, 1> &target_line_a,
                          const Eigen::Matrix<_T, 3, 1> &target_line_b,
                          Eigen::Matrix<_T, 4, 1> q_s = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                          Eigen::Matrix<_T, 3, 1> t_s = Eigen::Matrix<_T, 3, 1>(0, 0, 0)) : m_current_pt(current_pt),
                                                                                            m_target_line_a(target_line_a),
                                                                                            m_target_line_b(target_line_b),
                                                                                            m_q_last(q_s),
                                                                                            m_t_last(t_s)
    {
        m_unit_vec_ab = target_line_b - target_line_a;
        m_unit_vec_ab = m_unit_vec_ab / m_unit_vec_ab.norm();
        m_weigh = 1.0;
    };

    template <typename T>
    bool operator()(const T *_q, const T *_t, T *residual) const
    {
        Eigen::Quaternion<T> q_last{ (T) m_q_last(0), (T) m_q_last(1), (T) m_q_last(2), (T) m_q_last(3) };
        Eigen::Matrix<T, 3, 1> t_last = m_t_last.template cast<T>();

        Eigen::Quaternion<T> q_incre{ _q[3], _q[0], _q[1], _q[2] };
        Eigen::Matrix<T, 3, 1> t_incre{ _t[0], _t[1], _t[2] };

        Eigen::Matrix<T, 3, 1> pt = m_current_pt.template cast<T>();
        Eigen::Matrix<T, 3, 1> pt_transfromed;
        pt_transfromed = q_last * (q_incre * pt + t_incre) + t_last;

        Eigen::Matrix<T, 3, 1> tar_line_pt_a = m_target_line_a.template cast<T>();
        Eigen::Matrix<T, 3, 1> vec_line_ab_unit = m_unit_vec_ab.template cast<T>();

        Eigen::Matrix<T, 3, 1> vec_ac = pt_transfromed - tar_line_pt_a;
        Eigen::Matrix<T, 3, 1> residual_vec = vec_ac - Eigen_math::vector_project_on_unit_vector(vec_ac, vec_line_ab_unit);

        residual[0] = residual_vec(0) * T(m_weigh);
        residual[1] = residual_vec(1) * T(m_weigh);
        residual[2] = residual_vec(2) * T(m_weigh);

        return true;
    };

    static ceres::CostFunction *Create(const Eigen::Matrix<_T, 3, 1> &current_pt,
                                       const Eigen::Matrix<_T, 3, 1> &target_line_a,
                                       const Eigen::Matrix<_T, 3, 1> &target_line_b,
                                       Eigen::Matrix<_T, 4, 1> q_last = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                                       Eigen::Matrix<_T, 3, 1> t_last = Eigen::Matrix<_T, 3, 1>(0, 0, 0))
    {
        ROS_INFO_ONCE("QUATERNION_AUTO_DIFF_FUNCTION_USED");
        return (new ceres::AutoDiffCostFunction<quaternion_point2line, 3, 4, 3>(
            new quaternion_point2line(current_pt, target_line_a, target_line_b, q_last, t_last)));
    }
};

template <typename _T>
struct quaternion_point2plane
{
    Eigen::Matrix<_T, 3, 1> m_current_pt;
    Eigen::Matrix<_T, 3, 1> m_target_line_a, m_target_line_b, m_target_line_c;
    Eigen::Matrix<_T, 3, 1> m_unit_vec_ab, m_unit_vec_ac, m_unit_vec_n;
    _T m_weigh;
    Eigen::Matrix<_T, 4, 1> m_q_last;
    Eigen::Matrix<_T, 3, 1> m_t_last;
    quaternion_point2plane(const Eigen::Matrix<_T, 3, 1> &current_pt,
                           const Eigen::Matrix<_T, 3, 1> &target_line_a,
                           const Eigen::Matrix<_T, 3, 1> &target_line_b,
                           const Eigen::Matrix<_T, 3, 1> &target_line_c,
                           Eigen::Matrix<_T, 4, 1> q_s = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                           Eigen::Matrix<_T, 3, 1> t_s = Eigen::Matrix<_T, 3, 1>(0, 0, 0)) : m_current_pt(current_pt),
                                                                                             m_target_line_a(target_line_a),
                                                                                             m_target_line_b(target_line_b),
                                                                                             m_target_line_c(target_line_c),
                                                                                             m_q_last(q_s),
                                                                                             m_t_last(t_s)
    {
        m_unit_vec_ab = target_line_b - target_line_a;
        m_unit_vec_ab = m_unit_vec_ab / m_unit_vec_ab.norm();

        m_unit_vec_ac = target_line_c - target_line_a;
        m_unit_vec_ac = m_unit_vec_ac / m_unit_vec_ac.norm();

        m_unit_vec_n = m_unit_vec_ab.cross(m_unit_vec_ac);
        m_weigh = 1.0;
    };

    template <typename T>
    bool operator()(const T *_q, const T *_t, T *residual) const
    {
        Eigen::Quaternion<T> q_last{ (T) m_q_last(0), (T) m_q_last(1), (T) m_q_last(2), (T) m_q_last(3) };
        Eigen::Matrix<T, 3, 1> t_last = m_t_last.template cast<T>();

        Eigen::Quaternion<T> q_incre{ _q[3], _q[0], _q[1], _q[2] };
        Eigen::Matrix<T, 3, 1> t_incre{ _t[0], _t[1], _t[2] };

        Eigen::Matrix<T, 3, 1> pt = m_current_pt.template cast<T>();
        Eigen::Matrix<T, 3, 1> pt_transfromed;
        pt_transfromed = q_last * (q_incre * pt + t_incre) + t_last;

        Eigen::Matrix<T, 3, 1> tar_line_pt_a = m_target_line_a.template cast<T>();
        Eigen::Matrix<T, 3, 1> vec_line_plane_norm = m_unit_vec_n.template cast<T>();

        Eigen::Matrix<T, 3, 1> vec_ad = pt_transfromed - tar_line_pt_a;
        Eigen::Matrix<T, 3, 1> residual_vec = Eigen_math::vector_project_on_unit_vector(vec_ad, vec_line_plane_norm) * T(m_weigh);

        residual[0] = residual_vec(0) * T(m_weigh);
        residual[1] = residual_vec(1) * T(m_weigh);
        residual[2] = residual_vec(2) * T(m_weigh);

        return true;
    };

    static ceres::CostFunction *Create(const Eigen::Matrix<_T, 3, 1> &current_pt,
                                        const Eigen::Matrix<_T, 3, 1> &target_line_a,
                                        const Eigen::Matrix<_T, 3, 1> &target_line_b,
                                        const Eigen::Matrix<_T, 3, 1> &target_line_c,
                                        Eigen::Matrix<_T, 4, 1> q_last = Eigen::Matrix<_T, 4, 1>(1, 0, 0, 0),
                                        Eigen::Matrix<_T, 3, 1> t_last = Eigen::Matrix<_T, 3, 1>(0, 0, 0))
    {
        return (new ceres::AutoDiffCostFunction<quaternion_point2plane, 3, 4, 3>(
            new quaternion_point2plane(current_pt, target_line_a, target_line_b, target_line_c, q_last, t_last)));
    }
};

class QPoint2Line : public ceres::SizedCostFunction<3, 3, 3> 
{
public:
    QPoint2Line(const Eigen::Vector3d& current_pt,
                const Eigen::Vector3d& target_line_a,
                const Eigen::Vector3d& target_line_b,
                Eigen::Matrix<double, 4, 1> q_s = Eigen::Matrix<double, 4, 1>(1, 0, 0, 0),
                Eigen::Vector3d t_last = Eigen::Vector3d(0, 0, 0));
    virtual ~QPoint2Line(){}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;
public:
    Eigen::Vector3d current_pt_;
    Eigen::Vector3d target_a, target_b;
    Eigen::Vector3d unit_vec_ab;
    Eigen::Matrix<double, 4, 1> q_last_;
    Eigen::Vector3d t_last_;
};

QPoint2Line::QPoint2Line(const Eigen::Vector3d& current_pt,
                         const Eigen::Vector3d& target_line_a,
                         const Eigen::Vector3d& target_line_b,
                         Eigen::Matrix<double, 4, 1> q_s,
                         Eigen::Vector3d t_last) : current_pt_(current_pt),
                                                   target_a(target_line_a),
                                                   target_b(target_line_b),
                                                   q_last_(q_s),
                                                   t_last_(t_last)
{
    ROS_INFO_ONCE("SIZED_COST_FUNCTION_USED");
    unit_vec_ab = target_line_b - target_line_a;
    unit_vec_ab = unit_vec_ab / unit_vec_ab.norm();
}

bool QPoint2Line::Evaluate(double const* const* parameters,
                           double* residuals,
                           double** jacobians) const
{
    Eigen::Map<const Eigen::Vector3d>axis(parameters[0]);
    Eigen::Map<const Eigen::Vector3d>t_incre(parameters[1]);
    Eigen::Quaterniond q_incre = toQuaterniond(axis);
    Eigen::Quaterniond q_last(q_last_(0), q_last_(1), q_last_(2), q_last_(3));
    Eigen::Vector3d pt_transfromed;
    pt_transfromed = q_last * (q_incre * current_pt_ + t_incre) + t_last_;

    Eigen::Vector3d vec_ac = pt_transfromed - target_a;
    Eigen::Vector3d residual_vec = vec_ac - Eigen_math::vector_project_on_unit_vector(vec_ac, unit_vec_ab);

    residuals[0] = residual_vec(0);
    residuals[1] = residual_vec(1);
    residuals[2] = residual_vec(2);

    if (jacobians)
    {
        Eigen::Matrix3d L;
        L = Eigen::MatrixXd::Identity(3, 3) - unit_vec_ab * unit_vec_ab.transpose() / unit_vec_ab.norm() / unit_vec_ab.norm();
        if (jacobians[0])
        {
            Eigen::Matrix3d R = -L * q_last * hat(q_incre * current_pt_) * A(axis);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    jacobians[0][i * 3 + j] = R(i, j);
                    
        }
        if (jacobians[1])
        {
            Eigen::Vector3d p_x{1.0, 0.0, 0.0};
            Eigen::Vector3d p_y{0.0, 1.0, 0.0};
            Eigen::Vector3d p_z{0.0, 0.0, 1.0};
            Eigen::Vector3d j_x = L * q_last * p_x;
            Eigen::Vector3d j_y = L * q_last * p_y;
            Eigen::Vector3d j_z = L * q_last * p_z;
            for (int i = 0; i < 3; i++)
            {
                jacobians[1][0 + i * 3] = j_x(i);
                jacobians[1][1 + i * 3] = j_y(i);
                jacobians[1][2 + i * 3] = j_z(i);
            }
        }
    }

    return true;
}

class QPoint2Plane : public ceres::SizedCostFunction<3, 3, 3> 
{
public:
    QPoint2Plane(const Eigen::Vector3d& current_pt,
                 const Eigen::Vector3d& target_line_a,
                 const Eigen::Vector3d& target_line_b,
                 const Eigen::Vector3d& target_line_c,
                 Eigen::Matrix<double, 4, 1> q_s = Eigen::Matrix<double, 4, 1>(1, 0, 0, 0),
                 Eigen::Vector3d t_last = Eigen::Vector3d(0, 0, 0));
    virtual ~QPoint2Plane(){}
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const;
public:
    Eigen::Vector3d current_pt_;
    Eigen::Vector3d target_a, target_b, target_c;
    Eigen::Vector3d unit_vec_ab, unit_vec_ac, unit_vec_n;
    Eigen::Matrix<double, 4, 1> q_last_;
    Eigen::Vector3d t_last_;
};

QPoint2Plane::QPoint2Plane(const Eigen::Vector3d& current_pt,
                           const Eigen::Vector3d& target_line_a,
                           const Eigen::Vector3d& target_line_b,
                           const Eigen::Vector3d& target_line_c,
                           Eigen::Matrix<double, 4, 1> q_s,
                           Eigen::Vector3d t_last) : current_pt_(current_pt),
                                                     target_a(target_line_a),
                                                     target_b(target_line_b),
                                                     target_c(target_line_c),
                                                     q_last_(q_s),
                                                     t_last_(t_last)
{
    unit_vec_ab = target_line_b - target_line_a;
    unit_vec_ab = unit_vec_ab / unit_vec_ab.norm();

    unit_vec_ac = target_line_c - target_line_a;
    unit_vec_ac = unit_vec_ac / unit_vec_ac.norm();

    unit_vec_n = unit_vec_ab.cross(unit_vec_ac);
}

bool QPoint2Plane::Evaluate(double const* const* parameters,
                            double* residuals,
                            double** jacobians) const
{
    Eigen::Map<const Eigen::Vector3d>axis(parameters[0]);
    Eigen::Map<const Eigen::Vector3d>t_incre(parameters[1]);
    Eigen::Quaterniond q_incre = toQuaterniond(axis);
    Eigen::Quaterniond q_last(q_last_(0), q_last_(1), q_last_(2), q_last_(3));
    Eigen::Vector3d pt_transfromed;
    pt_transfromed = q_last * (q_incre * current_pt_ + t_incre) + t_last_;

    Eigen::Vector3d vec_ac = pt_transfromed - target_a;
    Eigen::Vector3d residual_vec = Eigen_math::vector_project_on_unit_vector(vec_ac, unit_vec_n);

    residuals[0] = residual_vec(0);
    residuals[1] = residual_vec(1);
    residuals[2] = residual_vec(2);

    if (jacobians)
    {
        Eigen::Matrix3d L;
        L = unit_vec_n * unit_vec_n.transpose() / unit_vec_n.norm() / unit_vec_n.norm();
        if (jacobians[0])
        {
            Eigen::Matrix3d R = -L * q_last * hat(q_incre * current_pt_) * A(axis);
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    jacobians[0][i * 3 + j] = R(i, j);
        }
        if (jacobians[1])
        {
            Eigen::Vector3d p_x{1.0, 0.0, 0.0};
            Eigen::Vector3d p_y{0.0, 1.0, 0.0};
            Eigen::Vector3d p_z{0.0, 0.0, 1.0};
            Eigen::Vector3d j_x = L * q_last * p_x;
            Eigen::Vector3d j_y = L * q_last * p_y;
            Eigen::Vector3d j_z = L * q_last * p_z;
            for (int i = 0; i < 3; i++)
            {
                jacobians[1][0 + i * 3] = j_x(i);
                jacobians[1][1 + i * 3] = j_y(i);
                jacobians[1][2 + i * 3] = j_z(i);
            }
        }
    }

    return true;
}

class LineParameterization : public ceres::LocalParameterization
{
public:
    LineParameterization(){}
    virtual ~LineParameterization(){}
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const;
    virtual bool ComputeJacobian(const double* x,
                                 double* jacobian) const;
    virtual int GlobalSize() const {return 3;}
    virtual int LocalSize() const {return 3;}
};

bool LineParameterization::Plus(const double* x,
                                const double* delta,
                                double* x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> so3_x(x);
    Sophus::SO3<double> SO3_x = Sophus::SO3<double>::exp(so3_x);
    Eigen::Map<const Eigen::Vector3d> so3_delta(delta);
    Sophus::SO3<double> SO3_delta = Sophus::SO3<double>::exp(so3_delta);
    Eigen::Map<Eigen::Vector3d> angles_plus(x_plus_delta);
    //angles_plus = (SO3_x * SO3_delta).log();
    angles_plus = so3_x + so3_delta;
    return true;
}

bool LineParameterization::ComputeJacobian(const double* x,
                                           double* jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobian);
    J.setIdentity();
    return true;
}

#endif // CERES_ICP_HPP
