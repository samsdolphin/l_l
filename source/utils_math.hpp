#ifndef UTILS_MATH_HPP
#define UTILS_MATH_HPP
#include <iostream>
#include <eigen3/Eigen/Dense>

#define SMALL_EPS 1e-10

Eigen::Matrix3d euler2rot(Eigen::Vector3d av)
{
    double a = av(0);
    double b = av(1);
    double g = av(2);
    Eigen::Matrix3d Rx, Ry, Rz;
    Rx << 1, 0, 0, 0, cos(a), -sin(a), 0, sin(a), cos(a);
    Ry << cos(b), 0, sin(b), 0, 1, 0, -sin(b), 0, cos(b);
    Rz << cos(g), -sin(g), 0, sin(g), cos(g), 0, 0, 0, 1;
    return Rz * Ry * Rx;
}

Eigen::Matrix<double, 1, 3> sign(Eigen::Vector3d in)
{
    Eigen::Vector3d out;
    for (int i = 0; i < 3; i++)
        out(i) = (in(i) > 0) ? 1 : ((in(i) < 0) ? -1 : 0);

    return out.transpose();
}

Eigen::Matrix3d hat(Eigen::Vector3d vec)
{
    Eigen::Matrix3d R;
    R.setZero();
    R(0, 1) = -vec(2);
    R(1, 0) = vec(2);
    R(0, 2) = vec(1);
    R(2, 0) = -vec(1);
    R(1, 2) = -vec(0);
    R(2, 1) = vec(0);
    return R;
}

Eigen::Matrix3d A(Eigen::Vector3d v)
{
    double nm = v.norm();
    Eigen::Matrix3d R = Eigen::MatrixXd::Identity(3, 3);
    if (nm < SMALL_EPS)
        return R;
    R += (1 - cos(nm)) / pow(nm, 2) * hat(v);
    R += (1 - sin(nm) / nm) * hat(v) * hat(v) / pow(nm, 2);
    return R;
}

inline Eigen::Vector3d toAngleAxis(const Eigen::Quaterniond& quaterd, double* angle = NULL)
{
    Eigen::Quaterniond unit_quaternion = quaterd.normalized();
    double n = unit_quaternion.vec().norm();
    double w = unit_quaternion.w();
    double squared_w = w * w;

    double two_atan_nbyw_by_n;

    if (n < SMALL_EPS)
    {
        assert(fabs(w) > SMALL_EPS);
        two_atan_nbyw_by_n = 2. / w - 2. * (n * n) / (w * squared_w);
    }
    else
    {
        if (fabs(w) < SMALL_EPS)
        {
            if (w > 0)
                two_atan_nbyw_by_n = M_PI / n;
            else
                two_atan_nbyw_by_n = -M_PI / n;
        }
        two_atan_nbyw_by_n = 2 * atan(n / w) / n;
    }
    if (angle != NULL)
        *angle = two_atan_nbyw_by_n * n;
    return two_atan_nbyw_by_n * unit_quaternion.vec();
}

inline Eigen::Quaterniond toQuaterniond(const Eigen::Vector3d& v3d, double* angle = NULL)
{
    double theta = v3d.norm();
    if(angle != NULL)
        *angle = theta;
    double half_theta = 0.5*theta;

    double imag_factor;
    double real_factor = cos(half_theta);
    if(theta < SMALL_EPS)
    {
        double theta_sq = theta * theta;
        double theta_po4 = theta_sq * theta_sq;
        imag_factor = 0.5 - 0.0208333 * theta_sq + 0.000260417 * theta_po4;
    }
    else
    {
        double sin_half_theta = sin(half_theta);
        imag_factor = sin_half_theta / theta;
    }

    return Eigen::Quaterniond(real_factor, imag_factor * v3d.x(), imag_factor * v3d.y(), imag_factor * v3d.z());
}

double cost_func(std::vector<Eigen::Vector3d> line,
                 std::vector<Eigen::Vector3d> plane,
                 std::vector<Eigen::Vector3d> tarA,
                 std::vector<Eigen::Vector3d> curPt,
                 Eigen::Quaterniond q_last,
                 Eigen::Vector3d p_last)
{
    Eigen::Vector3d l;
    Eigen::Matrix3d L;
    size_t num = tarA.size();
    double cost = 0.0;
    for (size_t i = 0; i < num; i++)
    {
        if (i < line.size())
        {
            l = line[i];
            L = Eigen::MatrixXd::Identity(3, 3) - l * l.transpose() / pow(l.norm(), 2);
        }   
        else
        {
            l = plane[i - line.size()];
            L = l * l.transpose() / pow(l.norm(), 2);
        }
        Eigen::Vector3d cur_pt = curPt[i];
        Eigen::Vector3d pa = tarA[i];
        Eigen::Vector3d temp = L * (q_last * cur_pt + p_last - pa);
        Eigen::Matrix<double, 1, 3> sig = sign(temp);
        cost += pow(sig * temp, 2);
    }
    return cost;
}

#endif // UTILS_MATH_HPP