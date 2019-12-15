#ifndef UTILS_MATH_HPP
#define UTILS_MATH_HPP
#include <iostream>
#include <eigen3/Eigen/Dense>

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

#endif // UTILS_MATH_HPP