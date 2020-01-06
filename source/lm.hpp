#ifndef LM_HPP
#define LM_HPP

#include <eigen3/Eigen/Dense>
#include <opencv/cv.h>
#include "utils_math.hpp"

class LM
{
public:
    LM(std::vector<Eigen::Vector3d> line,
       std::vector<Eigen::Vector3d> plane,
       std::vector<Eigen::Vector3d> tarA,
       std::vector<Eigen::Vector3d> curPt);
    ~LM(){}
    bool solve();
public:
    Eigen::Quaterniond q_last;
    Eigen::Vector3d t_last, a_inc, t_inc;
    double lambda, rho, thr, cost_ori, cost_aft;
    std::vector<Eigen::Vector3d> line_, plane_, tarA_, curPt_;
};

LM::LM(std::vector<Eigen::Vector3d> line,
       std::vector<Eigen::Vector3d> plane,
       std::vector<Eigen::Vector3d> tarA,
       std::vector<Eigen::Vector3d> curPt) : line_(line), plane_(plane), tarA_(tarA), curPt_(curPt)
{
    a_inc << 0, 0, 0;
    t_inc << 0, 0, 0;
    lambda = 1;
    rho = 1;
    thr = 10;
}

bool LM::solve()
{
    for (int it = 0; it < 20; it++)
    {
        size_t num = tarA_.size();
        Eigen::MatrixXd J(num, 6);
        Eigen::MatrixXd f0(num, 1);
        J.setZero();
        f0.setZero();
        Eigen::Matrix3d L;

        for (size_t i = 0; i < num; i++)
        {
            if (i < line_.size())
            {
                Eigen::Vector3d l = line_[i];
                L = Eigen::MatrixXd::Identity(3, 3) - l * l.transpose() / pow(l.norm(), 2);
            }
            else
            {
                Eigen::Vector3d l = plane_[i - line_.size()];
                L = l * l.transpose() / pow(l.norm(), 2);
            }
            Eigen::Vector3d cur_pt = curPt_[i];
            Eigen::Vector3d pa = tarA_[i];
            Eigen::Quaterniond q_inc = toQuaterniond(a_inc);
            Eigen::Vector3d temp = L * (q_last * (q_inc * cur_pt + t_inc) + t_last - pa);
            Eigen::Matrix<double, 1, 3> sig = sign(temp);
            f0(i, 0) = sig * temp;
            Eigen::Matrix3d R = -L * q_last * hat(q_inc * cur_pt) * A(a_inc);
            for (int j = 0; j < 3; j++)
                J(i, j) = sig * R.col(j);

            Eigen::Vector3d p_x{1.0, 0.0, 0.0};
            Eigen::Vector3d p_y{0.0, 1.0, 0.0};
            Eigen::Vector3d p_z{0.0, 0.0, 1.0};
            J(i, 3) = sig * (L * q_last * p_x);
            J(i, 4) = sig * (L * q_last * p_y);
            J(i, 5) = sig * (L * q_last * p_z);
        }

        Eigen::MatrixXd H(6, 6);
        H = J.transpose() * J;
        Eigen::MatrixXd JTf(6, 1);
        JTf = -J.transpose() * f0;
        Eigen::MatrixXd DD = Eigen::MatrixXd::Identity(6, 6);
        for (int i = 0; i < 6; i++)
            DD(i, i) = H(i, i);
        H += lambda * DD;
        cv::Mat matA(6, 6, CV_64F, cv::Scalar::all(0));
        cv::Mat matB(6, 1, CV_64F, cv::Scalar::all(0));
        cv::Mat matX(6, 1, CV_64F, cv::Scalar::all(0));

        for (int i = 0; i < 6; i++)
        {
            matB.at<double>(i, 0) = JTf(i, 0);
            for (int j = 0; j < 6; j++)
                matA.at<double>(i, j) = H(i, j);
        }
        cv::solve(matA, matB, matX, cv::DECOMP_QR);

        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver;
        eigensolver.compute(H);
        Eigen::Matrix<double, 6, 1> eig_vals = eigensolver.eigenvalues();
        Eigen::Matrix<double, 6, 6> eig_vecs = eigensolver.eigenvectors();
        Eigen::Matrix<double, 6, 6> Vf = eig_vecs;
        Eigen::Matrix<double, 6, 6> Vp = Eigen::MatrixXd::Zero(6, 6);
        Eigen::Matrix<double, 6, 6> Vu = Eigen::MatrixXd::Zero(6, 6);
        int direction = -1;

        for (int i = 0; i < 6; i++)
        {
            if (eig_vals(i) < thr)
            {
                Vp.col(i) = eig_vecs.col(i);
                if (i == 0)
                    direction = degenerate_dir(Vp.col(0));
            }
            else
                Vu.col(i) = eig_vecs.col(i);            
        }
        
        Eigen::Matrix<double, 6, 1> Xf, Xu, Xp;
        Eigen::Vector3d t_temp, a_temp;
        Eigen::Matrix<double, 6, 1> delta_x;
        for (int i = 0; i < 3; i++)
        {
            Xf(i) = matX.at<double>(i, 0);
            Xf(i + 3) = matX.at<double>(i + 3, 0);
            delta_x(i) = matX.at<double>(i, 0);
            delta_x(i + 3) = matX.at<double>(i + 3, 0);
        }

        Xu = Xf;
        Xp.setZero();
        if (direction != -1)
        {
            std::cout<<"--------------->          "<<direction<<std::endl;
            Xu(direction) = 0;
            Xp(direction) = Xf(direction);
        }
        
        Xf = Vf.transpose().inverse() * Vp.transpose() * Xp + Vf.transpose().inverse() * Vu.transpose() * Xu;
        a_temp = Xf.block<3, 1>(0, 0);
        t_temp = Xf.block<3, 1>(3, 0);
        
        cost_ori = cost_func(line_, plane_, tarA_, curPt_, q_last * toQuaterniond(a_inc), q_last * t_inc + t_last);
        cost_aft = cost_func(line_, plane_, tarA_, curPt_, q_last * toQuaterniond(a_inc + a_temp), q_last * (t_inc + t_temp) + t_last);
        rho = (cost_ori - cost_aft) / pow((J * delta_x).norm(), 2);
        if (rho > 0.5)
        {
            a_inc += a_temp;
            t_inc += t_temp;
        }
        if (rho > 0.75)
            lambda *= std::max(1.0 / 3, 1 - pow(2 * rho - 1 ,3));
        if (rho < 0.25)
            lambda *= 2;
        if (lambda > 1e6 || lambda < 1e-6)
            break;
    }
    return true;
}

#endif // LM_HPP