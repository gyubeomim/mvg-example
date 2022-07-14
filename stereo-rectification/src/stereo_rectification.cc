#include "util.h"

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace std;

Eigen::Matrix3d CalculateFundamentalMatrix(Eigen::MatrixXd x1,
                                           Eigen::MatrixXd x2) {
  Eigen::Matrix3d F;

  // compute H1norm.
  double x1_mean_x = x1.col(0).mean();
  double x1_mean_y = x1.col(1).mean();
  Eigen::Vector2d centroid;
  centroid << x1_mean_x, x1_mean_y;

  std::vector<double> dist;
  for (int i = 0; i < x1.rows(); i++) {
    double d = std::sqrt((x1(i, 0) - centroid(0)) * (x1(i, 0) - centroid(0)) +
                         (x1(i, 1) - centroid(1)) * (x1(i, 1) - centroid(1)));
    dist.push_back(d);
  }

  double mean_dist = std::accumulate(dist.begin(), dist.end(), 0) / dist.size();

  Eigen::Matrix3d H1norm;
    H1norm << std::sqrt(2) / mean_dist, 0, -std::sqrt(2) / mean_dist * centroid(0), 
	   	    0, std::sqrt(2) / mean_dist, -std::sqrt(2) / mean_dist * centroid(1), 
	   		0, 0, 1;

  // compute H2norm.
  double x2_mean_x = x2.col(0).mean();
  double x2_mean_y = x2.col(1).mean();
  Eigen::Vector2d centroid2;
  centroid2 << x2_mean_x, x2_mean_y;

  std::vector<double> dist2;
  for (int i = 0; i < x2.rows(); i++) {
    double d = std::sqrt((x2(i, 0) - centroid2(0)) * (x2(i, 0) - centroid2(0)) +
                         (x2(i, 1) - centroid2(1)) * (x2(i, 1) - centroid2(1)));
    dist2.push_back(d);
  }

  double mean_dist2 =
      std::accumulate(dist2.begin(), dist2.end(), 0) / dist2.size();

  Eigen::Matrix3d H2norm;
    H2norm << std::sqrt(2) / mean_dist2, 0, -std::sqrt(2) / mean_dist2 * centroid2(0), 
	  	    0, std::sqrt(2) / mean_dist2, -std::sqrt(2) / mean_dist2 * centroid2(1), 
	  		0, 0, 1;
	
  x1 = (H1norm * x1.transpose()).transpose();
  x2 = (H2norm * x2.transpose()).transpose();
	
  // estimate fundamental matrix.
  Eigen::MatrixXd A(8, 9);
  A.col(0) = x1.col(0).cwiseProduct(x2.col(0));
  A.col(1) = x1.col(1).cwiseProduct(x2.col(0));
  A.col(2) = x2.col(0);
  A.col(3) = x1.col(0).cwiseProduct(x2.col(1));
  A.col(4) = x1.col(1).cwiseProduct(x2.col(1));
  A.col(5) = x2.col(1);
  A.col(6) = x1.col(0);
  A.col(7) = x1.col(1);
  A.col(8) = Eigen::VectorXd::Ones(8);

  // first SVD to get F0
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU |
                                               Eigen::ComputeFullV);
  Eigen::MatrixXd V = svd.matrixV();
  Eigen::VectorXd f = V.col(V.cols() - 1);
  // Eigen::Matrix3d F0 = Eigen::Map<Eigen::Matrix3d>(f.data());
  Eigen::Matrix3d F0;
  F0 << f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8);

  // second SVD to get F_norm
  Eigen::JacobiSVD<Eigen::Matrix3d> svd2(F0, Eigen::ComputeFullU |
                                                 Eigen::ComputeFullV);
  Eigen::Matrix3d D = svd2.singularValues().asDiagonal();
  D(2, 2) = 0;
  Eigen::Matrix3d F_norm = svd2.matrixU() * D * svd2.matrixV().transpose();

  // get final fundamental matrix.
  F = H2norm.transpose() * F_norm * H1norm;

  return F;
}

void DrawEpipolarLines(cv::Mat img_left, cv::Mat img_right, Eigen::MatrixXd x1,
                       Eigen::MatrixXd x2, Eigen::Matrix3d F) {

  std::vector<cv::Scalar> colors;
  colors.push_back(cv::Scalar(255, 0, 0));
  colors.push_back(cv::Scalar(0, 255, 0));
  colors.push_back(cv::Scalar(0, 0, 255));
  colors.push_back(cv::Scalar(255, 255, 0));
  colors.push_back(cv::Scalar(255, 0, 255));
  colors.push_back(cv::Scalar(0, 255, 255));
  colors.push_back(cv::Scalar(125, 125, 0));
  colors.push_back(cv::Scalar(125, 0, 125));

  const int pt1 = -5000;
  const int pt2 = +5000;

  for (int i = 0; i < x1.rows(); i++) {
    Eigen::Vector3d x2_vec = x2.row(i);
    Eigen::Vector3d epipolar_line_left = F.transpose() * x2_vec;
    double a_l = epipolar_line_left(0);
    double b_l = epipolar_line_left(1);
    double c_l = epipolar_line_left(2);
    cv::Point2f pt_l1(pt1, (-a_l * pt1 - c_l) / b_l);
    cv::Point2f pt_l2(pt2, (-a_l * pt2 - c_l) / b_l);

    Eigen::Vector3d x1_vec = x1.row(i);
    Eigen::Vector3d epipolar_line_right = F * x1_vec;
    double a_r = epipolar_line_right(0);
    double b_r = epipolar_line_right(1);
    double c_r = epipolar_line_right(2);
    cv::Point2f pt_r1(pt1, (-a_r * pt1 - c_r) / b_r);
    cv::Point2f pt_r2(pt2, (-a_r * pt2 - c_r) / b_r);

#if 0
    std::cout << "Left" << std::endl;
    std::cout << a_l << ", " << b_l << ", " << c_l << std::endl;
    std::cout << pt_l1 << ", " << pt_l2 << std::endl;
    std::cout << "Right" << std::endl;
    std::cout << a_r << ", " << b_r << ", " << c_r << std::endl;
    std::cout << pt_r1 << ", " << pt_r2 << std::endl;
    std::cout << "-------------------------------------------------" << std::endl;
#endif
    cv::circle(img_left, cv::Point(x1_vec(0), x1_vec(1)), 20, colors[i],
               cv::FILLED, 8, 0);
    cv::circle(img_right, cv::Point(x2_vec(0), x2_vec(1)), 20, colors[i],
               cv::FILLED, 8, 0);
    cv::line(img_left, pt_l1, pt_l2, colors[i], 2);
    cv::line(img_right, pt_r1, pt_r2, colors[i], 2);
  }

  std::cout << "F\n" << F.matrix() << std::endl;

#if 1
  cv::resize(img_left, img_left, cv::Size(), 0.25, 0.25);
  cv::resize(img_right, img_right, cv::Size(), 0.25, 0.25);
  cv::imshow("epi_left", img_left);
  cv::imshow("epi_right", img_right);
  cv::moveWindow("epi_right", 1000, 0);
  cv::waitKey(1);
#endif
}



Eigen::MatrixXd Triangulation(Eigen::Matrix<double, 3, 4> P1,
                              Eigen::Matrix<double, 3, 4> P2,
                              Eigen::MatrixXd x1, Eigen::MatrixXd x2) {
  Eigen::MatrixXd X(1, 3);

  for (int i = 0; i < x1.rows(); i++) {
    Eigen::Matrix3d u_skew;
    u_skew << 0, -1, x1(i, 1), 1, 0, -x1(i, 0), -x1(i, 1), x1(i, 0), 0;
    Eigen::Matrix3d v_skew;
    v_skew << 0, -1, x2(i, 1), 1, 0, -x2(i, 0), -x2(i, 1), x2(i, 0), 0;

    Eigen::MatrixXd A(6, 4);
    A << u_skew * P1, v_skew * P2;

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU |
                                                 Eigen::ComputeFullV);
    Eigen::MatrixXd V = svd.matrixV();
    X.row(i) = V.block<3, 1>(0, V.cols() - 1) / V(V.rows() - 1, V.cols() - 1);

    if (i != x1.rows() - 1) {
      X.conservativeResize(X.rows() + 1, X.cols());
    }
  }

  return X;
}

std::tuple<Eigen::Matrix3d, Eigen::Vector3d>
DecomposeFundamentalMatrix(Eigen::Matrix3d K, Eigen::Matrix3d F,
                           Eigen::MatrixXd x1, Eigen::MatrixXd x2) {
  Eigen::Matrix3d E = K.transpose() * F * K;

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU |
                                               Eigen::ComputeFullV);

  Eigen::Matrix3d W;
  W << 0, -1, 0, 1, 0, 0, 0, 0, 1;
  Eigen::Vector3d t1 = svd.matrixU().col(2);
  Eigen::Matrix3d R1 = svd.matrixU() * W * svd.matrixV().transpose();
  if (R1.determinant() < 0) {
    t1 *= -1;
    R1 *= -1;
  }

  Eigen::Vector3d t2 = -svd.matrixU().col(2);
  Eigen::Matrix3d R2 = svd.matrixU() * W * svd.matrixV().transpose();
  if (R2.determinant() < 0) {
    t2 *= -1;
    R2 *= -1;
  }

  Eigen::Vector3d t3 = svd.matrixU().col(2);
  Eigen::Matrix3d R3 =
      svd.matrixU() * W.transpose() * svd.matrixV().transpose();
  if (R3.determinant() < 0) {
    t3 *= -1;
    R3 *= -1;
  }

  Eigen::Vector3d t4 = -svd.matrixU().col(2);
  Eigen::Matrix3d R4 =
      svd.matrixU() * W.transpose() * svd.matrixV().transpose();
  if (R4.determinant() < 0) {
    t4 *= -1;
    R4 *= -1;
  }

  // camera centers.
  Eigen::Vector3d C1 = -R1.transpose() * t1;
  Eigen::Vector3d C2 = -R2.transpose() * t2;
  Eigen::Vector3d C3 = -R3.transpose() * t3;
  Eigen::Vector3d C4 = -R4.transpose() * t4;

  // check four possible solutions.
  Eigen::Matrix<double, 3, 4> P1;
  P1 << K * Eigen::MatrixXd::Identity(3, 3),
      K * Eigen::VectorXd::Zero(3, 1);

  Eigen::Matrix<double, 3, 4> P2;
  P2 << K * R1 * Eigen::MatrixXd::Identity(3, 3), K * R1 * -C1;
  Eigen::MatrixXd X1 = Triangulation(P1, P2, x1, x2);

  P2 << K * R2 * Eigen::MatrixXd::Identity(3, 3), K * R2 * -C2;
  Eigen::MatrixXd X2 = Triangulation(P1, P2, x1, x2);

  P2 << K * R3 * Eigen::MatrixXd::Identity(3, 3), K * R3 * -C3;
  Eigen::MatrixXd X3 = Triangulation(P1, P2, x1, x2);

  P2 << K * R4 * Eigen::MatrixXd::Identity(3, 3), K * R4 * -C4;
  Eigen::MatrixXd X4 = Triangulation(P1, P2, x1, x2);

  std::vector<Eigen::Matrix3d> Rs;
  std::vector<Eigen::Vector3d> Cs;
  std::vector<Eigen::MatrixXd> Xs;

  Rs.push_back(R1);
  Rs.push_back(R2);
  Rs.push_back(R3);
  Rs.push_back(R4);

  Cs.push_back(C1);
  Cs.push_back(C2);
  Cs.push_back(C3);
  Cs.push_back(C4);

  Xs.push_back(X1);
  Xs.push_back(X2);
  Xs.push_back(X3);
  Xs.push_back(X4);

  // extract valid one solution.
  std::vector<int> valids(4);
  for (int i = 0; i < 4; i++) {
    int count = 0;

    for (int j = 0; j < Xs[i].rows(); j++) {
      Eigen::Vector3d Xs_vec = Xs[i].row(j);
      Eigen::VectorXd a = Rs[i].row(2) * (Xs_vec - Cs[i]);
      Eigen::VectorXd b = Xs[i].row(j).col(2);
      if (a.minCoeff() > 0 && b.minCoeff() > 0) {
        count += 1;
      }
    }
    valids[i] = count;
  }

  int best_idx =
      std::max_element(valids.begin(), valids.end()) - valids.begin();

  Eigen::Matrix3d R = Rs[best_idx];
  Eigen::Vector3d C = Cs[best_idx];

  return std::make_tuple(R, C);
}

std::tuple<Eigen::Matrix3d, Eigen::Matrix3d>
ComputeStereoHomography(Eigen::Matrix3d R, Eigen::Vector3d C,
                        Eigen::Matrix3d K) {
  Eigen::Vector3d rx, ry, rz, rz_tilde;

  rx = C / C.norm();
  rz_tilde << 0, 0, 1;

  Eigen::Vector3d tmp = rz_tilde - rz_tilde.dot(rx) * rx;
  rz = tmp / tmp.norm();
  ry = rz.cross(rx);

  Eigen::Matrix3d R_rect;
  R_rect << rx.transpose(), ry.transpose(), rz.transpose();

  Eigen::Matrix3d H1 = K * R_rect * K.inverse();
  Eigen::Matrix3d H2 = K * R_rect * R.transpose() * K.inverse();

  return std::make_tuple(H1, H2);
}

void ApplyHomography(cv::Mat &img, cv::Mat &img_out, Eigen::Matrix3d H) {
  cv::Mat _H;
  cv::eigen2cv(H, _H);
  cv::warpPerspective(img, img_out, _H, img.size());
}

int main() {
  std::string fn_left = util::GetProjectPath() + "/picture/left.jpg";
  std::string fn_right = util::GetProjectPath() + "/picture/right.jpg";

  cv::Mat img_left = cv::imread(fn_left, 1);
  cv::Mat img_right = cv::imread(fn_right, 1);

  Eigen::MatrixXd x1_best(8, 3);
  Eigen::MatrixXd x2_best(8, 3);

  x1_best <<
	    2201.606,  421.487, 1,
        2491.521, 1233.741, 1,
        2202.928, 294.5618, 1,
        1596.858, 621.5378, 1,
        2203.898,  1175.88, 1,
        2503.346,  1275.17, 1,
        2132.505, 298.8192, 1,
        2561.777, 745.0586, 1;

	x2_best <<
		 2404.31,  533.2378, 1,
         2403.97,  1370.911, 1,
        2450.759,  408.9355, 1,
        1749.202,  654.4105, 1,
        2140.022,  1267.529, 1,
        2399.647,   1413.43, 1,
         2380.94,  405.7941, 1,
        2653.428,  899.0267, 1;


  // load intrinsic parameter.
  Eigen::Matrix3d K;
  K << 2211.75963080077, 0, 2018.81623699895,
	   0,2218.36683671952, 1120.37532022008,
	   0, 0, 1;


  Eigen::Matrix3d F;
  F = CalculateFundamentalMatrix(x1_best, x2_best);

  DrawEpipolarLines(img_left, img_right, x1_best, x2_best, F);

  Eigen::Matrix3d R;
  Eigen::Vector3d C;

  std::tie(R, C) = DecomposeFundamentalMatrix(K, F, x1_best, x2_best);

  Eigen::Matrix3d H1;
  Eigen::Matrix3d H2;
  std::tie(H1, H2) = ComputeStereoHomography(R, C, K);

  cv::Mat img_left_warped, img_right_warped;
  ApplyHomography(img_left, img_left_warped, H1);
  ApplyHomography(img_right, img_right_warped, H2);
	
  cv::resize(img_left_warped, img_left_warped, cv::Size(), 0.25, 0.25);
  cv::resize(img_right_warped, img_right_warped, cv::Size(), 0.25, 0.25);
  cv::imshow("warped left", img_left_warped);
  cv::imshow("warped right", img_right_warped);
  cv::moveWindow("warped left", 0, 600);
  cv::moveWindow("warped right", 1000, 600);
  cv::waitKey(0);

  return 0;
}
