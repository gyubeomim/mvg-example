#include "util.h"

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>

void ApplyHomography(cv::Mat &img, cv::Mat &img_out, Eigen::Matrix3d H) {
  cv::Mat _H;
  cv::eigen2cv(H, _H);
  cv::warpPerspective(img, img_out, _H,
                      cv::Size(img.size().width * 4, img.size().height * 1));
}

int main() {
  std::string fn_img =
      util::GetProjectPath() + "/picture/affine_metric_rectification/01.png";
  cv::Mat img = cv::imread(fn_img, 1);

  Eigen::Matrix<double, 8, 3> line_points;

  line_points << 
	  113, 5, 1, 
	  223, 846, 1, 
	  435, 2, 1, 
	  707, 843, 1, 
	  2, 706, 1, 
	  841, 780, 1, 
	  4, 6, 1, 
	  841, 445, 1;

  Eigen::Matrix<double, 4, 3> lines;
  for (int i = 0; i < 8; i += 2) {
    Eigen::Vector3d v1, v2;
    v1 = line_points.row(i);
    v2 = line_points.row(i + 1);

    lines.row(i / 2) = v1.cross(v2);
  }

  // Affine Rectification-------------------
  // intersection of line #1 and #2.
  Eigen::Matrix<double, 2, 3> A;
  A << lines.row(0), lines.row(1);
  Eigen::FullPivLU<Eigen::MatrixXd> lu(A);
  Eigen::Vector3d nullspace1 = lu.kernel();
  nullspace1 = nullspace1 / nullspace1(2);

  // intersection of line #3 and #4.
  Eigen::Matrix<double, 2, 3> A2;
  A2 << lines.row(2), lines.row(3);
  Eigen::FullPivLU<Eigen::MatrixXd> lu2(A2);
  Eigen::Vector3d nullspace2 = lu2.kernel();
  nullspace2 = nullspace2 / nullspace2(2);

  // image of line at infinity.
  Eigen::Matrix<double, 2, 3> A3;
  A3 << nullspace1.transpose(), nullspace2.transpose();
  Eigen::FullPivLU<Eigen::MatrixXd> lu3(A3);
  Eigen::Vector3d image_of_line_at_inf = lu3.kernel();

  Eigen::Matrix3d H_ar;
  H_ar << 1, 0, 0, 0, 1, 0, image_of_line_at_inf.transpose();

  // affine rectification.
  cv::Mat img_aff;
  ApplyHomography(img, img_aff, H_ar);
  cv::resize(img_aff, img_aff, cv::Size(), 0.25, 0.25);
  cv::imshow("affine", img_aff);
  cv::waitKey(1);

  // Metric Rectification-------------------
  lines.row(0) /= lines.row(0)(2);
  lines.row(1) /= lines.row(1)(2);
  lines.row(2) /= lines.row(2)(2);
  lines.row(3) /= lines.row(3)(2);

  // remove last colmun.
  Eigen::Vector3d l1 = lines.row(0);
  Eigen::Vector3d l2 = lines.row(1);
  Eigen::Vector3d m1 = lines.row(3);
  Eigen::Vector3d m2 = lines.row(2);

  Eigen::Matrix<double, 2, 3> ortho_constraint;
  ortho_constraint << 
	  l1(0) * m1(0), 
	  l1(0) * m1(1) + l1(1) * m1(0),
      l1(1) * m1(1), 

	  l2(0) * m2(0), 
	  l2(0) * m2(1) + l2(1) * m2(0),
      l2(1) * m2(1);

  Eigen::JacobiSVD<Eigen::Matrix<double, 2, 3>> svd(
      ortho_constraint, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d a = svd.matrixV().col(svd.matrixV().cols() - 1);
  Eigen::Matrix2d A_;
  A_ << a(0), a(1),
	a(1), a(2);

  Eigen::JacobiSVD<Eigen::Matrix2d> svd2(A_, Eigen::ComputeFullU |
                                                Eigen::ComputeFullV);
  Eigen::MatrixXd U = svd2.matrixU();
  Eigen::MatrixXd D = svd2.singularValues().asDiagonal();
  Eigen::Matrix2d K = U * D.cwiseSqrt() * U.transpose();

  Eigen::Matrix3d H_mr;
  H_mr << 
	  K(0, 0), K(0, 1), 0, 
	  K(1, 0), K(1, 1), 0, 
	  0, 0, 1;

  // metric rectification.
  cv::Mat img_metric;
  ApplyHomography(img, img_metric, H_mr.inverse() * H_ar);
  cv::resize(img_metric, img_metric, cv::Size(), 0.25, 0.25);
  cv::imshow("metric", img_metric);
  cv::moveWindow("metric", 1000, 0);
  cv::waitKey(0);

  return 0;
}
