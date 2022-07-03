#include "util.h"

#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>

void ApplyHomography(cv::Mat &img, cv::Mat &img_out, Eigen::Matrix3d H) {
  cv::Mat _H;
  cv::eigen2cv(H, _H);

  cv::warpPerspective(img, img_out, _H,
                      cv::Size(img.size().width * 4, img.size().height * 2));
}

int main() {
  std::string fn_img = util::GetProjectPath() +
                       "/picture/test.png";
  cv::Mat img = cv::imread(fn_img, cv::IMREAD_COLOR);

  // four points.
  Eigen::Vector3d p1 = Eigen::Vector3d(620, 30, 1);
  Eigen::Vector3d p2 = Eigen::Vector3d(2144, 795, 1);
  Eigen::Vector3d p3 = Eigen::Vector3d(520, 1964, 1);
  Eigen::Vector3d p4 = Eigen::Vector3d(2248, 1834, 1);

  // four lines.
  Eigen::Matrix<double, 4, 3> lines;
  lines.row(0) = p1.cross(p2);
  lines.row(1) = p3.cross(p4);
  lines.row(2) = p1.cross(p3);
  lines.row(3) = p2.cross(p4);

  Eigen::JacobiSVD<Eigen::Matrix<double, 2, 3>> svd1(
      lines.block<2, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d V = svd1.matrixV();

  // vanishing point b/w line #1 and #2.
  Eigen::Vector3d vp1 = V.col(V.cols() - 1);
  vp1 = vp1 / vp1(2);

  Eigen::JacobiSVD<Eigen::Matrix<double, 2, 3>> svd2(
      lines.block<2, 3>(2, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d V2 = svd2.matrixV();

  // vanishing point b/w line #3 and #4.
  Eigen::Vector3d vp2 = V2.col(V2.cols() - 1);
  vp2 = vp2 / vp2(2);

  Eigen::Matrix<double, 2, 3> vpoints;
  vpoints << vp1.transpose(), vp2.transpose();

  Eigen::JacobiSVD<Eigen::Matrix<double, 2, 3>> svd3(
      vpoints, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d V3 = svd3.matrixV();

  // vanishnig line of vp1 and vp2.
  Eigen::Vector3d vline = V3.col(V3.cols() - 1);
  vline = vline / vline(2);

  // get affine homography.
  Eigen::Matrix3d H_aff;
  H_aff << 1, 0, 0, 0, 1, 0, vline.transpose();

  Eigen::Vector3d Hv1 = H_aff * vp1;
  Hv1 = Hv1 / Hv1.norm();

  Eigen::Vector3d Hv2 = H_aff * vp2;
  Hv2 = Hv2 / Hv2.norm();

  // find directions corresponding to vanishing points.
  Eigen::Matrix<double, 2, 4> dir;
  dir << Hv1(0), -Hv1(0), Hv2(0), -Hv2(0), Hv1(1), -Hv1(1), Hv2(1), -Hv2(1);

  std::vector<double> thetas;
  thetas.push_back(std::abs(std::atan2(dir(0, 0), dir(1, 0))));
  thetas.push_back(std::abs(std::atan2(dir(0, 1), dir(1, 1))));
  thetas.push_back(std::abs(std::atan2(dir(0, 2), dir(1, 2))));
  thetas.push_back(std::abs(std::atan2(dir(0, 3), dir(1, 3))));
  std::vector<double> thetas2;
  thetas2.push_back(std::atan2(dir(0, 2), dir(1, 2)));
  thetas2.push_back(std::atan2(dir(0, 3), dir(1, 3)));

  // find direction closest to vertical axis.
  int hidx = std::min_element(thetas.begin(), thetas.end()) - thetas.begin();
  int vidx = std::max_element(thetas2.begin(), thetas2.end()) - thetas2.begin();

  if(hidx <=2)
	  vidx += 2;

  Eigen::Matrix3d H_metric;
  H_metric << dir(0, vidx), dir(0, hidx), 0, dir(1, vidx), dir(1, hidx), 0, 0,
      0, 1;

  if (H_metric.determinant() < 0) {
    H_metric.row(0) *= -1;
  }
	

  // affine rectification.
  cv::Mat img_aff;
  ApplyHomography(img, img_aff, H_aff);
  cv::resize(img_aff, img_aff, cv::Size(), 0.10, 0.10);
  cv::imshow("affine", img_aff);
  cv::moveWindow("affine", 0, 0);

  // metric rectification.
  cv::Mat img_metric;
  ApplyHomography(img, img_metric, H_metric.inverse() * H_aff);
  cv::resize(img_metric, img_metric, cv::Size(), 0.10, 0.10);
  cv::imshow("metric", img_metric);
  cv::moveWindow("metric", 0, 500);
  cv::waitKey(0);

  return 0;
}
