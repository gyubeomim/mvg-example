#include "util.h"

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <numeric>

using namespace std;

Eigen::Matrix3d ComputeHomographyDLT(Eigen::MatrixXd x1, Eigen::MatrixXd x2){
	double x1_mean_x  = x1.col(0).mean();
	double x1_mean_y  = x1.col(1).mean();
	Eigen::Vector2d centroid1;
	centroid1 << x1_mean_x, x1_mean_y;

	double x2_mean_x  = x2.col(0).mean();
	double x2_mean_y  = x2.col(1).mean();
	Eigen::Vector2d centroid2;
	centroid2 << x2_mean_x, x2_mean_y;
		
	// compute H1norm
	vector<double> dist1;
	for(int i=0; i<x1.rows(); i++){
		double d = std::sqrt((x1(i, 0) - centroid1(0)) * (x1(i, 0) - centroid1(0)) +
							 (x1(i, 1) - centroid1(1)) * (x1(i, 1) - centroid1(1)));
		dist1.push_back(d);
	}

	double mean_dist1 = std::accumulate(dist1.begin(), dist1.end(), 0) / dist1.size();

	Eigen::Matrix3d H1norm;
	H1norm << 1/std::sqrt(2) / mean_dist1, 0, -centroid1(0) / std::sqrt(2) / mean_dist1, 
		   0, 1/std::sqrt(2) / mean_dist1, -centroid1(1) / std::sqrt(2) / mean_dist1, 
		   0, 0, 1;

	// compute H2norm
	vector<double> dist2;
	for(int i=0; i<x2.rows(); i++){
		double d = std::sqrt((x2(i, 0) - centroid2(0)) * (x2(i, 0) - centroid2(0)) +
							 (x2(i, 1) - centroid2(1)) * (x2(i, 1) - centroid2(1)));
		dist2.push_back(d);
	}

	double mean_dist2 = std::accumulate(dist2.begin(), dist2.end(), 0) / dist2.size();

	Eigen::Matrix3d H2norm;
	H2norm << 1/std::sqrt(2) / mean_dist2, 0, -centroid2(0) / std::sqrt(2) / mean_dist2, 
		   0, 1/std::sqrt(2) / mean_dist2, -centroid2(1) / std::sqrt(2) / mean_dist2, 
		   0, 0, 1;

	x1 = (H1norm * x1.transpose()).transpose();
	x2 = (H2norm * x2.transpose()).transpose();

	Eigen::MatrixXd A(8, 9);
	for(int i=0; i<4; i++){
		double x = x1.row(i)(0);
		double y = x1.row(i)(1);
		double u = x2.row(i)(0);
		double v = x2.row(i)(1);

		Eigen::Matrix<double, 9, 1> a1,a2;
		a1 << -x, -y, -1, 0, 0, 0, u*x, u*y, u;
		a2 << 0, 0, 0, -x, -y, -1, v*x, v*y, v;


		int k = 2*i;
		A.row(k) = a1.transpose();
		A.row(k+1) = a2.transpose();
	}

	Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXd V = svd.matrixV();
	Eigen::Matrix<double, 9 , 1> h = V.col(V.cols() - 1);

	Eigen::Matrix3d H;
	H << h(0), h(1), h(2), 
	  h(3), h(4), h(5), 
	  h(6), h(7), h(8);


	return H2norm.inverse() * H * H1norm;
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
  cv::resize(img_left, img_left, cv::Size(), 0.5, 0.5);
  cv::resize(img_right, img_right, cv::Size(), 0.5, 0.5);
  cv::imshow("epi_left", img_left);
  cv::imshow("epi_right", img_right);
  cv::moveWindow("epi_right", 1000, 0);
  cv::waitKey(0);
#endif
}



int main(){
	std::string fn_left =
		util::GetProjectPath() + "/picture/left.jpg";
	std::string fn_right =
		util::GetProjectPath() + "/picture/right.jpg";

	cv::Mat img_left = cv::imread(fn_left, 1);
	cv::Mat img_right = cv::imread(fn_right, 1);


	Eigen::Matrix<double, 4, 3> x_left, x_right;
	x_left << 966, 411, 1, 
		   1206, 352, 1, 
		   1219, 618, 1, 
		   1014, 625, 1;
	x_right << 711, 445, 1, 
			901, 461, 1, 
			957, 728, 1, 
			776, 649, 1;

	Eigen::Matrix3d H;

	H = ComputeHomographyDLT(x_left, x_right);

	Eigen::Vector3d x5_left, x5_right, x6_left, x6_right;
	x5_left << 889, 360, 1;
	x5_right << 386, 345, 1;
	x6_left << 1567, 487, 1;
	x6_right << 1554, 810, 1;

	Eigen::Vector3d Hx5_left = H * x5_left;
	Hx5_left = Hx5_left / Hx5_left.norm();

	Eigen::Vector3d Hx6_left = H * x6_left;
	Hx6_left = Hx6_left / Hx6_left.norm();

	Eigen::Vector3d l5, l6;

	l5 = Hx5_left.cross(x5_right);
	l6 = Hx6_left.cross(x6_right);
	
	Eigen::Vector3d e_r;
	e_r = l5.cross(l6);
	
	Eigen::Matrix3d skew_e_r;
	skew_e_r << 0, -e_r(2), e_r(1),
			 e_r(2), 0, -e_r(0),
			 -e_r(1), e_r(0), 0;

	Eigen::Matrix3d F;
	F = skew_e_r * H;
	
	DrawEpipolarLines(img_left, img_right, x_left, x_right, F);

}
