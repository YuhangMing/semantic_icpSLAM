#include <chrono>
#include <iostream>

#include "Solver.h"

bool Solver::PoseEstimate(std::vector<Eigen::Vector3d> & src,
						  std::vector<Eigen::Vector3d> & ref,
						  std::vector<bool> & outlier,
						  Eigen::Matrix4d & Tlastcurr,
						  int iteration, bool checkAngle) {

	Eigen::Matrix3d R_best = Eigen::Matrix3d::Identity();
	Eigen::Vector3d t_best = Eigen::Vector3d::Zero();
	int inliers_best = 0;
	int nMatches = src.size();

	auto now = std::chrono::system_clock::now();
	int seed = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()).count();
	srand(seed);

	int nIter = 0;
	int nBadSamples = 0;
	float ratio = 0.0f;
	float confidence = 0.0f;
	float thresh_inlier = 0.05f;
	const float thresh_confidence = 0.95f;
	const int minIter = 20;
	if(nMatches < 3)
		return false;

	while (nIter < iteration) {

		bool badSample = false;
		std::vector<int> samples;
		for (int i = 0; i < 3; ++i) {
			int s = rand() % nMatches;
			samples.push_back(s);
		}

		if (samples[0] == samples[1] ||
			samples[1] == samples[2] ||
			samples[2] == samples[0])
			badSample = true;

		Eigen::Vector3d src_a = src[samples[0]];
		Eigen::Vector3d src_b = src[samples[1]];
		Eigen::Vector3d src_c = src[samples[2]];

		Eigen::Vector3d ref_a = ref[samples[0]];
		Eigen::Vector3d ref_b = ref[samples[1]];
		Eigen::Vector3d ref_c = ref[samples[2]];

		float src_d = (src_b - src_a).cross(src_a - src_c).norm();
		float ref_d = (ref_b - ref_a).cross(ref_a - ref_c).norm();

		if (badSample || src_d < 1e-6 || ref_d < 1e-6) {
			nBadSamples++;
			nIter++;
			continue;
		}

		Eigen::Vector3d src_mean = (src_a + src_b + src_c) / 3;
		Eigen::Vector3d ref_mean = (ref_a + ref_b + ref_c) / 3;

		src_a -= src_mean;
		src_b -= src_mean;
		src_c -= src_mean;

		ref_a -= ref_mean;
		ref_b -= ref_mean;
		ref_c -= ref_mean;

		Eigen::Matrix3d Ab = Eigen::Matrix3d::Zero();
		Ab += src_a * ref_a.transpose();
		Ab += src_b * ref_b.transpose();
		Ab += src_c * ref_c.transpose();

		Eigen::JacobiSVD<Eigen::Matrix3d> svd(Ab, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::Matrix3d V = svd.matrixV();
		Eigen::Matrix3d U = svd.matrixU();
		Eigen::Matrix3d R = (V * U.transpose()).transpose();
		if(R.determinant() < 0)
			continue;
		Eigen::Vector3d t = src_mean - R * ref_mean;

		int nInliers = 0;
		outlier.resize(src.size());
		fill(outlier.begin(), outlier.end(), true);
		for (int i = 0; i < src.size(); ++i) {
			double d = (src[i] - (R * ref[i] + t)).norm();
			if (d <= thresh_inlier) {
				nInliers++;
				outlier[i] = false;
			}
		}

		if (nInliers > inliers_best) {

			Ab = Eigen::Matrix3d::Zero();
			src_mean = Eigen::Vector3d::Zero();
			ref_mean = Eigen::Vector3d::Zero();
			for (int i = 0; i < outlier.size(); ++i) {
				if (!outlier[i]) {
					src_mean += src[i];
					ref_mean += ref[i];
				}
			}

			src_mean /= nInliers;
			ref_mean /= nInliers;

			for (int i = 0; i < outlier.size(); ++i) {
				if (!outlier[i]) {
					Ab += (src[i] - src_mean) * (ref[i] - ref_mean).transpose();
				}
			}

			svd.compute(Ab, Eigen::ComputeFullU | Eigen::ComputeFullV);
			V = svd.matrixV();
			U = svd.matrixU();
			R_best = (V * U.transpose()).transpose();
			t_best = src_mean - R_best * ref_mean;
			inliers_best = nInliers;

			ratio = (float) nInliers / src.size();

			confidence = 1 - pow((1 - pow(ratio, 3)), nIter + 1);

			if (nIter >= minIter && confidence >= thresh_confidence)
				break;
		}

		nIter++;
	}

	Tlastcurr.topLeftCorner(3, 3) = R_best;
	Tlastcurr.topRightCorner(3, 1) = t_best;

	for (int i = 0; i < src.size(); ++i) {
		double d = (src[i] - (R_best * ref[i] + t_best)).norm();
		if (d <= thresh_inlier) {
			outlier[i] = false;
		}
	}

	int N = 0;
	Eigen::Vector3d src_mean = Eigen::Vector3d::Zero();
	Eigen::Vector3d dst_mean = Eigen::Vector3d::Zero();
	for (int i = 0; i < src.size(); ++i) {
		if(!outlier[i]) {
			src_mean += src[i];
			dst_mean += ref[i];
			N++;
		}
	}

	src_mean /= N;
	dst_mean /= N;

	Eigen::Matrix3d Ab = Eigen::Matrix3d::Zero();
	for (int i = 0; i < src.size(); ++i) {
		if(!outlier[i]) {
			Ab += (src[i] - src_mean) * (ref[i] - dst_mean).transpose();
		}
	}

	Eigen::JacobiSVD<Eigen::Matrix3d> svd(Ab, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d V = svd.matrixV();
	Eigen::Matrix3d U = svd.matrixU();
	R_best = (V * U.transpose()).transpose();
	if(R_best.determinant() < 0) {
		std::cout << "final check failed." << std::endl;
		return false;
	}

	t_best = src_mean - R_best * dst_mean;
	inliers_best = N;

	if(checkAngle && confidence < 0.8) {
		Eigen::Vector3d angles = R_best.eulerAngles(0, 1, 2).array().sin();
		if (angles.norm() >= 0.2 || t_best.norm() >= 0.1)
			return false;
	}

	return true;
}
