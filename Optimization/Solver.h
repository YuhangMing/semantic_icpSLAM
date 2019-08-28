#ifndef SOLVER_HPP__
#define SOLVER_HPP__

#include "Frame.h"

#include <vector>
#include <Eigen/Dense>

class Solver {
public:
	static bool PoseEstimate(std::vector<Eigen::Vector3d> & src,
			std::vector<Eigen::Vector3d> & ref, std::vector<bool> & outliers,
			Eigen::Matrix4d& T, int iteration, bool checkAngle = false);
};

#endif
