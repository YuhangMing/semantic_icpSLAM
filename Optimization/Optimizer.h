#ifndef OPTIMIZER_H__
#define OPTIMIZER_H__

#include "Mapping.h"

class Optimizer {

public:

	const int NUM_LOCAL_KF = 7;

	Optimizer();

	void run();

	void LocalBA();

	void GlobalBA();

	void GetLocalMap();

	static int OptimizePose(Frame * f, std::vector<Eigen::Vector3d> & points,
			std::vector<Eigen::Vector2d> & obs, Eigen::Matrix4d & dt);

	void SetMap(Mapping * map_);

protected:

	Mapping * map;

	size_t noKeyFrames;

	std::vector<KeyFrame *> localMap;

	std::vector<KeyFrame *> globalMap;
};

#endif
