#include "Optimizer.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

Optimizer::Optimizer() :
		map(NULL), noKeyFrames(0) {

}

void Optimizer::run() {

	while(1) {

		if(map->HasNewKF()) {

//			localMap = map->LocalMap();
//			globalMap = map->GlobalMap();
//
//			if(localMap.size() > 5)
//				LocalBA();
//
//			if(globalMap.size() % 100 == 0)
//				GlobalBA();
//
//			map->hasNewKFFlag = false;
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(3000));
	}
}

int Optimizer::OptimizePose(Frame * f, std::vector<Eigen::Vector3d> & points,
							std::vector<Eigen::Vector2d> & obs, Eigen::Matrix4d & dt) {

	g2o::SparseOptimizer optimizer;
	std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
	linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
	g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(
		g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
	);
	optimizer.setAlgorithm(solver);

	g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
	g2o::SE3Quat pose(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
	vSE3->setEstimate(pose);
	vSE3->setId(0);
	vSE3->setFixed(false);
	optimizer.addVertex(vSE3);

	const int N = f->N;
	const float delta = sqrt(7.815);
	std::vector<g2o::EdgeSE3ProjectXYZOnlyPose *> edges;

	for(int i = 0; i < N; ++i) {
		f->outliers[i] = false;
        g2o::EdgeSE3ProjectXYZOnlyPose * e = new g2o::EdgeSE3ProjectXYZOnlyPose();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(0)));
        e->setMeasurement(obs[i]);
        e->information() = Eigen::Matrix2d::Identity();
        g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
        e->setRobustKernel(rk);
        rk->setDelta(delta);

        e->fx = Frame::fx(0);
        e->fy = Frame::fy(0);
        e->cx = Frame::cx(0);
        e->cy = Frame::cy(0);
        e->Xw = points[i];

        optimizer.addEdge(e);
        edges.push_back(e);
	}

	const float chi2thresh[4] = { 10.815f, 10.815f, 10.815f, 10.815f };
	const int iteration[4] = { 10, 10, 10, 10 };
	int nBad = 0;
	for(size_t it = 0; it < 4; it++) {
		vSE3->setEstimate(g2o::SE3Quat(pose));
		optimizer.initializeOptimization(0);
		optimizer.optimize(iteration[it]);

		for(size_t i = 0, iend = edges.size(); i < iend; ++i) {
			g2o::EdgeSE3ProjectXYZOnlyPose * e = edges[i];
			if(f->outliers[i]) {
				e->computeError();
			}

			const float chi2 = e->chi2();
			if(chi2 > chi2thresh[it]) {
				f->outliers[i] = true;
				e->setLevel(1);
				nBad++;
			} else {
				e->setLevel(0);
				f->outliers[i] = false;
			}

			if(it == 2)
				e->setRobustKernel(0);
		}
	}

	g2o::VertexSE3Expmap * vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
	g2o::SE3Quat pose_recov = vSE3_recov->estimate();
	dt = pose_recov.to_homogeneous_matrix();

	return N - nBad;
}

void Optimizer::LocalBA() {

	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);

	std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
	linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
	g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(
		g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
	);

	optimizer.setAlgorithm(solver);
	int offset = localMap.size();

	for (int i = 0; i < offset; ++i) {
		KeyFrame * kf = localMap[i];
		g2o::SE3Quat pose(kf->Rotation().cast<double>(), kf->Translation().cast<double>());
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setId(i);
		vSE3->setEstimate(pose.inverse());
		if(i == 0)
			vSE3->setFixed(true);
		else
			vSE3->setFixed(false);

		optimizer.addVertex(vSE3);

		for (int j = 0; j < kf->N; ++j) {
			if (!kf->outliers[j] && kf->keyIndex[j] > -1) {

				g2o::EdgeSE3ProjectXYZOnlyPose * e = new g2o::EdgeSE3ProjectXYZOnlyPose();
				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vSE3));
				e->Xw = kf->mapPoints[j].cast<double>();
				e->information() = Eigen::Matrix2d::Identity();
				Eigen::Vector2d obs = Eigen::Vector2d::Identity();
				obs << kf->keyPoints[j].pt.x, kf->keyPoints[j].pt.y;
				e->setMeasurement(obs);
				g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);

				e->fx = Frame::fx(0);
				e->fy = Frame::fy(0);
				e->cx = Frame::cx(0);
				e->cy = Frame::cy(0);

				optimizer.addEdge(e);
			}
		}
	}

	optimizer.initializeOptimization();
	optimizer.setVerbose(false);
	optimizer.optimize(10);

	for(int i = 1; i < localMap.size(); ++i) {
		 g2o::VertexSE3Expmap * vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(i));
		 g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
		 Eigen::Matrix4d eigMat = SE3quat_recov.to_homogeneous_matrix();
		 localMap[i]->newPose = eigMat.inverse().cast<float>();
	}
}

void Optimizer::GlobalBA() {

	g2o::SparseOptimizer optimizer;
	optimizer.setVerbose(false);

	std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linearSolver;
	linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>>();
	g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(
		g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linearSolver))
	);

	optimizer.setAlgorithm(solver);
	int offset = globalMap.size();

	for (int i = 0; i < offset; ++i) {
		KeyFrame * kf = globalMap[i];
		g2o::SE3Quat pose(kf->Rotation().cast<double>(), kf->Translation().cast<double>());
		g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
		vSE3->setId(i);
		vSE3->setEstimate(pose.inverse());
		if(kf->frameId == 0)
			vSE3->setFixed(true);
		else
			vSE3->setFixed(false);

		optimizer.addVertex(vSE3);

		for (int j = 0; j < kf->N; ++j) {
			if (!kf->outliers[j] && kf->keyIndex[j] > -1) {

				g2o::EdgeSE3ProjectXYZOnlyPose * e = new g2o::EdgeSE3ProjectXYZOnlyPose();
				e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(vSE3));
				e->Xw = kf->mapPoints[j].cast<double>();
				e->information() = Eigen::Matrix2d::Identity();
				Eigen::Vector2d obs = Eigen::Vector2d::Identity();
				obs << kf->keyPoints[j].pt.x, kf->keyPoints[j].pt.y;
				e->setMeasurement(obs);
				g2o::RobustKernelHuber * rk = new g2o::RobustKernelHuber;
				e->setRobustKernel(rk);

				e->fx = Frame::fx(0);
				e->fy = Frame::fy(0);
				e->cx = Frame::cx(0);
				e->cy = Frame::cy(0);

				optimizer.addEdge(e);
			}
		}
	}

	optimizer.initializeOptimization();
	optimizer.setVerbose(false);
	optimizer.optimize(10);

	for(int i = 0; i < globalMap.size(); ++i) {
		 g2o::VertexSE3Expmap * vSE3_recov = static_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(i));
		 g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
		 Eigen::Matrix4d eigMat = SE3quat_recov.to_homogeneous_matrix();
		 globalMap[i]->newPose = eigMat.inverse().cast<float>();
	}
}

void Optimizer::SetMap(Mapping * map_) {

	map = map_;
}
