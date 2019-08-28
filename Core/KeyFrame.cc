#include "KeyFrame.h"

KeyFrame::KeyFrame() {
	N = 0;
	frameId = 0;
}

KeyFrame::KeyFrame(const Frame * f) {

	N = f->N;
	frameId = f->frameId;
	f->descriptors.copyTo(descriptors);
	mapPoints = f->mapPoints;
	keyPoints = f->keyPoints;
	pointNormal = f->pointNormal;
	pose = f->pose.cast<float>();
	observations.resize(mapPoints.size());
	std::fill(observations.begin(), observations.end(), 0);

	keyIndex.resize(mapPoints.size());
	std::fill(keyIndex.begin(), keyIndex.end(), -1);
}

Matrix3f KeyFrame::GpuRotation() const {
	Matrix3f Rot;
	Rot.rowx = make_float3(pose(0, 0), pose(0, 1), pose(0, 2));
	Rot.rowy = make_float3(pose(1, 0), pose(1, 1), pose(1, 2));
	Rot.rowz = make_float3(pose(2, 0), pose(2, 1), pose(2, 2));
	return Rot;
}

Matrix3f KeyFrame::GpuInvRotation() const {
	Matrix3f Rot;
	const Eigen::Matrix3f mPoseInv = Rotation().transpose();
	Rot.rowx = make_float3(mPoseInv(0, 0), mPoseInv(0, 1), mPoseInv(0, 2));
	Rot.rowy = make_float3(mPoseInv(1, 0), mPoseInv(1, 1), mPoseInv(1, 2));
	Rot.rowz = make_float3(mPoseInv(2, 0), mPoseInv(2, 1), mPoseInv(2, 2));
	return Rot;
}

float3 KeyFrame::GpuTranslation() const {
	return make_float3(pose(0, 3), pose(1, 3), pose(2, 3));
}

Eigen::Matrix3f KeyFrame::Rotation() const {
	return pose.topLeftCorner(3, 3);
}

Eigen::Vector3f KeyFrame::Translation() const {
	return pose.topRightCorner(3, 1);
}

Eigen::Vector3f KeyFrame::GetWorldPoint(int i) const {
	Eigen::Matrix3f r = Rotation();
	Eigen::Vector3f t = Translation();
	return r * mapPoints[i] + t;
}
