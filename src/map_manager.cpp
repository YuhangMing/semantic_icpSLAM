#include "map_manager.h"

namespace fusion
{

SubMapManager::SubMapManager() : bKFCreated(false) {
	ResetSubmaps();
}

void SubMapManager::Create(const fusion::IntrinsicMatrix base, 
						   int submapIdx, bool bTrack, bool bRender)
{
	std::cout << "Create submap no. " << submapIdx << std::endl;

	auto submap = std::make_shared<DenseMapping>(base, submapIdx, bTrack, bRender);
	submap->poseGlobal = Sophus::SE3d();	// set to identity
	active_submaps.push_back(submap);

	bHasNewSM = false;
	renderIdx = submapIdx;
	ref_frame_id = 0;
}

void SubMapManager::Create(const fusion::IntrinsicMatrix base, 
						   int submapIdx, RgbdImagePtr ref_img, bool bTrack, bool bRender)
{
	std::cout << "Create submap no. " << submapIdx << std::endl;

	auto ref_frame = ref_img->get_reference_frame();
	// create new submap
	auto submap = std::make_shared<DenseMapping>(base, submapIdx, bTrack, bRender);
	submap->poseGlobal = active_submaps[renderIdx]->poseGlobal * ref_frame->pose;
	// store new submap
	active_submaps.push_back(submap);
	// stop previous rendering submap from fusing depth info
	active_submaps[renderIdx]->bRender = false;

	// create new model frame for tracking and rendering
	auto model_i = std::make_shared<DeviceImage>(base, ref_img->NUM_PYRS);
	copyDeviceImage(ref_img, model_i);
	auto model_f = model_i->get_reference_frame();	// new frame created when perform copy above, new pointer here
	model_f->pose = Sophus::SE3d();	// every new submap starts its own reference coordinate system
	odometry->vModelFrames.push_back(model_f);
	odometry->vModelDeviceMapPyramid.push_back(model_i);

	// some other parameters
	bHasNewSM = true;
	renderIdx = active_submaps.size()-1;
	ref_frame_id = ref_frame->id;
}

void SubMapManager::ResetSubmaps(){
	for(size_t i=0; i < active_submaps.size(); i++){
		active_submaps[i]->reset_mapping();
	}
	for(size_t i=0; i < passive_submaps.size(); i++){
		passive_submaps[i]->reset_mapping();
	}

	// submap storage
	active_submaps.clear();
	passive_submaps.clear();
	activeTOpassiveIdx.clear();
}

float SubMapManager::CheckVisPercent(int submapIdx){
	return active_submaps[submapIdx]->CheckVisPercent();
}

void SubMapManager::CheckActive(){
	for(size_t i=0; i < activeTOpassiveIdx.size(); i++){
		// std::cout << "Removing submap " << activeTOpassiveIdx[i] << std::endl;
		auto tmp_map = active_submaps[activeTOpassiveIdx[i]];

		// // store map points and normal in manager
		// mPassiveMPs.insert({tmp_map->submapIdx, tmp_map->vMapPoints});
		// mPassiveNs.insert({tmp_map->submapIdx, tmp_map->vMapNormals});
		// // Store all passive submaps in RAM, save memories on GPU
		// tmp_map->DownloadToRAM();
		// tmp_map->Release();

		passive_submaps.push_back(tmp_map);
		active_submaps.erase(active_submaps.begin() + activeTOpassiveIdx[i]);

		// delete corresponding model frame
		auto tmp_image = odometry->vModelDeviceMapPyramid[activeTOpassiveIdx[i]];
		auto tmp_frame = odometry->vModelFrames[activeTOpassiveIdx[i]];

		odometry->vModelDeviceMapPyramid.erase(odometry->vModelDeviceMapPyramid.begin() + activeTOpassiveIdx[i]);
		odometry->vModelFrames.erase(odometry->vModelFrames.begin() + activeTOpassiveIdx[i]);
		renderIdx--;
	}
	activeTOpassiveIdx.clear();
}

void SubMapManager::CheckTrackAndRender(int cur_frame_id, int max_perct_idx){
	if(bHasNewSM)
	{
		if(cur_frame_id - ref_frame_id >= 17){
			active_submaps[renderIdx]->bTrack = true;
			bHasNewSM = false;
			std::cout << "Start tracking on the new submap (" 
					  << ref_frame_id << "-" << cur_frame_id
					  << ")" << std::endl;
		}
	}
	else
	{
		// currently disabled !!!!!!!!!!!!!!!!!!!!!!!!!!
		// if(max_perct_idx != renderIdx){
		// 	active_submaps[renderIdx]->bRender = false;
		//  active_submaps[renderIdx]->bTrack = false;
		// 	active_submaps[max_perct_idx]->bRender = true;
		// 	active_submaps[max_perct_idx]->bTrack = true;
		// 	renderIdx = max_perct_idx;
		// }
	}
}

void SubMapManager::AddKeyFrame(RgbdFramePtr currKF){
	// extract key points from current kf
	cv::Mat source_image = currKF->image;
    auto frame_pose = currKF->pose.cast<float>();

    cv::Mat raw_descriptors;
    std::vector<cv::KeyPoint> raw_keypoints;
    extractor->extract_features_surf(
        source_image,
        raw_keypoints,
        raw_descriptors);
    std::cout << "# of raw keypoints is " << raw_keypoints.size() << std::endl;

    extractor->compute_3d_points(
        currKF->vmap,
        currKF->nmap,
        raw_keypoints,
        raw_descriptors,
        currKF->cv_key_points,
        currKF->descriptors,
        currKF->key_points,
        frame_pose);

	// copy a version to store
	auto kf = std::make_shared<RgbdFrame>();
    currKF->copyTo(kf);
    active_submaps[renderIdx]->vKFs.push_back(kf);
    std::cout << "# of 3d keypoints is " << kf->key_points.size() << std::endl;

}

std::vector<Eigen::Matrix<float, 4, 4>> SubMapManager::GetKFPoses(){
	std::vector<Eigen::Matrix<float, 4, 4>> poses;
	Eigen::Matrix4f Tw2rfinv = active_submaps[renderIdx]->poseGlobal.cast<float>().matrix().inverse();
    
    // actives
    for (size_t i=0; i<active_submaps.size(); ++i)
    {
    	Eigen::Matrix4f Twm = active_submaps[i]->poseGlobal.cast<float>().matrix();
    	for(size_t j=0; j<active_submaps[i]->vKFs.size(); ++j){
    		Eigen::Matrix4f Tmf = active_submaps[i]->vKFs[j]->pose.cast<float>().matrix();
	        Eigen::Matrix4f pose = Tw2rfinv * Twm * Tmf;
	        poses.emplace_back(pose);	
    	}
    }

    // passives
    for (size_t i=0; i<passive_submaps.size(); ++i)
    {
    	Eigen::Matrix4f Twm = passive_submaps[i]->poseGlobal.cast<float>().matrix();
    	for(size_t j=0; j<passive_submaps[i]->vKFs.size(); ++j){
    		Eigen::Matrix4f Tmf = passive_submaps[i]->vKFs[j]->pose.cast<float>().matrix();
	        Eigen::Matrix4f pose = Tw2rfinv * Twm * Tmf;
	        poses.emplace_back(pose);	
    	}
    }

    return poses;
}

void SubMapManager::GetPoints(float *pt3d, size_t &count, size_t max_size){
	count = 0;
	Sophus::SE3f Tw2rfinv = active_submaps[renderIdx]->poseGlobal.cast<float>().inverse();
	
	// actives
	for(size_t i=0; i<active_submaps.size(); ++i)
	{
		auto sm = active_submaps[i];
		Sophus::SE3f Twm = sm->poseGlobal.cast<float>();
		for(size_t j=0; j<sm->vKFs.size(); ++j)
		{
			auto kf = sm->vKFs[j];
			for(size_t k=0; k<kf->key_points.size(); ++k)
			{
				auto pt = kf->key_points[k];
				Eigen::Vector3f mp_global_renderSM = Tw2rfinv * Twm * pt->pos;
				pt3d[count * 3 + 0] = mp_global_renderSM(0);
	            pt3d[count * 3 + 1] = mp_global_renderSM(1);
	            pt3d[count * 3 + 2] = mp_global_renderSM(2);
	            count++;
			}  // pts
		}  // kfs
    }  // sms

    // passives
    for(size_t i=0; i<passive_submaps.size(); ++i)
	{
		auto sm = passive_submaps[i];
		Sophus::SE3f Twm = sm->poseGlobal.cast<float>();
		for(size_t j=0; j<sm->vKFs.size(); ++j)
		{
			auto kf = sm->vKFs[j];
			for(size_t k=0; k<kf->key_points.size(); ++k)
			{
				auto pt = kf->key_points[k];
				Eigen::Vector3f mp_global_renderSM = Tw2rfinv * Twm * pt->pos;
				pt3d[count * 3 + 0] = mp_global_renderSM(0);
	            pt3d[count * 3 + 1] = mp_global_renderSM(1);
	            pt3d[count * 3 + 2] = mp_global_renderSM(2);
	            count++;
			}  // pts
		}  // kfs
    }  // sms

    // std::cout << "NUM KEY POINTS: " << count << std::endl;
}

void SubMapManager::SetTracker(std::shared_ptr<DenseOdometry> pOdometry){
	odometry = pOdometry;
}

void SubMapManager::SetExtractor(std::shared_ptr<FeatureExtractor> pExtractor){
	extractor = pExtractor;
}

} // namespace fusion