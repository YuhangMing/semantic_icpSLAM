#include "system.h"

#include "cuda_runtime.h"
#include <ctime>
#define CUDA_MEM
// #define TIMING

namespace fusion
{

System::~System()
{
    graph->terminate();
    graphThread.join();
}

System::System(const fusion::IntrinsicMatrix base, const int NUM_PYR)
    : frame_id(0), is_initialized(false), hasNewKeyFrame(false)
{
    
    // mapping = std::make_shared<DenseMapping>(base);
    odometry = std::make_shared<DenseOdometry>(base, NUM_PYR);
    extractor = std::make_shared<FeatureExtractor>();
    matcher = std::make_shared<DescriptorMatcher>();

    #ifdef CUDA_MEM
        // inaccurate, the driver decides when to release the memory
        size_t free_t0, total_t0;
        float free_1, free_2, total_0, used_0;
        cudaMemGetInfo(&free_t0, &total_t0);
        total_0 = (uint)total_t0/1048576.0 ;
        free_1 = (uint)free_t0/1048576.0 ;
    #endif
    manager = std::make_shared<SubMapManager>();
    manager->Create(base, 0, true, true);
    manager->SetTracker(odometry);
    manager->SetExtractor(extractor);
    odometry->SetManager(manager);
    #ifdef CUDA_MEM
        cudaMemGetInfo(&free_t0, &total_t0);
        free_2 = (uint)free_t0/1048576.0 ;
        used_0 = free_1 - free_2;
        std::cout << "## Create a new submap used " << used_0 << " MB memory." << std::endl
                << "   with " 
                << free_1 << " MB free mem before, "
                << free_2 << " MB free mem after" << std::endl 
                << "   out of " << total_0 << " MB total memroy." << std::endl;
    #endif

    graph = std::make_shared<KeyFrameGraph>(base, NUM_PYR);
    graph->set_feature_extractor(extractor);
    graph->set_descriptor_matcher(matcher);

    relocalizer = std::make_shared<Relocalizer>(base);
    relocalizer->setDescriptorMatcher(matcher);
    relocalizer->setFeatureExtractor(extractor);

    #ifdef CUDA_MEM
        size_t free_t, total_t;
        float free_m1, free_m2, free_m3, total_m, used_m;
        cudaMemGetInfo(&free_t, &total_t);
        total_m = (uint)total_t/1048576.0 ;
        free_m1 = (uint)free_t/1048576.0 ;

        std::clock_t start = std::clock();
    #endif  
    detector = new semantic::MaskRCNN("bridge");
    #ifdef CUDA_MEM
        cudaMemGetInfo(&free_t, &total_t);
        free_m2 = (uint)free_t/1048576.0 ;
        used_m = free_m1 - free_m2;
        std::cout << "## Initialize the detector used " << used_m << " MB memory." << std::endl
                << "   with " 
                << free_m1 << " MB free mem before, "
                << free_m2 << " MB free mem after" << std::endl 
                << "   out of " << total_m << " MB total memroy." << std::endl;
        std::cout << "   and takes "
                  << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
                  << " seconds" << std::endl;
        start = std::clock();
    #endif
    detector->initializeDetector("/home/lk18493/github/maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml", 0);
    #ifdef CUDA_MEM 
        cudaMemGetInfo(&free_t, &total_t);
        free_m3 = (uint)free_t/1048576.0 ;
        used_m = free_m2 - free_m3;
        std::cout << "## Load the model used " << used_m << " MB memory." << std::endl
                << "   with " 
                << free_m2 << " MB free mem before, "
                << free_m3 << " MB free mem after" << std::endl 
                << "   out of " << total_m << " MB total memroy." << std::endl;
        std::cout << "   and takes "
                  << ( std::clock() - start ) / (double) CLOCKS_PER_SEC 
                  << " seconds" << std::endl;
    #endif
    // odometry->SetDetector(detector);

    graphThread = std::thread(&KeyFrameGraph::main_loop, graph.get());
}

void System::initialization()
{
    is_initialized = true;
    current_frame->pose = initialPose;
    current_keyframe = current_frame;
    // graph->add_keyframe(current_keyframe);
    hasNewKeyFrame = true;

    std::cout << "KeyFrame needed at frame " << current_frame->id << std::endl; 
    std::cout << current_keyframe->pose.cast<float>().matrix() << std::endl;

    // // perform semantic analysis on keyframe
    // extract_objects(1, 0.002, 5, 20);
}

void System::process_images(const cv::Mat depth, const cv::Mat image, const fusion::IntrinsicMatrix base)
{
    cv::Mat depth_float;
    depth.convertTo(depth_float, CV_32FC1, 1 / 1000.f);
    float max_perct = 0.;
    int max_perct_idx = -1;
    float thres_new_sm = 0.50;
    float thres_passive = 0.20;
    renderIdx = manager->renderIdx;

    // std::cout << "Looping active submaps" << std::endl;
    // In tracking and Mapping, loop through all active submaps
    for(size_t i=0; i<manager->active_submaps.size(); ++i)
    {
        odometry->setSubmapIdx(i);

        // NOTE new pointer created here!
        // new frame for every submap
        current_frame = std::make_shared<RgbdFrame>(depth_float, image, frame_id, 0);

        /* INITIALIZATION */ 
        if (!is_initialized)
        {
            initialization();
            // std::cout << "Initializing... " << std::endl;
        }
        // else{
        //     std::cout << "Initialized " << std::endl;
        // }

        // std::cout << "Tracking - " << i <<  std::endl;
        /* TRACKING */
        if (!odometry->trackingLost){
            // std::cout << "Tracking - " << i <<  std::endl;
            // update pose of current_frame and reference_frame in corresponding DeviceImage
            odometry->trackFrame(current_frame);
            //---- only create kf in the rendering map ----
            if(keyframe_needed() && i == renderIdx)
            {
                create_keyframe();
            }
        }

        /* RENDERING */
        if (!odometry->trackingLost)
        {
            // std::cout << "Rendering - " << i <<  std::endl;
            auto reference_image = odometry->get_reference_image(i);
            auto reference_frame = reference_image->get_reference_frame();

            if(manager->active_submaps[i]->bRender){
                // update the map
                manager->active_submaps[i]->update(reference_image);
                manager->active_submaps[i]->raycast(reference_image->get_vmap(), reference_image->get_nmap(0), reference_frame->pose);
                // 
                if(hasNewKeyFrame){
                    reference_image->downloadVN(odometry->vModelFrames[i]);
                    manager->AddKeyFrame(current_keyframe);
                    // color object on the map
                    if(current_keyframe->numDetection > 0)
                        manager->active_submaps[i]->color_objects(reference_image);
                }
            } else {
                // PROBLEM with Xingrui's implementation! no vixible blocks found
                // manager->active_submaps[i]->raycast_check_visibility(reference_image->get_vmap(), reference_image->get_nmap(0), reference_frame->pose);
                //-Yohann
                manager->active_submaps[i]->check_visibility(reference_image);
                manager->active_submaps[i]->raycast(reference_image->get_vmap(), reference_image->get_nmap(0), reference_frame->pose);
            }
            reference_image->resize_device_map();

            // set current map idx to trackIdx in the odometry
            odometry->setTrackIdx(i);
        }
        // !!!!!!!!!!!!!!!!! Consider move relocalization out the loop
        /* RELOCALIZATION */
        else
        {
            std::cout << "\n !!!! Tracking Lost at frame " << frame_id << "! Trying to recover..." << std::endl;
            std::vector<std::shared_ptr<Point3d>> points;
            auto descriptors = graph->get_descriptor_all(points);

            relocalizer->set_target_frame(current_frame);
            relocalizer->set_map_points(points, descriptors);

            std::vector<Sophus::SE3d> candidates;
            relocalizer->compute_pose_candidates(candidates);

            for (const auto &candidate : candidates)
            {
                auto reference_image = odometry->get_reference_image(i);
                auto reference_frame = reference_image->get_reference_frame();
                reference_frame->pose = candidate;
                // mapping->raycast_check_visibility(reference_image->get_vmap(), reference_image->get_nmap(0), reference_frame->pose);
                manager->active_submaps[i]->raycast_check_visibility(reference_image->get_vmap(), reference_image->get_nmap(0), reference_frame->pose);
                reference_image->resize_device_map();

                // cv::Mat img(reference_image->get_vmap());
                // cv::imshow("img", img);
                // cv::waitKey(0);
            }

            //TODO : raycast verification

            // odometry->trackingLost = false;
        }

        // check visible block percentage
        float tmp_perct = manager->CheckVisPercent(i);
        if(tmp_perct < thres_passive){
            std::cout << "Move submap " << manager->active_submaps[i]->submapIdx << " from active to passive."
                      << " With visible_percentage = " << tmp_perct << std::endl;
            manager->activeTOpassiveIdx.push_back(i);
        }
        if(tmp_perct > max_perct){
            max_perct = tmp_perct;
            max_perct_idx = i;
        }
        
    } // end loop for active submaps

    frame_id += 1;

    /* POST-PROCESSING */
    // deactivate unwanted submaps
    if(manager->activeTOpassiveIdx.size()>0)
    {
        renderIdx -= manager->activeTOpassiveIdx.size();
        manager->CheckActive();
    }

    // std::cout << "check new sm/render&track" << std::endl;
    // check if new submap is needed
    if(max_perct < thres_new_sm)
    {
        std::cout << "NEW SUBMAP NEEDED at frame " << current_frame->id << std::endl;

        // int new_map_idx_all = manager->all_submaps.size();
        int new_map_idx_all = manager->active_submaps.size() + manager->passive_submaps.size();
        manager->Create(base, new_map_idx_all, odometry->vModelDeviceMapPyramid[renderIdx], 
                        false, true);
        renderIdx = manager->renderIdx;
        create_keyframe();
        odometry->vModelDeviceMapPyramid[renderIdx]->downloadVN(odometry->vModelFrames[renderIdx]);
        manager->AddKeyFrame(current_keyframe);
    } 
    // check which submap to track and render
    else
    {
        // std::cout << renderIdx << "-" << odometry->vModelFrames.size() << std::endl;
        manager->CheckTrackAndRender(odometry->vModelFrames[renderIdx]->id, max_perct_idx);
    }

    /* OPTIMIZATION */
    if (hasNewKeyFrame)
    {
        auto reference_image = odometry->get_reference_image(renderIdx);
        auto reference_frame = reference_image->get_reference_frame();
        reference_image->get_vmap().download(reference_frame->vmap);
        reference_image->get_nmap().download(reference_frame->nmap);

        graph->add_keyframe(reference_frame);
        hasNewKeyFrame = false;
    }
}

bool System::keyframe_needed() const
{
    auto pose = current_frame->pose;
    auto ref_pose = current_keyframe->pose;
    //---- create kf more frequently to get more object detection ----
    // if ((pose.inverse() * ref_pose).translation().norm() > 0.1f)
    if ((pose.inverse() * ref_pose).translation().norm() > 0.05f)
        return true;
    return false;
}

void System::create_keyframe()
{
    current_keyframe = odometry->vModelFrames[renderIdx];
    // graph->add_keyframe(current_keyframe);
    hasNewKeyFrame = true;

    std::cout << "\nKeyFrame needed at frame " << odometry->vModelFrames[renderIdx]->id << std::endl; 
    std::cout << current_keyframe->pose.cast<float>().matrix() << std::endl;

    // perform semantic analysis on keyframe
    extract_objects(1, 0.002, 5, 20);
}

void System::extract_objects(float lamb, float tao, int win_size, int thre)
{
    current_keyframe->ExtractObjects(detector, false, true, true);
    
    cv::Mat edge(current_keyframe->row_frame, current_keyframe->col_frame, CV_8UC1);
    auto current_keyimage = odometry->get_reference_image(renderIdx);
    current_keyimage->GeometricRefinement(lamb, tao, win_size, edge);
    
    current_keyframe->FuseMasks(edge, thre);

    odometry->vModelDeviceMapPyramid[renderIdx]->upload_semantics(current_keyframe);

    std::cout << "-- number of objects detected: " << current_keyframe->numDetection << std::endl;
}

cv::Mat System::get_detected_image()
{
    return current_keyframe->image;
}

cv::Mat System::get_shaded_depth()
{
    if (odometry->get_current_image())
        return cv::Mat(odometry->get_current_image()->get_rendered_image());
}

cv::Mat System::get_rendered_scene() const
{
    return cv::Mat(odometry->get_reference_image(renderIdx)->get_rendered_image());
}

cv::Mat System::get_rendered_scene_textured() const
{
    return cv::Mat(odometry->get_reference_image(renderIdx)->get_rendered_scene_textured());
}

void System::restart()
{
    // initialPose = last_tracked_frame->pose;
    is_initialized = false;
    frame_id = 0;

    manager->ResetSubmaps();
    odometry->reset();
    graph->reset();
}

void System::setLost(bool lost)
{
    odometry->trackingLost = true;
}

void System::save_mesh_to_file(const char *str)
{
}

size_t System::fetch_mesh_vertex_only(float *vertex)
{
    return manager->active_submaps[renderIdx]->fetch_mesh_vertex_only(vertex);
}

size_t System::fetch_mesh_with_normal(float *vertex, float *normal)
{
    return manager->active_submaps[renderIdx]->fetch_mesh_with_normal(vertex, normal);
}

size_t System::fetch_mesh_with_colour(float *vertex, unsigned char *colour)
{
    return manager->active_submaps[renderIdx]->fetch_mesh_with_colour(vertex, colour);
}

void System::fetch_key_points(float *points, size_t &count, size_t max)
{
    manager->GetPoints(points, count, max);
}

void System::fetch_key_points_with_normal(float *points, float *normal, size_t &max_size)
{
}

Eigen::Matrix4f System::get_camera_pose() const
{
    // Eigen::Matrix4f Tmf, Twm; 
    Eigen::Matrix4f T;
    if (odometry->get_reference_image(renderIdx))
    {
        // final display map is primary submap centered.
        T = odometry->get_reference_image(renderIdx)->get_reference_frame()->pose.cast<float>().matrix();
        // Twm = manager->active_submaps[renderIdx]->poseGlobal.cast<float>().matrix();
        // T = Twm * Tmf;
    }
    return T;
}

void System::writeMapToDisk(std::string file_name) const
{
    // mapping->writeMapToDisk(file_name);
    manager->active_submaps[0]->writeMapToDisk(file_name);
}

void System::readMapFromDisk(std::string file_name)
{
    // mapping->readMapFromDisk(file_name);
    manager->active_submaps[0]->readMapFromDisk(file_name);
}

std::vector<Eigen::Matrix<float, 4, 4>> System::getKeyFramePoses() const
{
    return manager->GetKFPoses();
}

} // namespace fusion