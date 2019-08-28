#include "system.h"
#include "visualization/main_window.h"
#include "input/oni_camera.h"

int main(int argc, char **argv)
{
    fusion::ONICamera camera(640, 480, 30);
    fusion::IntrinsicMatrix K(640, 480, 580, 580, 319.5, 239.5);
    fusion::System slam(K, 5);

    MainWindow window("Semantic_ICP_SLAM", 1920, 920);
    window.SetSystem(&slam);
    cv::Mat image, depth;

    while (!pangolin::ShouldQuit())
    {
        if (camera.get_next_images(depth, image))
        {
            window.SetRGBSource(image);
            // window.SetDepthSource(depth);   // raw depth
            if (!window.IsPaused())
            {
                slam.process_images(depth, image, K);

                window.SetDetectedSource(slam.get_detected_image());
                // window.SetDepthSource(slam.get_shaded_depth());      // rendered depth
                window.SetRenderScene(slam.get_rendered_scene());
                // window.SetRenderScene(slam.get_rendered_scene_textured());
                window.SetCurrentCamera(slam.get_camera_pose());
                window.mbFlagUpdateMesh = true;
            }

            if (window.IsPaused() && window.mbFlagUpdateMesh)
            {
                // auto *vertex = window.GetMappedVertexBuffer();
                // auto *normal = window.GetMappedNormalBuffer();
                // window.VERTEX_COUNT = slam.fetch_mesh_with_normal(vertex, normal);

                auto *vertex = window.GetMappedVertexBuffer();
                auto *colour = window.GetMappedColourBuffer();
                window.VERTEX_COUNT = slam.fetch_mesh_with_colour(vertex, colour);

                window.mbFlagUpdateMesh = false;
            }
        }

        window.Render();
    }
}