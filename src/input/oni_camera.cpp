#include "input/oni_camera.h"

namespace fusion
{

ONICamera::ONICamera() : ONICamera(640, 480, 30)
{
}

ONICamera::ONICamera(int cols, int rows, int fps)
    : width(cols), height(rows), frame_rate(fps)
{
    // openni context initialization
    if (openni::OpenNI::initialize() != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    // openni camera open
    if (device.open(openni::ANY_DEVICE) != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    // create depth stream
    if (depth_stream.create(device, openni::SENSOR_DEPTH) != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    // create colour stream
    if (color_stream.create(device, openni::SENSOR_COLOR) != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    auto video_mode = openni::VideoMode();
    video_mode.setResolution(width, height);
    video_mode.setFps(frame_rate);
    video_mode.setPixelFormat(openni::PIXEL_FORMAT_DEPTH_1_MM);
    depth_stream.setVideoMode(video_mode);

    video_mode.setPixelFormat(openni::PIXEL_FORMAT_RGB888);
    color_stream.setVideoMode(video_mode);

    if (device.isImageRegistrationModeSupported(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR))
        device.setImageRegistrationMode(openni::IMAGE_REGISTRATION_DEPTH_TO_COLOR);

    color_stream.setMirroringEnabled(false);
    depth_stream.setMirroringEnabled(false);

    if (depth_stream.start() != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    if (color_stream.start() != openni::STATUS_OK)
    {
        std::cout << openni::OpenNI::getExtendedError() << std::endl;
        exit(0);
    }

    std::cout << "ONICamera Ready" << std::endl;
}

ONICamera::~ONICamera()
{
    color_stream.stop();
    color_stream.destroy();
    depth_stream.stop();
    depth_stream.destroy();
    device.close();
    openni::OpenNI::shutdown();

    std::cout << "ONICamera Stopped" << std::endl;
    return;
}

bool ONICamera::get_next_images(cv::Mat &depth, cv::Mat &image)
{
    openni::VideoStream *streams[] = {&depth_stream, &color_stream};

    int stream_ready = -1;
    auto last_state = openni::STATUS_OK;

    while (last_state == openni::STATUS_OK)
    {
        last_state = openni::OpenNI::waitForAnyStream(streams, 2, &stream_ready, 0);

        if (last_state == openni::STATUS_OK)
        {
            switch (stream_ready)
            {
            case 0: //depth ready
                if (depth_stream.readFrame(&depth_frame) == openni::STATUS_OK)
                    depth = cv::Mat(height, width, CV_16UC1, const_cast<void *>(depth_frame.getData()));
                break;

            case 1: // color ready
                if (color_stream.readFrame(&color_frame) == openni::STATUS_OK)
                {
                    image = cv::Mat(height, width, CV_8UC3, const_cast<void *>(color_frame.getData()));
                }
                break;

            default: // unexpected stream
                return false;
            }
        }
    }

    if (!depth_frame.isValid() || !color_frame.isValid())
        return false;

    return true;
}

} // namespace fusion
