#include "visualization/main_window.h"

#define ENTER_KEY 13

MainWindow::~MainWindow()
{
    delete keypoints;
    pangolin::DestroyWindow(WindowName);
    std::cout << "opengl released. " << std::endl;
}

MainWindow::MainWindow(const char *name, size_t width, size_t height)
    : mbFlagRestart(false), WindowName(name), mbFlagUpdateMesh(false),
      VERTEX_COUNT(0), MAX_VERTEX_COUNT(20000000), sizeKeyPoint(0),
      maxSizeKeyPoint(8000000)
{
    ResetAllFlags();

    pangolin::CreateWindowAndBind(WindowName, width, height);
    keypoints = (float *)malloc(sizeof(float) * maxSizeKeyPoint);

    SetupGLFlags();
    SetupDisplays();
    RegisterKeyCallback();
    InitTextures();
    InitMeshBuffers();
    InitGlSlPrograms();
}

void MainWindow::SetupGLFlags()
{
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    // glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void MainWindow::InitTextures()
{
    TextureRGB.Reinitialise(
        640, 480,
        GL_RGB,
        true,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        NULL);

    TextureDetected.Reinitialise(
        640, 480,
        GL_RGB,
        true,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        NULL);

    TextureDepth.Reinitialise(
        640, 480,
        GL_RGBA,
        true,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        NULL);

    TextureScene.Reinitialise(
        640, 480,
        GL_RGBA,
        true,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        NULL);
}

void MainWindow::InitMeshBuffers()
{
    auto size = sizeof(float) * 3 * MAX_VERTEX_COUNT;

    BufferVertex.Reinitialise(
        pangolin::GlArrayBuffer,
        size,
        cudaGLMapFlagsWriteDiscard,
        GL_STATIC_DRAW);

    BufferNormal.Reinitialise(
        pangolin::GlArrayBuffer,
        size,
        cudaGLMapFlagsWriteDiscard,
        GL_STATIC_DRAW);

    BufferColour.Reinitialise(
        pangolin::GlArrayBuffer,
        size,
        cudaGLMapFlagsWriteDiscard,
        GL_STATIC_DRAW);

    MappedVertex = std::make_shared<pangolin::CudaScopedMappedPtr>(BufferVertex);
    MappedNormal = std::make_shared<pangolin::CudaScopedMappedPtr>(BufferNormal);
    MappedColour = std::make_shared<pangolin::CudaScopedMappedPtr>(BufferColour);

    glGenVertexArrays(1, &VAOShade);
    glBindVertexArray(VAOShade);

    BufferVertex.Bind();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    BufferNormal.Bind();
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    glGenVertexArrays(1, &VAOColour);
    glBindVertexArray(VAOColour);

    BufferVertex.Bind();
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    BufferColour.Bind();
    glVertexAttribPointer(2, 3, GL_UNSIGNED_BYTE, GL_TRUE, 0, 0);
    glEnableVertexAttribArray(2);

    // previous buffers are unbinded automatically when binding next buffer.
    // so only last one need to call Unbind explicitly
    BufferColour.Unbind();
    glBindVertexArray(0);
}

bool MainWindow::IsPaused()
{
    return *BoxPaused;
}

void MainWindow::InitGlSlPrograms()
{
    ShadingProg.AddShaderFromFile(
        pangolin::GlSlShaderType::GlSlVertexShader,
        "./glsl_shader/phong.vert");

    ShadingProg.AddShaderFromFile(
        pangolin::GlSlShaderType::GlSlFragmentShader,
        "./glsl_shader/direct_output.frag");

    ShadingProg.Link();

    ShadingColorProg.AddShaderFromFile(
        pangolin::GlSlShaderType::GlSlVertexShader,
        "./glsl_shader/colour.vert");
    ShadingColorProg.AddShaderFromFile(
        pangolin::GlSlShaderType::GlSlFragmentShader,
        "./glsl_shader/direct_output.frag");
    ShadingColorProg.Link();
}

void MainWindow::SetupDisplays()
{
    CameraView = std::make_shared<pangolin::OpenGlRenderState>(
        pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
        pangolin::ModelViewLookAtRDF(0, 0, 0, 0, 0, -1, 0, 1, 0));

    auto MenuDividerLeft = pangolin::Attach::Pix(200);
    float RightSideBarDividerLeft = 0.7f;

    pangolin::CreatePanel("Menu").SetBounds(0, 1, 0, MenuDividerLeft);

    // name of the button, default value, shi fou you xuan ze kuang
    BtnReset = std::make_shared<pangolin::Var<bool>>("Menu.RESET", false, false);
    BtnSaveMap = std::make_shared<pangolin::Var<bool>>("Menu.Save Map", false, false);
    BtnSetLost = std::make_shared<pangolin::Var<bool>>("Menu.Set Lost", false, false);
    BtnReadMap = std::make_shared<pangolin::Var<bool>>("Menu.Read Map", false, false);
    BoxPaused = std::make_shared<pangolin::Var<bool>>("Menu.PAUSE", true, true);
    BoxDisplayImage = std::make_shared<pangolin::Var<bool>>("Menu.Display Image", true, true);
    BoxDisplayDepth = std::make_shared<pangolin::Var<bool>>("Menu.Display Depth", true, true);
    BoxDisplayScene = std::make_shared<pangolin::Var<bool>>("Menu.Display Scene", true, true);
    BoxDisplayMesh = std::make_shared<pangolin::Var<bool>>("Menu.Display Mesh", false, true);
    BtnReloadMap = std::make_shared<pangolin::Var<bool>>("Menu.Reload Map", false, false);
    BoxDisplayColor = std::make_shared<pangolin::Var<bool>>("Menu.Display Colored Mesh", true, true);
    BoxDisplayCamera = std::make_shared<pangolin::Var<bool>>("Menu.Display Camera", false, true);
    BoxDisplayKeyCameras = std::make_shared<pangolin::Var<bool>>("Menu.Display KeyFrame", false, true);
    BoxDisplayKeyPoint = std::make_shared<pangolin::Var<bool>>("Menu.Display KeyPoint", true, true);

    mpViewSideBar = &pangolin::Display("Right Side Bar");
    mpViewSideBar->SetBounds(0, 1, RightSideBarDividerLeft, 1);
    mpViewRGB = &pangolin::Display("RGB");
    mpViewRGB->SetBounds(0, 0.5, 0, 1);
    mpViewDepth = &pangolin::Display("Depth");
    mpViewDepth->SetBounds(0.5, 1, 0, 1);
    mpViewScene = &pangolin::Display("Scene");
    mpViewScene->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft);
    mpViewMesh = &pangolin::Display("Mesh");
    mpViewMesh->SetBounds(0, 1, MenuDividerLeft, RightSideBarDividerLeft).SetHandler(new pangolin::Handler3D(*CameraView));

    mpViewSideBar->AddDisplay(*mpViewRGB);
    mpViewSideBar->AddDisplay(*mpViewDepth);
}

void MainWindow::RegisterKeyCallback()
{
    //! Retart the system
    pangolin::RegisterKeyPressCallback('r', pangolin::SetVarFunctor<bool>("Menu.RESET", true));
    pangolin::RegisterKeyPressCallback('R', pangolin::SetVarFunctor<bool>("Menu.RESET", true));
    //! Pause / Resume the system
    pangolin::RegisterKeyPressCallback(ENTER_KEY, pangolin::ToggleVarFunctor("Menu.PAUSE"));
    //! Display keyframes
    pangolin::RegisterKeyPressCallback('c', pangolin::ToggleVarFunctor("Menu.Display KeyFrame"));
    pangolin::RegisterKeyPressCallback('C', pangolin::ToggleVarFunctor("Menu.Display KeyFrame"));
    //! Save Maps
    pangolin::RegisterKeyPressCallback('s', pangolin::SetVarFunctor<bool>("Menu.Save Map", true));
    pangolin::RegisterKeyPressCallback('S', pangolin::SetVarFunctor<bool>("Menu.Save Map", true));
    //! Load Maps
    pangolin::RegisterKeyPressCallback('l', pangolin::SetVarFunctor<bool>("Menu.Read Map", true));
    pangolin::RegisterKeyPressCallback('L', pangolin::SetVarFunctor<bool>("Menu.Read Map", true));
}

void MainWindow::ResetAllFlags()
{
    mbFlagRestart = false;
    // mbFlagUpdateMesh = false;
}

void MainWindow::SetRGBSource(cv::Mat RgbImage)
{
    TextureRGB.Upload(RgbImage.data, GL_RGB, GL_UNSIGNED_BYTE);
}

void MainWindow::SetDepthSource(cv::Mat DepthImage)
{
    TextureDepth.Upload(DepthImage.data, GL_RGBA, GL_UNSIGNED_BYTE);
}

void MainWindow::SetDetectedSource(cv::Mat DetectedImage)
{
    TextureDetected.Upload(DetectedImage.data, GL_RGB, GL_UNSIGNED_BYTE);
}

void MainWindow::SetRenderScene(cv::Mat SceneImage)
{
    TextureScene.Upload(SceneImage.data, GL_RGBA, GL_UNSIGNED_BYTE);
}

void MainWindow::SetFeatureImage(cv::Mat featureImage)
{
}

void MainWindow::Render()
{
    ResetAllFlags();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.f, 0.f, 0.f, 1.f);

    if (pangolin::Pushed(*BtnReset))
    {
        slam->restart();
        if (IsPaused())
            UpdateMeshWithNormal();
    }

    if (pangolin::Pushed(*BtnSaveMap))
    {
        slam->writeMapToDisk("map.data");
    }

    if (pangolin::Pushed(*BtnSetLost))
    {
        slam->setLost(true);
    }

    if (pangolin::Pushed(*BtnReadMap))
    {
        slam->readMapFromDisk("map.data");
        if (IsPaused())
            UpdateMeshWithNormal();
    }

    if (*BoxDisplayImage)
    {
        mpViewRGB->Activate();
        TextureRGB.RenderToViewportFlipY();
    }

    if (*BoxDisplayDepth)
    {
        mpViewDepth->Activate();
        // TextureDepth.RenderToViewportFlipY();
        TextureDetected.RenderToViewportFlipY();
    }

    if (!IsPaused())
    {
        mpViewScene->Activate();
        TextureScene.RenderToViewportFlipY();
    }
    else
    {
        mpViewMesh->Activate(*CameraView);

        if (*BoxDisplayMesh)
        {
            if(pangolin::Pushed(*BtnReloadMap))
                UpdateMeshWithNormal();
            DrawMeshShaded();
        }

        if (*BoxDisplayColor)
        {
            if(pangolin::Pushed(*BtnReloadMap))
                UpdateMeshWithColour();
            DrawMeshColoured();
        }

        Eigen::Matrix3f K;
        K << 580, 0, 320, 0, 580, 240, 0, 0, 1;

        if (*BoxDisplayCamera)
        {
            pangolin::glDrawFrustum(K.inverse().eval(), 640, 480, CameraPose, 0.1f);
            pangolin::glDrawAxis(CameraPose, 0.1f);
        }

        if (*BoxDisplayKeyCameras)
        {
            auto keyframe_poses = slam->getKeyFramePoses();
            std::vector<Eigen::Matrix<float, 3, 1>> camera_centers;
            for (const auto &pose : keyframe_poses)
            {
                camera_centers.push_back(pose.topRightCorner(3, 1));
                pangolin::glDrawFrustum(K.inverse().eval(), 640, 480, pose, 0.02f);
                pangolin::glDrawAxis(pose, 0.02f);
            }

            pangolin::glDrawVertices(camera_centers, GL_LINE_STRIP);
        }

        if (*BoxDisplayKeyPoint)
        {
            slam->fetch_key_points(&keypoints[0], sizeKeyPoint, maxSizeKeyPoint);
            glColor4f(0.f, 1.f, 0.f, 1.f);
            glPointSize(3);
            pangolin::glDrawVertices(sizeKeyPoint, &keypoints[0], GL_POINTS, 3);
            glPointSize(1);
            glColor4f(1.f, 1.f, 1.f, 1.f);
        }
    }

    pangolin::FinishFrame();
}

void MainWindow::UpdateMeshWithNormal()
{
    auto *vertex = GetMappedVertexBuffer();
    auto *normal = GetMappedNormalBuffer();
    VERTEX_COUNT = slam->fetch_mesh_with_normal(vertex, normal);
}

void MainWindow::DrawMeshShaded()
{
    if (VERTEX_COUNT == 0)
        return;

    ShadingProg.Bind();
    glBindVertexArray(VAOShade);

    ShadingProg.SetUniform("mvp_matrix", CameraView->GetProjectionModelViewMatrix());

    glDrawArrays(GL_TRIANGLES, 0, VERTEX_COUNT * 3);

    glBindVertexArray(0);
    ShadingProg.Unbind();
}

void MainWindow::UpdateMeshWithColour()
{
    auto *vertex = GetMappedVertexBuffer();
    auto *colour = GetMappedColourBuffer();
    VERTEX_COUNT = slam->fetch_mesh_with_colour(vertex, colour);
}

void MainWindow::DrawMeshColoured()
{
    if (VERTEX_COUNT == 0)
        return;

    ShadingColorProg.Bind();
    glBindVertexArray(VAOColour);

    ShadingColorProg.SetUniform("mvp_matrix", CameraView->GetProjectionModelViewMatrix());

    glDrawArrays(GL_TRIANGLES, 0, VERTEX_COUNT * 3);

    glBindVertexArray(0);
    ShadingColorProg.Unbind();
}

void MainWindow::DrawMeshNormalMapped()
{
}

void MainWindow::SetCurrentCamera(Eigen::Matrix4f T)
{
    CameraPose = T;
    // pangolin::OpenGlMatrix ViewMat(CameraPose.inverse().eval());
    // CameraView->SetModelViewMatrix(ViewMat);
}

void MainWindow::SetSystem(fusion::System *sys)
{
    slam = sys;
}

float *MainWindow::GetMappedVertexBuffer()
{
    return (float *)**MappedVertex;
}

float *MainWindow::GetMappedNormalBuffer()
{
    return (float *)**MappedNormal;
}

unsigned char *MainWindow::GetMappedColourBuffer()
{
    return (unsigned char *)**MappedColour;
}
