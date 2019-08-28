#ifndef GL_WINDOW_H
#define GL_WINDOW_H

#include "system.h"
#include <pangolin/pangolin.h>
#include <pangolin/gl/glcuda.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

class MainWindow
{
public:
    ~MainWindow();
    MainWindow(const char *name = "Untitled", size_t width = 640, size_t height = 480);

    //! Do not copy this class
    MainWindow(const MainWindow &) = delete;
    MainWindow &operator=(const MainWindow &) = delete;

    //! Main loop
    void Render();

    void ResetAllFlags();
    void SetVertexSize(size_t Size);
    void SetRGBSource(cv::Mat RgbImage);
    void SetDepthSource(cv::Mat DepthImage);
    void SetDetectedSource(cv::Mat DetectedImage);
    void SetRenderScene(cv::Mat SceneImage);
    void SetFeatureImage(cv::Mat featureImage);
    void SetCurrentCamera(Eigen::Matrix4f T);
    void SetSystem(fusion::System *sys);

    bool IsPaused();
    bool mbFlagRestart;
    bool mbFlagUpdateMesh;

    float *GetMappedVertexBuffer();
    float *GetMappedNormalBuffer();
    unsigned char *GetMappedColourBuffer();

    size_t VERTEX_COUNT;
    size_t MAX_VERTEX_COUNT;

private:
    //! Window Title
    std::string WindowName;

    void SetupDisplays();
    void SetupGLFlags();
    void InitTextures();
    void InitMeshBuffers();
    void InitGlSlPrograms();
    void RegisterKeyCallback();

    //! Displayed Views
    pangolin::View *mpViewSideBar;
    pangolin::View *mpViewRGB;
    pangolin::View *mpViewDepth;
    pangolin::View *mpViewScene;
    pangolin::View *mpViewMesh;
    pangolin::View *mpViewMenu;

    //! Displayed textures
    pangolin::GlTexture TextureRGB;
    pangolin::GlTexture TextureDepth;
    pangolin::GlTexture TextureScene;
    pangolin::GlTexture TextureDetected;

    //! Main 3D View Camera
    std::shared_ptr<pangolin::OpenGlRenderState> CameraView;

    //! GUI buttons and checkboxes
    std::shared_ptr<pangolin::Var<bool>> BtnReset;
    std::shared_ptr<pangolin::Var<bool>> BtnSaveMap;
    std::shared_ptr<pangolin::Var<bool>> BtnSetLost;
    std::shared_ptr<pangolin::Var<bool>> BtnReadMap;
    std::shared_ptr<pangolin::Var<bool>> BtnReloadMap;
    std::shared_ptr<pangolin::Var<bool>> BoxPaused;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayImage;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayDepth;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayScene;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayMesh;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayColor;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayCamera;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayKeyCameras;
    std::shared_ptr<pangolin::Var<bool>> BoxDisplayKeyPoint;

    //! Acquire Mehs Functions
    void UpdateMeshWithNormal();
    void UpdateMeshWithColour();

    //! Draw Mesh Functions
    void DrawMeshShaded();
    void DrawMeshColoured();
    void DrawMeshNormalMapped();

    //! Mesh Vertices
    // following variables store teh vertex, normal and color info from the map
    pangolin::GlBufferCudaPtr BufferVertex;
    pangolin::GlBufferCudaPtr BufferNormal;
    pangolin::GlBufferCudaPtr BufferColour;

    //! Registered CUDA Ptrs
    std::shared_ptr<pangolin::CudaScopedMappedPtr> MappedVertex;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> MappedNormal;
    std::shared_ptr<pangolin::CudaScopedMappedPtr> MappedColour;

    //! GL Shading program
    pangolin::GlSlProgram ShadingProg, ShadingColorProg;

    //! Vertex Array Objects
    //! Cannot find a replacement in Pangolin
    GLuint VAOShade, VAOColour;
    // stores vertex buffer

    //! Current Camera Pose
    Eigen::Matrix4f CameraPose;

    //! Key Frame Poses
    std::vector<Eigen::Matrix4f> ListOfKeyCameras;

    //! system ref
    fusion::System *slam;

    //! key point array
    float *keypoints;
    size_t sizeKeyPoint;
    size_t maxSizeKeyPoint;
};

#endif