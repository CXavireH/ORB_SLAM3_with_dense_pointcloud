/**
* This file is part of ORB-SLAM3
*
* Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
* Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
*
* ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with ORB-SLAM3.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"
#include "G2oTypes.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include "ORBextractor.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include "GeometricCamera.h"

#include <thread>
#include <include/CameraModels/Pinhole.h>
#include <include/CameraModels/KannalaBrandt8.h>

// 自己加的--------------------------------------------

#include "Segment.h"

// The previous image
cv::Mat imGrayPre;
std::vector<cv::Point2f> prepoint, nextpoint;
std::vector<cv::Point2f> F_prepoint, F_nextpoint;
std::vector<cv::Point2f> F2_prepoint, F2_nextpoint;

std::vector<uchar> state;
std::vector<float> err;
std::vector<std::vector<cv::KeyPoint>> mvAllKeysPre;

// 间隔10Frames
// int skip = 0;

// ---------------------------------------------------


namespace ORB_SLAM3
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

//For stereo fisheye matching
cv::BFMatcher Frame::BFmatcher = cv::BFMatcher(cv::NORM_HAMMING);

Frame::Frame(): mpcpi(NULL), mpImuPreintegrated(NULL), mpPrevFrame(NULL), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mbHasPose(false), mbHasVelocity(false)
{
#ifdef REGISTER_TIMES
    mTimeStereoMatch = 0;
    mTimeORB_Ext = 0;
#endif
}


//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpcpi(frame.mpcpi),mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mK_(Converter::toMatrix3f(frame.mK)), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mImuCalib(frame.mImuCalib), mnCloseMPs(frame.mnCloseMPs),
     mpImuPreintegrated(frame.mpImuPreintegrated), mpImuPreintegratedFrame(frame.mpImuPreintegratedFrame), mImuBias(frame.mImuBias),
     mnId(frame.mnId), mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors), mNameFile(frame.mNameFile), mnDataset(frame.mnDataset),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2), mpPrevFrame(frame.mpPrevFrame), mpLastKeyFrame(frame.mpLastKeyFrame),
     mbIsSet(frame.mbIsSet), mbImuPreintegrated(frame.mbImuPreintegrated), mpMutexImu(frame.mpMutexImu),
     mpCamera(frame.mpCamera), mpCamera2(frame.mpCamera2), Nleft(frame.Nleft), Nright(frame.Nright),
     monoLeft(frame.monoLeft), monoRight(frame.monoRight), mvLeftToRightMatch(frame.mvLeftToRightMatch),
     mvRightToLeftMatch(frame.mvRightToLeftMatch), mvStereo3Dpoints(frame.mvStereo3Dpoints),
     mTlr(frame.mTlr), mRlr(frame.mRlr), mtlr(frame.mtlr), mTrl(frame.mTrl),
     mTcw(frame.mTcw), mbHasPose(false), mbHasVelocity(false)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++){
            mGrid[i][j]=frame.mGrid[i][j];
            if(frame.Nleft > 0)
            {
                mGridRight[i][j] = frame.mGridRight[i][j];
            }
        }

    if(frame.mbHasPose)
        SetPose(frame.GetPose());

    if(frame.HasVelocity())
    {
        SetVelocity(frame.GetVelocity());
    }

    mmProjectPoints = frame.mmProjectPoints;
    mmMatchedInImage = frame.mmMatchedInImage;

#ifdef REGISTER_TIMES
    mTimeStereoMatch = frame.mTimeStereoMatch;
    mTimeORB_Ext = frame.mTimeORB_Ext;
#endif
}

// Constructor for stereo cameras
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera) ,mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    // // 20231019自己加的------------------------------------------------------------
    thread threadLeft(&Frame::ExtractORBKeyPoints,this,0,imLeft,0,0);
    thread threadRight(&Frame::ExtractORBKeyPoints,this,1,imRight,0,0);
    // // --------------------------------------------------------------------
    // thread threadLeft(&Frame::ExtractORB,this,0,imLeft,0,0);
    // thread threadRight(&Frame::ExtractORB,this,1,imRight,0,0);
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
    mmProjectPoints.clear();
    mmMatchedInImage.clear();


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);



        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
            SetVelocity(pPrevF->GetVelocity());
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();
}

// Constructor for RGB-D cameras
// `Original`
// ****将  Frame::Frame(...)拆分成了： 提取特征点并检测动态部分/计算静态特征点的描述子
// ****即  Frame + calculeverything()    
// Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
//     :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
//      mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
//      mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
//      mpCamera(pCamera),mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
// {
//     // Frame ID
//     mnId=nNextId++;

//     // Scale Level Info
//     mnScaleLevels = mpORBextractorLeft->GetLevels();  // 尺度级别
//     mfScaleFactor = mpORBextractorLeft->GetScaleFactor();  // 不用尺度级别之间的尺度因子
//     mfLogScaleFactor = log(mfScaleFactor);  // 计算了尺度因子的自然对数
//     mvScaleFactors = mpORBextractorLeft->GetScaleFactors();  // 获取所有尺度级别的尺度因子的数组
//     mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();  // 获取所有尺度级别的尺度因子的倒数
//     mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();  // 获取了一个尺度Sigma平方的数组或向量。Sigma通常用于描述高斯函数的标准差，而这些Sigma值可能与不同尺度级别上应用的高斯模糊有关。
//     mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();  // 获取了尺度Sigma平方的倒数数组或向量。

//     // ORB extraction
// #ifdef REGISTER_TIMES
//     std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
// #endif
//     ExtractORB(0,imGray,0,0);
// #ifdef REGISTER_TIMES
//     std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

//     mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
// #endif

//     // 获取关键点数量 N，通常是图像中检测到的特征点数量
//     N = mvKeys.size();
//     // 如果特征点列表为空，就返回（不执行后续操作）
//     if(mvKeys.empty())
//         return;
//     // 对特征点进行去畸变处理
//     UndistortKeyPoints();
//     // 从RGB-D图像计算立体视觉信息
//     ComputeStereoFromRGBD(imDepth);
//     // 创建一个包含 N 个空指针的地图点指针数组
//     mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
//     // 清空两个关于投影点和匹配图像的数据结构
//     mmProjectPoints.clear();
//     mmMatchedInImage.clear();
//     // 创建一个包含 N 个布尔值的数组，用于标记哪些特征点是离群点
//     mvbOutlier = vector<bool>(N,false);

//     // This is done only for the first Frame (or after a change in the calibration)
//     // 如果是第一帧或者相机标定参数发生了变化，执行以下初始化操作
//     if(mbInitialComputations)
//     {
//         // 计算图像的边界
//         ComputeImageBounds(imGray);
//         // 计算网格的宽度和高度的倒数
//         mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
//         mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);
//         // 从相机内参矩阵 K 中提取一些参数
//         fx = K.at<float>(0,0);
//         fy = K.at<float>(1,1);
//         cx = K.at<float>(0,2);
//         cy = K.at<float>(1,2);
//         invfx = 1.0f/fx;
//         invfy = 1.0f/fy;

//         mbInitialComputations=false;
//     }
//     // 设置相机基线 mb，它是深度信息与视差信息之间的关系
//     mb = mbf/fx;
//     // 如果有上一帧的信息，将上一帧的速度信息传递给当前帧
//     if(pPrevF){
//         if(pPrevF->HasVelocity())
//             SetVelocity(pPrevF->GetVelocity());
//     }
//     else{
//         mVw.setZero();
//     }
//     // 创建一个互斥锁，用于处理IMU数据（惯性测量单元）
//     mpMutexImu = new std::mutex();

//     //Set no stereo fisheye information
//     Nleft = -1;
//     Nright = -1;
//     mvLeftToRightMatch = vector<int>(0);
//     mvRightToLeftMatch = vector<int>(0);
//     mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
//     monoLeft = -1;
//     monoRight = -1;
//     // 将特征点分配到网格中
//     AssignFeaturesToGrid();
// }

// 20231019
// Constructor for RGB-D cameras
// `自己写的`
// ****将  Frame::Frame(...)拆分成了： 提取特征点并检测动态部分/计算静态特征点的描述子
// ****即  Frame + calculeverything()  
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera,Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF), mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false),
     mpCamera(pCamera),mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();  // 尺度级别
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();  // 不用尺度级别之间的尺度因子
    mfLogScaleFactor = log(mfScaleFactor);  // 计算了尺度因子的自然对数
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();  // 获取所有尺度级别的尺度因子的数组
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();  // 获取所有尺度级别的尺度因子的倒数
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();  // 获取了一个尺度Sigma平方的数组或向量。Sigma通常用于描述高斯函数的标准差，而这些Sigma值可能与不同尺度级别上应用的高斯模糊有关。
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();  // 获取了尺度Sigma平方的倒数数组或向量。

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    // ExtractORB(0,imGray,0,0);

    // 20231019自己加的----------------------------
    ExtractORBKeyPoints(0,imGray,0,0);
    // ------------------------------------
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    // 20231019自己加的-----------------------------------------------
    cv::Mat imGrayT = imGray;
    if (imGrayPre.data)
    {
        ProcessMovingObject(imGray);
        std::swap(imGrayPre,imGrayT);
    }
    else{
        // // 间隔10Frames---------
        // if (skip < 6)
        // {
        //     flag_mov = 0;
        //     skip++;
        // }
        // // --------------------
        // else{
        std::swap(imGrayPre, imGrayT);
        flag_mov = 0;
        // }

    }
    // ------------------------------------------------------
}

//  处理落在动态部分的特征点,并完成之前ORBExtract函数中的剩余部分(计算描述子\assignfeaturestogrid())

void Frame::CalculEverything(cv::Mat &imRGB, const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imS)
{
    // check whether there are dynamic objects
    if (!T_M.empty()) //T_M: Outliers vector && people exists 如果場景中有人並且有异常點，則通過label進行進一步的運動檢查
    {
        std::chrono::steady_clock::time_point tc1 = std::chrono::steady_clock::now();
        //erase outliers from mvKeysTemp
        flag_mov = mpORBextractorLeft->CheckMovingKeyPoints(imGray, imS, mvKeysTemp, T_M);
        std::chrono::steady_clock::time_point tc2 = std::chrono::steady_clock::now();
        double tc = std::chrono::duration_cast<std::chrono::duration<double>>(tc2 - tc1).count();
        cout << "check time =" << tc * 1000 << endl;
    }

    ExtractORBDesp(0, imGray, 0, 0);
    // ******************************************************************
    // 获取关键点数量 N，通常是图像中检测到的特征点数量
    N = mvKeys.size();
    // 如果特征点列表为空，就返回（不执行后续操作）
    if(mvKeys.empty())
        return;
    // 对特征点进行去畸变处理
    UndistortKeyPoints();
    // 从RGB-D图像计算立体视觉信息
    ComputeStereoFromRGBD(imDepth);
    // 创建一个包含 N 个空指针的地图点指针数组
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    // 清空两个关于投影点和匹配图像的数据结构
    mmProjectPoints.clear();
    mmMatchedInImage.clear();
    // 创建一个包含 N 个布尔值的数组，用于标记哪些特征点是离群点
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    // 如果是第一帧或者相机标定参数发生了变化，执行以下初始化操作
    if(mbInitialComputations)
    {
        // 计算图像的边界
        ComputeImageBounds(imGray);
        // 计算网格的宽度和高度的倒数
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);
        // 从相机内参矩阵 K 中提取一些参数
        fx = mK.at<float>(0,0);
        fy = mK.at<float>(1,1);
        cx = mK.at<float>(0,2);
        cy = mK.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }
    // 设置相机基线 mb，它是深度信息与视差信息之间的关系
    mb = mbf/fx;
    // 如果有上一帧的信息，将上一帧的速度信息传递给当前帧
    if(mpPrevFrame){
        if(mpPrevFrame->HasVelocity())
            SetVelocity(mpPrevFrame->GetVelocity());
    }
    else{
        mVw.setZero();
    }
    // 创建一个互斥锁，用于处理IMU数据（惯性测量单元）
    mpMutexImu = new std::mutex();

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;
    // 将特征点分配到网格中
    AssignFeaturesToGrid();
}

void Frame::ProcessMovingObject(const cv::Mat &imgray)
{
    // Clear the previous data
    F_prepoint.clear();
    F_nextpoint.clear();
    F2_prepoint.clear();
    F2_nextpoint.clear();
    T_M.clear();

    // Detect dynamic target and ultimately optput the T matrix

    /*
    imGrayPre：输入的灰度图像，这里是先前的图像帧。
    prepoint：存储检测到的特征点的向量。
    1000：要检测的最大特征点数量。
    0.01：角点质量水平，控制角点筛选的严格程度。
    8：角点之间的最小欧几里得距离。
    cv::Mat()：表示没有掩码图像。
    3：表示使用的窗口大小。
    true：表示是否使用 Harris 角点检测方法（如果为 false，则使用 Shi-Tomasi 方法）。
    0.04：Harris 角点检测方法的参数。
    */
    cv::goodFeaturesToTrack(imGrayPre, prepoint, 1000, 0.01, 8, cv::Mat(), 3, true, 0.04);
    /*
    对上一步中检测到的角点进行亚像素级别的精确化。具体参数解释如下：

    imGrayPre：输入的灰度图像，同样是先前的图像帧。
    prepoint：包含待精确化的角点的向量。
    cv::Size(10, 10)：搜索窗口的大小，用于精确化角点位置。
    cv::Size(-1, -1)：表示没有搜索的初始位置。
    cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, 0.03)：
    用于确定亚像素级别搜索的终止条件。这里表示最大迭代次数为20，或者达到亚像素级别的收敛误差为0.03时停止搜索。
    */
    cv::cornerSubPix(imGrayPre, prepoint, cv::Size(10, 10), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, 0.03));
    /*
    计算了图像中特征点的光流。具体参数解释如下：
    imGrayPre：先前的灰度图像帧。
    imgray：当前的灰度图像帧。
    prepoint：先前帧中的特征点的位置。
    nextpoint：在当前帧中估计的特征点的新位置。
    state：输出的状态向量，表示每个特征点是否成功跟踪。
    err：输出的误差向量，表示每个特征点的跟踪误差。
    cv::Size(22, 22)：金字塔窗口的大小，用于多尺度光流计算。
    5：表示金字塔的层数。
    cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, 0.01)：
    光流计算的终止条件，最大迭代次数为20，或者达到指定误差水平（0.01）时停止计算。
    */
    cv::calcOpticalFlowPyrLK(imGrayPre, imgray, prepoint, nextpoint, state, err, cv::Size(22, 22), 5, cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 20, 0.01));

    for (int i = 0; i < state.size(); i++)
    {
        // 如果光流法的状态为0，表示特征点跟踪失败，跳过此点
        if (state[i] != 0)
        {
            // 定义一个3x3的邻域，在该邻域内计算特征点的光流
            int dx[10] = {-1, 0, 1, -1, 0, 1, -1, 0, 1};
            int dy[10] = {-1, -1, -1, 0, 0, 0, 1, 1, 1};
            // 获取当前特征点的坐标
            int x1 = prepoint[i].x, y1 = prepoint[i].y;
            int x2 = nextpoint[i].x, y2 = nextpoint[i].y;
            // 检查特征点是否位于图像边缘，如果是，则将状态设置为0并跳过此点
            if ((x1 < limit_edge_corner || x1 >= imgray.cols - limit_edge_corner || x2 < limit_edge_corner || x2 >= imgray.cols - limit_edge_corner || y1 < limit_edge_corner || y1 >= imgray.rows - limit_edge_corner || y2 < limit_edge_corner || y2 >= imgray.rows - limit_edge_corner))
            {
                state[i] = 0;
                continue;
            }
            // 计算邻域内的灰度差异的绝对值之和
            double sum_check = 0;
            for (int j = 0; j < 9; j++)
                sum_check += abs(imGrayPre.at<uchar>(y1 + dy[j], x1 + dx[j]) - imgray.at<uchar>(y2 + dy[j], x2 + dx[j]));
            // 如果灰度差异之和超过了阈值，将状态设置为0，表示特征点跟踪失败
            if (sum_check > limit_of_check)
                state[i] = 0;
            if (state[i])
            {
                F_prepoint.push_back(prepoint[i]);
                F_nextpoint.push_back(nextpoint[i]);
            }
        }
    }
    // F-Matrix
    cv::Mat mask = cv::Mat(cv::Size(1, 300), CV_8UC1);
    // 在这里用到了RANSAC的方法
    /*
    mask 是一个输出参数，也是输入参数。
    它是一个与特征点一一对应的掩码，用于标记哪些对应点用于估计基本矩阵，哪些点应该被忽略。
    在函数调用之前，mask 的初始值通常是全1，表示所有对应点都参与估计。
    函数执行后，mask 的值将根据 RANSAC（随机抽样一致性）算法的结果进行更新，
    即被认为是异常值的对应点将被标记为0，而被认为是内点的对应点将保持为1。
    */
    cv::Mat F = cv::findFundamentalMat(F_prepoint, F_nextpoint, mask, cv::FM_RANSAC, 0.1, 0.99);
    for (int i = 0; i < mask.rows; i++)
    {
        if (mask.at<uchar>(i, 0) == 0)
            ;
        else
        {
            // Circle(pre_frame, F_prepoint[i], 6, Scalar(255, 255, 0), 3);
            double A = F.at<double>(0, 0) * F_prepoint[i].x + F.at<double>(0, 1) * F_prepoint[i].y + F.at<double>(0, 2);
            double B = F.at<double>(1, 0) * F_prepoint[i].x + F.at<double>(1, 1) * F_prepoint[i].y + F.at<double>(1, 2);
            double C = F.at<double>(2, 0) * F_prepoint[i].x + F.at<double>(2, 1) * F_prepoint[i].y + F.at<double>(2, 2);
            double dd = fabs(A * F_nextpoint[i].x + B * F_nextpoint[i].y + C) / sqrt(A * A + B * B); //Epipolar constraints
            if (dd <= 0.1)
            {
                // 把满足对极约束的特征点对存入F2_point
                F2_prepoint.push_back(F_prepoint[i]);
                F2_nextpoint.push_back(F_nextpoint[i]);
            }
        }
    }
    F_prepoint = F2_prepoint;
    F_nextpoint = F2_nextpoint;

    for (int i = 0; i < prepoint.size(); i++)
    {
        if (state[i] != 0)
        {
            double A = F.at<double>(0, 0) * prepoint[i].x + 
                       F.at<double>(0, 1) * prepoint[i].y + F.at<double>(0, 2);
            double B = F.at<double>(1, 0) * prepoint[i].x + 
                       F.at<double>(1, 1) * prepoint[i].y + F.at<double>(1, 2);
            double C = F.at<double>(2, 0) * prepoint[i].x + 
                       F.at<double>(2, 1) * prepoint[i].y + F.at<double>(2, 2);
            double dd = fabs(A * nextpoint[i].x + B * nextpoint[i].y + C) / sqrt(A * A + B * B);

            // Judge outliers
            if (dd <= limit_dis_epi)
                continue;
            T_M.push_back(nextpoint[i]);
        }
    }
}


// Constructor for Monocular cameras
Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, GeometricCamera* pCamera, cv::Mat &distCoef, const float &bf, const float &thDepth, Frame* pPrevF, const IMU::Calib &ImuCalib)
    :mpcpi(NULL),mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(static_cast<Pinhole*>(pCamera)->toK()), mK_(static_cast<Pinhole*>(pCamera)->toK_()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mImuCalib(ImuCalib), mpImuPreintegrated(NULL),mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbIsSet(false), mbImuPreintegrated(false), mpCamera(pCamera),
     mpCamera2(nullptr), mbHasPose(false), mbHasVelocity(false)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    // ExtractORB(0,imGray,0,1000);
    // // 20231019自己加的---------------------
    ExtractORBKeyPoints(0,imGray,0,1000);
    // // ----------------------------
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif


    N = mvKeys.size();
    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);
    mnCloseMPs = 0;

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));

    mmProjectPoints.clear();// = map<long unsigned int, cv::Point2f>(N, static_cast<cv::Point2f>(NULL));
    mmMatchedInImage.clear();

    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,0);
        fy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,1);
        cx = static_cast<Pinhole*>(mpCamera)->toK().at<float>(0,2);
        cy = static_cast<Pinhole*>(mpCamera)->toK().at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }


    mb = mbf/fx;

    //Set no stereo fisheye information
    Nleft = -1;
    Nright = -1;
    mvLeftToRightMatch = vector<int>(0);
    mvRightToLeftMatch = vector<int>(0);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(0);
    monoLeft = -1;
    monoRight = -1;

    AssignFeaturesToGrid();

    if(pPrevF)
    {
        if(pPrevF->HasVelocity())
        {
            SetVelocity(pPrevF->GetVelocity());
        }
    }
    else
    {
        mVw.setZero();
    }

    mpMutexImu = new std::mutex();
}


void Frame::AssignFeaturesToGrid()
{
    // Fill matrix with points
    const int nCells = FRAME_GRID_COLS*FRAME_GRID_ROWS;

    int nReserve = 0.5f*N/(nCells);

    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++){
            mGrid[i][j].reserve(nReserve);
            if(Nleft != -1){
                mGridRight[i][j].reserve(nReserve);
            }
        }



    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = (Nleft == -1) ? mvKeysUn[i]
                                                 : (i < Nleft) ? mvKeys[i]
                                                                 : mvKeysRight[i - Nleft];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY)){
            if(Nleft == -1 || i < Nleft)
                mGrid[nGridPosX][nGridPosY].push_back(i);
            else
                mGridRight[nGridPosX][nGridPosY].push_back(i - Nleft);
        }
    }
}

// void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
// {
//     vector<int> vLapping = {x0,x1};
//     if(flag==0)
//         monoLeft = (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors,vLapping);
//     else
//         monoRight = (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight,vLapping);
// }

/*  将原本void Frame::ExtractORB(int flag, const cv::Mat &im, const int x0, const int x1)
    拆分成提取特征点和计算描述子两部分

*/
// // 20231019自己加的-------------------------------------------------------
void Frame::ExtractORBKeyPoints(int flag, const cv::Mat &imgray, const int x0, const int x1)
{
    vector<int> vLapping = {x0, x1};
    if (flag == 0)
        monoLeft = (*mpORBextractorLeft)(imgray,cv::Mat(),mvKeysTemp,vLapping);
    else
        monoRight = (*mpORBextractorRight)(imgray,cv::Mat(),mvKeysTemp,vLapping);
}

// outliers has already removed from mvkeys
void Frame::ExtractORBDesp(int flag, const cv::Mat &imgray, const int x0, const int x1)
{
    // Save final result to mvKeys and mDescriptors
    vector<int> vLapping = {x0, x1};
    if (flag == 0)
        (*mpORBextractorLeft).ProcessDesp(imgray, cv::Mat(), mvKeysTemp, mvKeys, mDescriptors,vLapping);
    else
        (*mpORBextractorLeft).ProcessDesp(imgray, cv::Mat(), mvKeysTemp, mvKeysRight, mDescriptorsRight,vLapping);
}


// // --------------------------------------------------------------



bool Frame::isSet() const {
    return mbIsSet;
}

void Frame::SetPose(const Sophus::SE3<float> &Tcw) {
    mTcw = Tcw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::SetNewBias(const IMU::Bias &b)
{
    mImuBias = b;
    if(mpImuPreintegrated)
        mpImuPreintegrated->SetNewBias(b);
}

void Frame::SetVelocity(Eigen::Vector3f Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;
}

Eigen::Vector3f Frame::GetVelocity() const
{
    return mVw;
}

void Frame::SetImuPoseVelocity(const Eigen::Matrix3f &Rwb, const Eigen::Vector3f &twb, const Eigen::Vector3f &Vwb)
{
    mVw = Vwb;
    mbHasVelocity = true;

    Sophus::SE3f Twb(Rwb, twb);
    Sophus::SE3f Tbw = Twb.inverse();

    mTcw = mImuCalib.mTcb * Tbw;

    UpdatePoseMatrices();
    mbIsSet = true;
    mbHasPose = true;
}

void Frame::UpdatePoseMatrices()
{
    Sophus::SE3<float> Twc = mTcw.inverse();
    mRwc = Twc.rotationMatrix();
    mOw = Twc.translation();
    mRcw = mTcw.rotationMatrix();
    mtcw = mTcw.translation();
}

Eigen::Matrix<float,3,1> Frame::GetImuPosition() const {
    return mRwc * mImuCalib.mTcb.translation() + mOw;
}

Eigen::Matrix<float,3,3> Frame::GetImuRotation() {
    return mRwc * mImuCalib.mTcb.rotationMatrix();
}

Sophus::SE3<float> Frame::GetImuPose() {
    return mTcw.inverse() * mImuCalib.mTcb;
}

Sophus::SE3f Frame::GetRelativePoseTrl()
{
    return mTrl;
}

Sophus::SE3f Frame::GetRelativePoseTlr()
{
    return mTlr;
}

Eigen::Matrix3f Frame::GetRelativePoseTlr_rotation(){
    return mTlr.rotationMatrix();
}

Eigen::Vector3f Frame::GetRelativePoseTlr_translation() {
    return mTlr.translation();
}


bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    if(Nleft == -1){
        pMP->mbTrackInView = false;
        pMP->mTrackProjX = -1;
        pMP->mTrackProjY = -1;

        // 3D in absolute coordinates
        Eigen::Matrix<float,3,1> P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const Eigen::Matrix<float,3,1> Pc = mRcw * P + mtcw;
        const float Pc_dist = Pc.norm();

        // Check positive depth
        const float &PcZ = Pc(2);
        const float invz = 1.0f/PcZ;
        if(PcZ<0.0f)
            return false;

        const Eigen::Vector2f uv = mpCamera->project(Pc);

        if(uv(0)<mnMinX || uv(0)>mnMaxX)
            return false;
        if(uv(1)<mnMinY || uv(1)>mnMaxY)
            return false;

        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const Eigen::Vector3f PO = P - mOw;
        const float dist = PO.norm();

        if(dist<minDistance || dist>maxDistance)
            return false;

        // Check viewing angle
        Eigen::Vector3f Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn)/dist;

        if(viewCos<viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist,this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjXR = uv(0) - mbf*invz;

        pMP->mTrackDepth = Pc_dist;

        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }
    else{
        pMP->mbTrackInView = false;
        pMP->mbTrackInViewR = false;
        pMP -> mnTrackScaleLevel = -1;
        pMP -> mnTrackScaleLevelR = -1;

        pMP->mbTrackInView = isInFrustumChecks(pMP,viewingCosLimit);
        pMP->mbTrackInViewR = isInFrustumChecks(pMP,viewingCosLimit,true);

        return pMP->mbTrackInView || pMP->mbTrackInViewR;
    }
}

bool Frame::ProjectPointDistort(MapPoint* pMP, cv::Point2f &kp, float &u, float &v)
{

    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    // 3D in camera coordinates
    const Eigen::Vector3f Pc = mRcw * P + mtcw;
    const float &PcX = Pc(0);
    const float &PcY= Pc(1);
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
    {
        cout << "Negative depth: " << PcZ << endl;
        return false;
    }

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    u=fx*PcX*invz+cx;
    v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    float u_distort, v_distort;

    float x = (u - cx) * invfx;
    float y = (v - cy) * invfy;
    float r2 = x * x + y * y;
    float k1 = mDistCoef.at<float>(0);
    float k2 = mDistCoef.at<float>(1);
    float p1 = mDistCoef.at<float>(2);
    float p2 = mDistCoef.at<float>(3);
    float k3 = 0;
    if(mDistCoef.total() == 5)
    {
        k3 = mDistCoef.at<float>(4);
    }

    // Radial distorsion
    float x_distort = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
    float y_distort = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

    // Tangential distorsion
    x_distort = x_distort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
    y_distort = y_distort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

    u_distort = x_distort * fx + cx;
    v_distort = y_distort * fy + cy;


    u = u_distort;
    v = v_distort;

    kp = cv::Point2f(u, v);

    return true;
}

Eigen::Vector3f Frame::inRefCoordinates(Eigen::Vector3f pCw)
{
    return mRcw * pCw + mtcw;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel, const bool bRight) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    float factorX = r;
    float factorY = r;

    const int nMinCellX = max(0,(int)floor((x-mnMinX-factorX)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
    {
        return vIndices;
    }

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+factorX)*mfGridElementWidthInv));
    if(nMaxCellX<0)
    {
        return vIndices;
    }

    const int nMinCellY = max(0,(int)floor((y-mnMinY-factorY)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
    {
        return vIndices;
    }

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+factorY)*mfGridElementHeightInv));
    if(nMaxCellY<0)
    {
        return vIndices;
    }

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = (!bRight) ? mGrid[ix][iy] : mGridRight[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = (Nleft == -1) ? mvKeysUn[vCell[j]]
                                                         : (!bRight) ? mvKeys[vCell[j]]
                                                                     : mvKeysRight[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<factorX && fabs(disty)<factorY)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);

    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat, static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);


    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }

}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,static_cast<Pinhole*>(mpCamera)->toK(),mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // Undistort corners
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));
    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

bool Frame::UnprojectStereo(const int &i, Eigen::Vector3f &x3D)
{
    const float z = mvDepth[i];
    if(z>0) {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        Eigen::Vector3f x3Dc(x, y, z);
        x3D = mRwc * x3Dc + mOw;
        return true;
    } else
        return false;
}

bool Frame::imuIsPreintegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    return mbImuPreintegrated;
}

void Frame::setIntegrated()
{
    unique_lock<std::mutex> lock(*mpMutexImu);
    mbImuPreintegrated = true;
}

// constructor for fisheye camera
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth, GeometricCamera* pCamera, GeometricCamera* pCamera2, Sophus::SE3f& Tlr,Frame* pPrevF, const IMU::Calib &ImuCalib)
        :mpcpi(NULL), mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()), mK_(Converter::toMatrix3f(K)),  mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
         mImuCalib(ImuCalib), mpImuPreintegrated(NULL), mpPrevFrame(pPrevF),mpImuPreintegratedFrame(NULL), mpReferenceKF(static_cast<KeyFrame*>(NULL)), mbImuPreintegrated(false), mpCamera(pCamera), mpCamera2(pCamera2),
         mbHasPose(false), mbHasVelocity(false)

{
    imgLeft = imLeft.clone();
    imgRight = imRight.clone();

    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartExtORB = std::chrono::steady_clock::now();
#endif
    // thread threadLeft(&Frame::ExtractORB,this,0,imLeft,static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
    // thread threadRight(&Frame::ExtractORB,this,1,imRight,static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
    // // 20231019自己加的--------------------------------------------------
    thread threadLeft(&Frame::ExtractORBKeyPoints,this,0,imLeft,static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera)->mvLappingArea[1]);
    thread threadRight(&Frame::ExtractORBKeyPoints,this,1,imRight,static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[0],static_cast<KannalaBrandt8*>(mpCamera2)->mvLappingArea[1]);
    // // ---------------------------------------------------------
    threadLeft.join();
    threadRight.join();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndExtORB = std::chrono::steady_clock::now();

    mTimeORB_Ext = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndExtORB - time_StartExtORB).count();
#endif

    Nleft = mvKeys.size();
    Nright = mvKeysRight.size();
    N = Nleft + Nright;

    if(N == 0)
        return;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf / fx;

    // Sophus/Eigen
    mTlr = Tlr;
    mTrl = mTlr.inverse();
    mRlr = mTlr.rotationMatrix();
    mtlr = mTlr.translation();

#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_StartStereoMatches = std::chrono::steady_clock::now();
#endif
    ComputeStereoFishEyeMatches();
#ifdef REGISTER_TIMES
    std::chrono::steady_clock::time_point time_EndStereoMatches = std::chrono::steady_clock::now();

    mTimeStereoMatch = std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(time_EndStereoMatches - time_StartStereoMatches).count();
#endif

    //Put all descriptors in the same matrix
    cv::vconcat(mDescriptors,mDescriptorsRight,mDescriptors);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(nullptr));
    mvbOutlier = vector<bool>(N,false);

    AssignFeaturesToGrid();

    mpMutexImu = new std::mutex();

    UndistortKeyPoints();

}

void Frame::ComputeStereoFishEyeMatches() {
    //Speed it up by matching keypoints in the lapping area
    vector<cv::KeyPoint> stereoLeft(mvKeys.begin() + monoLeft, mvKeys.end());
    vector<cv::KeyPoint> stereoRight(mvKeysRight.begin() + monoRight, mvKeysRight.end());

    cv::Mat stereoDescLeft = mDescriptors.rowRange(monoLeft, mDescriptors.rows);
    cv::Mat stereoDescRight = mDescriptorsRight.rowRange(monoRight, mDescriptorsRight.rows);

    mvLeftToRightMatch = vector<int>(Nleft,-1);
    mvRightToLeftMatch = vector<int>(Nright,-1);
    mvDepth = vector<float>(Nleft,-1.0f);
    mvuRight = vector<float>(Nleft,-1);
    mvStereo3Dpoints = vector<Eigen::Vector3f>(Nleft);
    mnCloseMPs = 0;

    //Perform a brute force between Keypoint in the left and right image
    vector<vector<cv::DMatch>> matches;

    BFmatcher.knnMatch(stereoDescLeft,stereoDescRight,matches,2);

    int nMatches = 0;
    int descMatches = 0;

    //Check matches using Lowe's ratio
    for(vector<vector<cv::DMatch>>::iterator it = matches.begin(); it != matches.end(); ++it){
        if((*it).size() >= 2 && (*it)[0].distance < (*it)[1].distance * 0.7){
            //For every good match, check parallax and reprojection error to discard spurious matches
            Eigen::Vector3f p3D;
            descMatches++;
            float sigma1 = mvLevelSigma2[mvKeys[(*it)[0].queryIdx + monoLeft].octave], sigma2 = mvLevelSigma2[mvKeysRight[(*it)[0].trainIdx + monoRight].octave];
            float depth = static_cast<KannalaBrandt8*>(mpCamera)->TriangulateMatches(mpCamera2,mvKeys[(*it)[0].queryIdx + monoLeft],mvKeysRight[(*it)[0].trainIdx + monoRight],mRlr,mtlr,sigma1,sigma2,p3D);
            if(depth > 0.0001f){
                mvLeftToRightMatch[(*it)[0].queryIdx + monoLeft] = (*it)[0].trainIdx + monoRight;
                mvRightToLeftMatch[(*it)[0].trainIdx + monoRight] = (*it)[0].queryIdx + monoLeft;
                mvStereo3Dpoints[(*it)[0].queryIdx + monoLeft] = p3D;
                mvDepth[(*it)[0].queryIdx + monoLeft] = depth;
                nMatches++;
            }
        }
    }
}

bool Frame::isInFrustumChecks(MapPoint *pMP, float viewingCosLimit, bool bRight) {
    // 3D in absolute coordinates
    Eigen::Vector3f P = pMP->GetWorldPos();

    Eigen::Matrix3f mR;
    Eigen::Vector3f mt, twc;
    if(bRight){
        Eigen::Matrix3f Rrl = mTrl.rotationMatrix();
        Eigen::Vector3f trl = mTrl.translation();
        mR = Rrl * mRcw;
        mt = Rrl * mtcw + trl;
        twc = mRwc * mTlr.translation() + mOw;
    }
    else{
        mR = mRcw;
        mt = mtcw;
        twc = mOw;
    }

    // 3D in camera coordinates
    Eigen::Vector3f Pc = mR * P + mt;
    const float Pc_dist = Pc.norm();
    const float &PcZ = Pc(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    Eigen::Vector2f uv;
    if(bRight) uv = mpCamera2->project(Pc);
    else uv = mpCamera->project(Pc);

    if(uv(0)<mnMinX || uv(0)>mnMaxX)
        return false;
    if(uv(1)<mnMinY || uv(1)>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const Eigen::Vector3f PO = P - twc;
    const float dist = PO.norm();

    if(dist<minDistance || dist>maxDistance)
        return false;

    // Check viewing angle
    Eigen::Vector3f Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn) / dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    if(bRight){
        pMP->mTrackProjXR = uv(0);
        pMP->mTrackProjYR = uv(1);
        pMP->mnTrackScaleLevelR= nPredictedLevel;
        pMP->mTrackViewCosR = viewCos;
        pMP->mTrackDepthR = Pc_dist;
    }
    else{
        pMP->mTrackProjX = uv(0);
        pMP->mTrackProjY = uv(1);
        pMP->mnTrackScaleLevel= nPredictedLevel;
        pMP->mTrackViewCos = viewCos;
        pMP->mTrackDepth = Pc_dist;
    }

    return true;
}

Eigen::Vector3f Frame::UnprojectStereoFishEye(const int &i){
    return mRwc * mvStereo3Dpoints[i] + mOw;
}

} //namespace ORB_SLAM
