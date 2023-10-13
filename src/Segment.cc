#include "Segment.h"
#include "Tracking.h"
#include <fstream>
#include "sam.h"
#include <thread>
#define SKIP_NUMBER 1
using namespace std;

namespace ORB_SLAM3
{
Segment::Segment()
        :mbFinishRequested(false),
         mSkipIndex(SKIP_NUMBER),
         mSegmentTime(0),
         imgIndex(0),
         param("models/sam_preprocess.onnx", "models/sam_vit_h_4b8939.onnx", std::thread::hardware_concurrency())
{
    param.providers[0].deviceType = 0;
    mImgSegmentLatest = cv::Mat(mImg.rows, mImg.cols, CV_64FC1);
    mbNewImgFlag = false;
}

void Segment::SetTracker(Tracking *pTracker)
{
    mpTracker = pTracker;
}

bool Segment::isNewImgArrived()
{
    unique_lock<mutex> lock(mMutexGetNewImg);
    if (mbNewImgFlag)
    {
        mbNewImgFlag = false;
        return true;
    }
    else
        return false;
}

void Segment::Run()
{
    Sam sam(param);
    std::cout << "load the model!" << std::endl;

    while (1)
    {

        usleep(1);
        if (!isNewImgArrived())
            continue;

        cout << "Wait for new RGB img time =" << endl;
        if (mSkipIndex == SKIP_NUMBER)
        {
            std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
            // Recognise by Semantin segmentation
            auto inputSize = sam.getInputSize();
            cv::Size imageSize = mImg.size();
            cv::resize(mImg, mImg, inputSize);
            sam.loadImage(mImg);
            mImgSegment = sam.autoSegment({10, 10});
            cv::resize(mImgSegment, mImgSegment, imageSize);
            std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
            mSegmentTime += std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3).count();
            mSkipIndex = 0;
            imgIndex++;
        }
        mSkipIndex++;
        ProduceImgSegment();
        if (CheckFinish())
        {
            break;
        }
    }
}

bool Segment::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Segment::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

void Segment::ProduceImgSegment()
{
    std::unique_lock<std::mutex> lock(mMutexNewImgSegment);
    mImgTemp = mImgSegmentLatest;
    mImgSegmentLatest = mImgSegment;
    mImgSegment = mImgTemp;
    mpTracker->mbNewSegImgFlag = true;
}

} // namespace ORB_SLAM3
