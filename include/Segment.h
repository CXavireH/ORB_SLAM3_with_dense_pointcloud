/*
Sam-SLAM
*/

#ifndef SEGMENT_H
#define SEGMENT_H


#include "KeyFrame.h"
#include "Atlas.h"
#include "Tracking.h"
// #include "libsegmentation.hpp" 
#include "sam.h"
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <mutex>
#include <condition_variable>

namespace ORB_SLAM3
{
class Tracking;
class Segment
{

public:
    Segment();
    void SetTracker(Tracking* pTracker);
    void Run();



    // cv::Mat label_colours;
    // Classifier* classifier;
    bool isNewImgArrived();
    bool CheckFinish();
    void RequestFinish();
    // void Initialize(const cv::Mat& img);
    cv::Mat mImg;
    cv::Mat mImgTemp;
    cv::Mat mImgSegment_color;
    cv::Mat mImgSegment_color_final;
    cv::Mat mImgSegment;
    cv::Mat mImgSegmentLatest;
    Tracking* mpTracker;
    std::mutex mMutexGetNewImg;
    std::mutex mMutexFinish;
    bool mbFinishRequested;
    void ProduceImgSegment();
    std::mutex mMutexNewImgSegment;
    std::condition_variable mbcvNewImgSegment;
    bool mbNewImgFlag;
    int mSkipIndex;
    double mSegmentTime;
    int imgIndex;
    Sam::Parameter param;
};

}



#endif
