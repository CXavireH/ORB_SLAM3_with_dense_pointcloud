#ifndef POINTCLOUDE_H
#define POINTCLOUDE_H

#include "pointcloudmapping.h"
//#include "Frame.h"
//#include "Map.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <condition_variable>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <opencv2/core/core.hpp>
#include <mutex>

namespace ORB_SLAM3
{

class PointCloude
{
    typedef pcl::PointXYZRGBA PointT;
    typedef pcl::PointCloud<PointT> PointCloud;
public:
    PointCloud::Ptr pcE;
    Eigen::Isometry3d T;
    int pcID;  
};

}

#endif // POINTCLOUDE_H
