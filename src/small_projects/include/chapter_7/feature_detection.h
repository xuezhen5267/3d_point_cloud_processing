#pragma once
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree_search.h>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>

class FeatureDetection
{
public:
    FeatureDetection() = default;
    ~FeatureDetection() = default;

    void setInputPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
    {   cloud_ = cloud; }

    void setFeatureDetectionMethod(std::string method)
    {   method_ = method; }

    void setOctree(pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGBA>::Ptr octree)
    {   
        octree_ = octree;
        octree_->setInputCloud(cloud_);
        octree_->addPointsFromInputCloud();
    }

    void setRnnRadius(float rnn_radius)
    {   rnn_radius_ = rnn_radius; }

    void execDetection()
    {   
        if (method_ == "ISS")
        execISS();
    }

    void execISS();

    void setNMSRadius(float nms_radius)
    {   nms_radius_ = nms_radius; }
    
    void NMS(std::list<std::pair<int, float>>& input_points, std::vector<std::pair<int, float>>& output_points);

    void getKeyPointIndices( std::vector<int>& key_point_indices)
    {  
        key_point_indices.clear();
        for (int point_index = 0; point_index < key_points_after_NMS_.size(); ++point_index)
        {
            key_point_indices.push_back(key_points_after_NMS_.at(point_index).first);
        }
    }

private:
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_;
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGBA>::Ptr octree_;
    std::string method_;
    float rnn_radius_;
    float nms_radius_;
    std::list<std::pair<int, float>> key_points_after_filter_;
    std::vector<std::pair<int, float>> key_points_after_NMS_;
};