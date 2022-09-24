#pragma once
#include "chapter_7/feature_detection.h"
// #include <Eigen/Core>

bool smaller(std::pair<int, float> p1, std::pair<int, float> p2) // this function is used for sort
{   
    if (p1.second >= p2.second)
        return true;
    else 
        return false;
};

void FeatureDetection::execISS()
{
    int size = cloud_->size();
    key_points_after_filter_.clear();
    for (int point_index = 0; point_index < size; ++point_index) // go over all the points
    {
        // RNN search based on PCL octree
        std::vector<int> points_index_search;
        std::vector<float> points_radius_squared_distance;
        octree_->radiusSearch(cloud_->at(point_index), rnn_radius_, points_index_search, points_radius_squared_distance);
        
        float weight_sum = 0;
        Eigen::Matrix3f cov(Eigen::Matrix3f::Zero());
        Eigen::Vector3f query_point;
        query_point << cloud_->at(point_index).x, cloud_->at(point_index).y, cloud_->at(point_index).z;
        for (int neighboor_index = 0; neighboor_index < points_index_search.size(); ++neighboor_index) // go over all the neightboors
        {
            if (points_index_search.at(neighboor_index) != point_index)
            {
                // Step 1. calculate the weight for each neighboor
                float weight = 1 / std::sqrt(points_radius_squared_distance.at(neighboor_index));
                weight_sum += weight;
                Eigen::Vector3f neighboor_point;
                neighboor_point << cloud_->at(points_index_search.at(neighboor_index)).x, cloud_->at(points_index_search.at(neighboor_index)).y, cloud_->at(points_index_search.at(neighboor_index)).z;            
                cov += weight * (neighboor_point - query_point)*(neighboor_point - query_point).transpose();
            }
        }
        // Step 2. calculate the weighted covariance matrix for each query point
        cov = cov / weight_sum;

        // Step 3. calculate the eigen values of the covariance matrix
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
        std::vector<float> eigen_values = {es.eigenvalues()(0), es.eigenvalues()(1), es.eigenvalues()(2)};
        std::sort(eigen_values.begin(), eigen_values.end()); 

        // Step 4. fitler the key points based on the proposion of eigenvalues
        if (eigen_values.at(2)/eigen_values.at(1) > 2 && eigen_values.at(1)/eigen_values.at(0) > 2)
        {
            std::pair<int, float> point(point_index, eigen_values.at(0)); // the smallest eigen value
            key_points_after_filter_.push_back(point);
        }
    }

    // Step 5. NMS on filterd key points.
    NMS(key_points_after_filter_, key_points_after_NMS_);
};

void FeatureDetection::NMS(std::list<std::pair<int, float>>& input_points, std::vector<std::pair<int, float>>& output_points)
{
    // Step 1. sort the input data
    output_points.clear();
    input_points.sort(smaller);
    while (input_points.size() != 0)
    {
        // Step 2. move the data with max feature into output dataset
        std::pair<int, float> max_point = input_points.front(); 
        output_points.push_back(max_point);
        input_points.pop_front(); // remove the selected data from input dataset

        if (~input_points.empty())
        {
            auto iter = input_points.begin();
            while (iter != input_points.end())
            {
                // Step 3. remove the data if it is closed to the data with max feature
                Eigen::Vector3f max_point_coors, other_point_coors;
                max_point_coors << cloud_->points.at(max_point.first).x, cloud_->points.at(max_point.first).y,cloud_->points.at(max_point.first).z;
                other_point_coors << cloud_->points.at(iter->first).x, cloud_->points.at(iter->first).y, cloud_->points.at(iter->first).z;
                float distance = (max_point_coors - other_point_coors).norm();
                if (distance > nms_radius_) // if far
                {                
                    ++iter; // do not delete, iter point to the next pos
                }
                else // if close
                {
                    iter = input_points.erase(iter); // delete, and iter point to the next pos
                }
            }
        }
    }
};