#pragma once
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>


class ClusterAlgorithm
{
public:
    ClusterAlgorithm() = default;
    ~ClusterAlgorithm() = default;

    void setInputPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
    {   cloud_ = cloud; }

    void setClusterNumber(int cluster_num)
    {   cluster_num_ = cluster_num; }

    void setClusterMethod(std::string cluster_method)
    {   cluster_method_ = cluster_method; }
    
    void setMaxIterations(int max_iterations)
    {   max_iterations_ = max_iterations; }

    void execCluster()
    {
        if (cluster_method_ == "KMeans")
            cluster_KMeans();
        else if (cluster_method_ == "GMM")
            cluster_GMM();
        else if (cluster_method_ == "Spectral_Clustering")
            cluster_spectral_clustring();
    }

    void getClusterResult(boost::shared_ptr<std::vector<pcl::PointIndices>>& cluster_indices)
    {
        cluster_indices = cluster_indices_;
    }

private:
    void cluster_KMeans();
    void cluster_GMM();
    void cluster_spectral_clustring();
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_;
    int cluster_num_;
    boost::shared_ptr<std::vector<pcl::PointIndices>> cluster_indices_ = boost::make_shared<std::vector<pcl::PointIndices>>();
    std::string cluster_method_;
    int max_iterations_;
};