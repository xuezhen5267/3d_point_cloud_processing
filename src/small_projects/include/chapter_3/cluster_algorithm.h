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
        {
            std::vector<Eigen::VectorXf> km_vectors;
            Eigen::VectorXf temp_vector3f;
            temp_vector3f.resize(3);
            for (int point_index = 0; point_index < cloud_->size(); ++point_index)
            {
                temp_vector3f(0) = cloud_->points.at(point_index).x;
                temp_vector3f(1) = cloud_->points.at(point_index).y;
                temp_vector3f(2) = cloud_->points.at(point_index).z;
                km_vectors.push_back(temp_vector3f); // calculate km_vectors
            }
            cluster_KMeans_vectors(km_vectors); // KMeans algorithm on km_vectors
        }
        else if (cluster_method_ == "GMM")
        {
            cluster_GMM();
        }
        else if (cluster_method_ == "Spectral_Clustering")
        {
            std::vector<Eigen::VectorXf> km_vectors;
            calculate_KM_input(km_vectors); // calculate km_vectors
            cluster_KMeans_vectors(km_vectors); // KMeans algorithm on km_vectors
        }
        else
        {
            std::cout << "Input method is invalid!" << std::endl;
        }
    }

    void getClusterResult(boost::shared_ptr<std::vector<pcl::PointIndices>>& cluster_indices)
    {
        cluster_indices = cluster_indices_;
    }

private:
    void cluster_KMeans_3dPointCloud();
    void cluster_KMeans_vectors(std::vector<Eigen::VectorXf>& km_vectors); // used for spectral clustering
    void cluster_GMM();
    void cluster_spectral_clustring();
    void calculate_KM_input(std::vector<Eigen::VectorXf>& km_vectors);
    float gaussian_distribution(Eigen::Vector3f x, Eigen::Vector3f mu, Eigen::Matrix3f sigma);
    void sortEigenvectorsByEigenvalues(Eigen::MatrixXf& eigenvalues, Eigen::MatrixXf& eigenvectors);
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_;
    int cluster_num_;
    boost::shared_ptr<std::vector<pcl::PointIndices>> cluster_indices_ = boost::make_shared<std::vector<pcl::PointIndices>>();
    std::string cluster_method_;
    int max_iterations_;
};