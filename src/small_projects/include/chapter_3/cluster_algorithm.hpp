#include <chapter_3/cluster_algorithm.h>
#include <random>
#include <ctime>

void ClusterAlgorithm::cluster_KMeans()
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud = ClusterAlgorithm::cloud_;
    int cluster_num = ClusterAlgorithm::cluster_num_;
    int max_iterations = ClusterAlgorithm::max_iterations_;
    int size = cloud_->size();
    int dim = 3;

    // random select K (cluster_num) points as the intial position of clusters 
    std::vector<Eigen::Vector3f> cluster_centroid;
    Eigen::Vector3f temp_point;
    int rand_index;
    std::default_random_engine e;
    for (int cluster_index = 0; cluster_index < cluster_num; ++cluster_index)
    {
        rand_index = std::rand() % size;
        std::cout << rand_index << std::endl;
        temp_point(0) = cloud->points.at(rand_index).x;
        temp_point(1) = cloud->points.at(rand_index).y;
        temp_point(2) = cloud->points.at(rand_index).z;
        cluster_centroid.push_back(temp_point);
    }
    
    // interation loop
    std::vector<int> cluster_indexes(size, -1);
    std::vector<float> distance(cluster_num, -1.0); // For one point, the distance btw the point and the cluster_centroid
    int iteration = 0;
    while (iteration < max_iterations)
    {
        // E step: calculate which cluster does each point belong to
        for (int point_index = 0; point_index < size; ++point_index)
        {
            for (int cluster_index = 0; cluster_index < cluster_num; ++cluster_index)
            {
                float x_diff = cluster_centroid.at(cluster_index)(0) - cloud->points.at(point_index).x;
                float y_diff = cluster_centroid.at(cluster_index)(1) - cloud->points.at(point_index).y;
                float z_diff = cluster_centroid.at(cluster_index)(2) - cloud->points.at(point_index).z;
                distance.at(cluster_index) = sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
            }
            cluster_indexes.at(point_index) = (min_element(distance.begin(), distance.end())) - distance.begin(); // the cluster which has the minium distance with point
        }

        // M step: update the centroid of each cluster
        std::vector<Eigen::Vector3f> cluster_centroid_sum(cluster_num, Eigen::Vector3f::Zero());
        std::vector<int> cluster_sum_num(cluster_num, 0);
        for (int point_index = 0; point_index < size; ++point_index)
        {
            int cluster_index = cluster_indexes.at(point_index);
            cluster_centroid_sum.at(cluster_index)(0) += cloud->points.at(point_index).x; 
            cluster_centroid_sum.at(cluster_index)(1) += cloud->points.at(point_index).y;
            cluster_centroid_sum.at(cluster_index)(2) += cloud->points.at(point_index).z;
            cluster_sum_num.at(cluster_index) += 1; // add 1 at the corresponding cluster
        }

        for (int cluster_index = 0; cluster_index < cluster_num; ++cluster_index)
        {
            cluster_centroid.at(cluster_index)(0) = cluster_centroid_sum.at(cluster_index)(0) / cluster_sum_num.at(cluster_index);
            cluster_centroid.at(cluster_index)(1) = cluster_centroid_sum.at(cluster_index)(1) / cluster_sum_num.at(cluster_index);
            cluster_centroid.at(cluster_index)(2) = cluster_centroid_sum.at(cluster_index)(2) / cluster_sum_num.at(cluster_index);
        }

        ++iteration;
    }

    // store the cluster result
    ClusterAlgorithm::cluster_indices_->resize(cluster_num);
    for (int point_index = 0; point_index < size; ++point_index)
    {
        int cluster_index = cluster_indexes.at(point_index);
        ClusterAlgorithm::cluster_indices_->at(cluster_index).indices.push_back(point_index);
    }
} 

void ClusterAlgorithm::cluster_GMM()
{
    
} 

void ClusterAlgorithm::cluster_spectral_clustring()
{
    
} 