#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include "chapter_4/model_fit_algorithm.h"

void callback(const sensor_msgs::PointCloud2ConstPtr& input_msg_pc2, ModelFitAlgorithem& model_fit, ros::Publisher& ground_pub, ros::Publisher& objects_pub)
{
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr input_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::fromROSMsg(*input_msg_pc2, *input_cloud);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::PointIndices::Ptr outliers(new pcl::PointIndices);
    model_fit.setInputPointCloud(input_cloud); // put in the point cloud
    model_fit.execFit();
    model_fit.getInliers(inliers);
    model_fit.getOutliers(outliers);

    // publish the inliers (ground point cloud)
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr ground_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    ground_cloud->width = inliers->indices.size();
    ground_cloud->height = 1;
    ground_cloud->is_dense = input_cloud->is_dense;
    ground_cloud->points.resize(ground_cloud->width * ground_cloud->height);
    for (int inlier_index = 0; inlier_index < inliers->indices.size(); ++inlier_index)
    {
        ground_cloud->points.at(inlier_index) = input_cloud->points.at(inliers->indices.at(inlier_index));
        ground_cloud->points.at(inlier_index).b = static_cast<uint8_t>(255); //set the color of ground
    }
    sensor_msgs::PointCloud2 ground_msg_pc2;
    pcl::toROSMsg(*ground_cloud, ground_msg_pc2);
    ground_msg_pc2.header.frame_id = "velodyne";
    ground_pub.publish(ground_msg_pc2);

    // publish the outliers (objects point cloud)
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr objects_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    objects_cloud->width = outliers->indices.size();
    objects_cloud->height = 1;
    objects_cloud->is_dense = input_cloud->is_dense;
    objects_cloud->points.resize(objects_cloud->width * objects_cloud->height);
    for (int outlier_index = 0; outlier_index < outliers->indices.size(); ++outlier_index)
    {
        objects_cloud->points.at(outlier_index) = input_cloud->points.at(outliers->indices.at(outlier_index));
    }


    sensor_msgs::PointCloud2 objects_msg_pc2;
    pcl::toROSMsg(*objects_cloud, objects_msg_pc2);
    objects_msg_pc2.header.frame_id = "velodyne";
    objects_pub.publish(objects_msg_pc2);
}

int main(int argc, char* argv[])
{
    ros::init(argc, argv, "ground_removal");
    ros::NodeHandle nh;

    // setup the model_fit object, which is used for fit a plane based on input point cloud
    ModelFitAlgorithem model_fit;
    ModelParameters model_paras;
    model_fit.setFitMethod("RANSAC");
    model_fit.setInlierThreshod(0.15);
    model_fit.setMaxIteration(500);
    model_fit.setInputModelParameters(model_paras);
    model_fit.setInputModel("plane");
    std::cout << "begin subscribe" << std::endl;
    ros::Publisher ground_pub = nh.advertise<sensor_msgs::PointCloud2>("/ground", 1);
    ros::Publisher objects_pub = nh.advertise<sensor_msgs::PointCloud2>("/objects_w/o_cluster", 1);
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne", 10, boost::bind(&callback, _1, model_fit, ground_pub, objects_pub));
    std::cout << "end subscribe" << std::endl;
    ros::spin();

    return 0;
}
