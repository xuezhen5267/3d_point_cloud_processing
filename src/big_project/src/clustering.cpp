#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <chapter_4/dbscan_clustering.h>
#include <vector>

void callback(const sensor_msgs::PointCloud2ConstPtr& input_msg_pc2, DBSCANSimpleCluster<pcl::PointXYZRGBA>& dbscan, ros::Publisher& pub)
{
    ROS_INFO("Subscribe /objects_w/o_cluster successfully!" );
    // transfer msg to pcl point cloud
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr input_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::fromROSMsg(*input_msg_pc2, *input_cloud);
    std::vector<pcl::PointIndices> cluster_indices;

    // setup the search tree
    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBA>);
    tree->setInputCloud(input_cloud);
    // setup the dbscan
    dbscan.setSearchMethod(tree);
    dbscan.setInputCloud(input_cloud); 
    dbscan.extract(cluster_indices);

    // set the color
    const int colors[10][3] = 
    {
	    255, 0,   0,	// red 			1
	    0,   255, 0,	// green		2
	    0,   0,   255,	// blue			3
	    255, 255, 0,	// yellow		4
	    0,   255, 255,	// light blue	5
	    255, 0,   255,	// magenta		6
	    255, 255, 255,	// white		7
	    255, 128, 0,	// orange		8
	    255, 153, 255,	// pink			9
	    155, 48, 255,	// purple		10
    };

    for (int cluster_index = 0; cluster_index < cluster_indices.size(); ++cluster_index)
    {
        for(int indice_index = 0; indice_index < cluster_indices.at(cluster_index).indices.size(); ++indice_index)
        {
            int point_index = cluster_indices.at(cluster_index).indices.at(indice_index);
            input_cloud->points.at(point_index).r = (uint8_t)colors[cluster_index % 10][0];
            input_cloud->points.at(point_index).g = (uint8_t)colors[cluster_index % 10][1];
            input_cloud->points.at(point_index).b = (uint8_t)colors[cluster_index % 10][2];
        }
    }

    // publish the colored point cloud
    sensor_msgs::PointCloud2 clustered_msg_pc2;
    pcl::toROSMsg(*input_cloud, clustered_msg_pc2);
    clustered_msg_pc2.header.frame_id = "velodyne";
    pub.publish(clustered_msg_pc2);

}


int main(int argc, char* argv[])
{
    ros::init(argc, argv, "clustering");
    ros::NodeHandle nh;

    DBSCANSimpleCluster<pcl::PointXYZRGBA> dbscan; // 自定义的类
    dbscan.setCorePointMinPts(5);
    dbscan.setClusterTolerance(1);
    dbscan.setMinClusterSize(5);
    dbscan.setMaxClusterSize(1000);

    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("/objects_with_cluster", 1);
    ros::Subscriber sub = nh.subscribe<sensor_msgs::PointCloud2>("/objects_w/o_cluster", 10, boost::bind(&callback, _1, dbscan, pub));
    ros::spin();

    return 0;
}