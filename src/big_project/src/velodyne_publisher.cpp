#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <chapter_1/basic_algorithm.h>

int main(int argc, char* argv[])
{
  // read pcd processing
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr raw_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
  pcl::io::loadPCDFile<pcl::PointXYZRGBA> (argv[1], *raw_cloud);
  BasicAlgorithm basic_algorithm;
  float voxel_size = 0.2;
  basic_algorithm.setInputPointCloud(raw_cloud);
  basic_algorithm.voxelGridDownSample(voxel_size, "random");
  basic_algorithm.getDownSampledCloud(downsampled_cloud);
  

  // ros related
  ros::init(argc, argv, "velodyne_publisher");
  ros::NodeHandle nh;
  ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("/velodyne", 1);

  // generate PointClud2 message
  sensor_msgs::PointCloud2 msg_pc2;
  pcl::toROSMsg(*downsampled_cloud, msg_pc2);
  msg_pc2.header.frame_id = "velodyne";

  // loop for publisher
  ros::Rate loop_rate(10);
  while (ros::ok())
  {
    pub.publish(msg_pc2);
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}