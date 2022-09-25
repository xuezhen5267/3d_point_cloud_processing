#include "chapter_1/modelnet40_reader.h"
#include "chapter_1/basic_algorithm.h"
#include "chapter_7/feature_detection.h"
#include "chapter_7/feature_detection.hpp"

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/pcl_plotter.h>
#include <pcl/common/common.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/integral_image_normal.h>
#include <boost/thread/thread.hpp>


int main(int argc, char* argv[])
{
    // Read the ModelNet40 txt file
    std::string file_path = argv[1];
    // std::string file_path = "data/modelnet40_normal_resampled/monitor/monitor_0001.txt";
    // std::string pcd_file_path = argv[2];
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    ModelNet40Reader reader;
    reader.setInputFilePath(file_path);
    reader.setPointCloud(cloud);
    reader.read();
    reader.getPointCloud(cloud); // the raw point cloud

    // Feature Detection
    pcl::PointXYZRGBA minpt, maxpt;
    pcl::getMinMax3D(*cloud, minpt, maxpt);
    float longest_dist = std::sqrt((maxpt.x - minpt.x) * (maxpt.x - minpt.x) + (maxpt.y - minpt.y) * (maxpt.y - minpt.y) + (maxpt.z - minpt.z) * (maxpt.z - minpt.z));
    std::cout << "Longest distance = " << longest_dist << std::endl;
    
    // Feature Detection
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGBA> octree(0.001f);
    boost::shared_ptr<pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGBA>> octree_ptr = boost::make_shared<pcl::octree::OctreePointCloudSearch<pcl::PointXYZRGBA>>(octree);
    FeatureDetection fd; // chapter 7
    pcl::PointIndices::Ptr key_point_indices (new pcl::PointIndices);
    fd.setInputPointCloud(cloud);
    fd.setOctree(octree_ptr);
    fd.setFeatureDetectionMethod("ISS");
    fd.setRnnRadius(longest_dist / 20);
    fd.setNMSRadius(longest_dist / 10);
    fd.execDetection();
    fd.getKeyPointIndices(key_point_indices);

    // estimate the normals
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);
    BasicAlgorithm ba; // chapter 1
    ba.setInputPointCloud(cloud);
    ba.setOutputNormals(normals);
    ba.estimateNormal(longest_dist / 20);
    ba.getNormals(normals);
    
    // calculate the FPFH using PCL
    pcl::search::KdTree<pcl::PointXYZRGBA>::Ptr tree_ptr(new pcl::search::KdTree<pcl::PointXYZRGBA>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfhs_ptr(new pcl::PointCloud<pcl::FPFHSignature33>());
    pcl::FPFHEstimation<pcl::PointXYZRGBA, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    fpfh.setIndices(key_point_indices);
    fpfh.setSearchMethod(tree_ptr);
    fpfh.setRadiusSearch(longest_dist / 20);
    fpfh.compute(*fpfhs_ptr);



    // Build the point cloud with key point indices
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr key_cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    key_cloud->width = key_point_indices->indices.size();
    key_cloud->height = 1;
    key_cloud->points.resize(key_cloud->width * key_cloud->height);
    std::cout << "key points indices: ";
    for (int point_index = 0; point_index < key_cloud->width; ++point_index)
    {
        key_cloud->points.at(point_index).x = cloud->at(key_point_indices->indices.at(point_index)).x;
        key_cloud->points.at(point_index).y = cloud->at(key_point_indices->indices.at(point_index)).y;
        key_cloud->points.at(point_index).z = cloud->at(key_point_indices->indices.at(point_index)).z;
        key_cloud->points.at(point_index).r = 0; // set color is black
        key_cloud->points.at(point_index).g = 0;
        key_cloud->points.at(point_index).b = 0;
        std::cout << key_point_indices->indices.at(point_index) << ", ";
    }
    std::cout << std::endl;
    pcl::io::savePCDFileASCII("data/modelnet40_normal_resampled/test/test_0001.pcd", *key_cloud);

    // point cloud visuliation 
    boost::shared_ptr<pcl::visualization::PCLVisualizer> data_viewer (new pcl::visualization::PCLVisualizer("data_viewer"));
    data_viewer->setBackgroundColor(1,1,1);
    // data_viewer->addCoordinateSystem();
    data_viewer->addPointCloud<pcl::PointXYZRGBA> (cloud, "raw_cloud"); // visulize downsampled point cloud
    data_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "raw_cloud");
    data_viewer->addPointCloud<pcl::PointXYZRGBA> (key_cloud, "key_cloud"); // visulize downsampled point cloud
    data_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 20, "key_cloud");
    // data_viewer->addPointCloudNormals <pcl::PointXYZRGBA, pcl::Normal> (cloud, normals, 1, 0.05, "normals");

    // FPFH visulization
    pcl::visualization::PCLPlotter::Ptr plotter (new pcl::visualization::PCLPlotter("FPFH plotter"));
    plotter->setShowLegend(true);
    // plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",0, "FPFH of 1st key point");
    // plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",1, "FPFH of 2ed key point");
    // plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",2, "FPFH of 3rd key point");
    // plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",3, "FPFH of 4th key point");
    // plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",4, "FPFH of 5th key point");
    // plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",5, "FPFH of 6th key point");
    // plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",6, "FPFH of 7th key point");
    // plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",7, "FPFH of 8th key point");
    // plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",8, "FPFH of 9th key point");
    // plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",9, "FPFH of 10th key point");
    // plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",10, "FPFH of 11th key point");
    plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",11, "FPFH of 12th key point");
    plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",12, "FPFH of 13th key point");
    // plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",13, "FPFH of 14th key point");
    // plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfhs_ptr, "fpfh",14, "FPFH of 15th key point");
    plotter->spinOnce(100);

    while (!data_viewer->wasStopped()) // keep viewer
    {
        data_viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    return 0;
}