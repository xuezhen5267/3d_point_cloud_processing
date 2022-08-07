// this cpp file is used to test the class BasicAlgorithm for 3D point cloud processing
#include "chapter_1/modelnet40_reader.h"
#include "chapter_1/basic_algorithm.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

int main(int argc, char* argv[])
{
    // Read the ModelNet40 txt file
    std::string file_path = argv[1];
    // std::string file_path = "data/modelnet40_normal_resampled/sofa/sofa_0004.txt";
    // std::string pcd_file_path = argv[2];
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGBA>);
    ModelNet40Reader reader;
    reader.setInputFilePath(file_path);
    reader.setPointCloud(cloud);
    reader.read();
    reader.getPointCloud(cloud); // the raw point cloud
    //reader.setOutputFilePath(pcd_file_path);
    //file_reader.save();

    // PCA
    BasicAlgorithm basic_algorithm;
    Eigen::Matrix3f matrix_u; // used to store the U matrix based on SVD of input x
    basic_algorithm.setInputPointCloud(cloud);
    basic_algorithm.execPCA();
    basic_algorithm.getMatrixU(matrix_u);

    // Estimate normals
    pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>); // used to store the normal info 
    float neighbourhood_radius = 0.1;
    basic_algorithm.setOutputNormals(normals);
    basic_algorithm.estimateNormal(neighbourhood_radius); 
    basic_algorithm.getNormals(normals);
    
    // Voxel grid downsample
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>); // used to store the downsampled point cloud
    float voxel_size = 0.05;
    // basic_algorithm.voxelGridDownSample(0.05, "random"); // voxel grid downsampling based on random method
    basic_algorithm.voxelGridDownSample(voxel_size, "centroid"); // voxel grid downsampling based on centroid method
    basic_algorithm.getDownSampledCloud(downsampled_cloud);

    // visualization
    boost::shared_ptr<pcl::visualization::PCLVisualizer> pca_normal_viewer (new pcl::visualization::PCLVisualizer("pca_viewer"));
    boost::shared_ptr<pcl::visualization::PCLVisualizer> centroid_voxel_grid_viewer (new pcl::visualization::PCLVisualizer("centroid_voxel_grid_viewer"));
    boost::shared_ptr<pcl::visualization::PCLVisualizer> random_voxel_grid_viewer (new pcl::visualization::PCLVisualizer("random_voxel_grid_viewer"));
    
    // visulize PCA and normal results
    pca_normal_viewer->addPointCloud<pcl::PointXYZRGBA> (cloud, "raw_point_cloud"); // visulize raw point cloud
    pca_normal_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "raw_point_cloud");
    Eigen::Vector3f centroid;
    basic_algorithm.getCentroid(centroid);
    pcl::PointXYZ origin(centroid(0), centroid(1), centroid(2));
    pcl::PointXYZ z1(1.5*matrix_u(0,0), 1.5*matrix_u(1,0), 1.5*matrix_u(2,0));
    pcl::PointXYZ z2(1.5*matrix_u(0,1), 1.5*matrix_u(1,1), 1.5*matrix_u(2,1));
    pcl::PointXYZ z3(1.5*matrix_u(0,2), 1.5*matrix_u(1,2), 1.5*matrix_u(2,2));
    pca_normal_viewer->addArrow(z1, origin, 255, 0, 0, false, "z1");
    pca_normal_viewer->addArrow(z2, origin, 0, 255, 0, false, "z2");
    pca_normal_viewer->addArrow(z3, origin, 0, 0, 255, false, "z3");
    pca_normal_viewer->addPointCloudNormals<pcl::PointXYZRGBA, pcl::Normal> (cloud, normals, 5, 0.1, "normals"); // visualize normals
    // visulize voxel grid downsampling results
    centroid_voxel_grid_viewer->addPointCloud<pcl::PointXYZRGBA> (downsampled_cloud, "downsampled_point_cloud"); // visulize downsampled point cloud
    centroid_voxel_grid_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "downsampled_point_cloud");
    basic_algorithm.voxelGridDownSample(voxel_size, "random"); // voxel grid downsampling based on random method
    basic_algorithm.getDownSampledCloud(downsampled_cloud);
    random_voxel_grid_viewer->addPointCloud<pcl::PointXYZRGBA> (downsampled_cloud, "downsampled_point_cloud"); // visulize downsampled point cloud
    random_voxel_grid_viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "downsampled_point_cloud");    

    while (!pca_normal_viewer->wasStopped()) // keep viewer
    {
        pca_normal_viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    return 0;
}