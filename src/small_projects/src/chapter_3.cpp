#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

#include <chapter_3/cluster_algorithm.h>
#include <chapter_3/cluster_algorithm.hpp>

int main(int argc, char* argv[])
{
    // read pcd file
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
    pcl::io::loadPCDFile<pcl::PointXYZRGBA> (argv[1], *cloud);
    //pcl::io::loadPCDFile<pcl::PointXYZRGBA> ("data/cluster_data/noisy_moons.pcd", *cloud);

    // cluster
    ClusterAlgorithm cluster;
    boost::shared_ptr<std::vector<pcl::PointIndices>> cluster_indeces(new std::vector<pcl::PointIndices>);
    int cluster_num = std::stoi(argv[3]);
    int max_iterations = 10;
    cluster.setInputPointCloud(cloud);
    cluster.setClusterMethod(argv[2]); // "KMeans", "GMM", or "Spectral_Clustering"
    cluster.setClusterNumber(cluster_num);
    cluster.setMaxIterations(max_iterations);
    cluster.execCluster();
    cluster.getClusterResult(cluster_indeces);

    // Add color
    const int colors[][3] = 
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
    for ( int cluster_index = 0; cluster_index < cluster_num; ++cluster_index) // go over all the cluster
    {
        pcl::PointIndices point_indices = cluster_indeces->at(cluster_index); // go over all the indices in oine cluster
        uint32_t rand_rgba = std::rand();
        for (int indice_index = 0; indice_index < point_indices.indices.size(); ++indice_index )
        {
            cloud->points.at(point_indices.indices.at(indice_index)).r = (uint8_t)colors[cluster_index % 10][0];
            cloud->points.at(point_indices.indices.at(indice_index)).g = (uint8_t)colors[cluster_index % 10][1];
            cloud->points.at(point_indices.indices.at(indice_index)).b = (uint8_t)colors[cluster_index % 10][2];
        }
    }
    // visualization
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("cluster_viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZRGBA> (cloud, "point_cloud"); // visulize raw point cloud
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "point_cloud");
    while (!viewer->wasStopped()) // keep viewer
    {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }

    return 0;
}