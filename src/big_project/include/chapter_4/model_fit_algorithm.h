#pragma once
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <string>
#include <vector>
#include <Eigen/Dense>
#include <ctime>
#include <cstdlib>

struct ModelParameters
{
    float para_1 = 0; // a for plane model Ax + By + Cz + D = 0
    float para_2 = 0; // b for plane model Ax + By + Cz + D = 0
    float para_3 = 0; // c for plane model Ax + By + Cz + D = 0
};

class ModelFitAlgorithem
{
public:
    ModelFitAlgorithem() = default;
    ~ModelFitAlgorithem() = default;

    void setInputPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud)
    {   cloud_ = cloud; }

    void setInputModelParameters(ModelParameters& model_paras)
    {   model_paras_ = model_paras; }
    
    void setInputModel(std::string model_name) // "plane"
    {   model_name_ = model_name;   }

    void setMaxIteration(int max_iteration)
    {   max_iteration_ = max_iteration; }

    void setFitMethod(std::string method) // "RANSAC"
    {   method_ = method; }

    void setInlierThreshod(float threshold)
    {
        threshold_ = threshold;
    }

    void execFit()
    {
        if (method_ == "RANSAC")
        {
            ransacFit();
        }
    }

    void getInliers(pcl::PointIndices::Ptr& inliers)
    {  inliers = inliers_; }

    void getOutliers(pcl::PointIndices::Ptr& outliers)
    {  outliers = outliers_; }

private:
    float calculateDistance(pcl::PointXYZRGBA& point, ModelParameters& model_paras)
    {
        float d = 0;
        if (model_name_ == "plane")
        {
            float a = model_paras.para_1;
            float b = model_paras.para_2;
            float c = model_paras.para_3;
            float d = 1;
            float x = point.x;
            float y = point.y;
            float z = point.z;
            d = abs(a * x + b * y + c * z + d) / sqrt(a * a + b * b + c * c);
            return d;
        }
        return 0;
    }

    void calculateModelParas(std::vector<pcl::PointXYZRGBA>& points, ModelParameters& model_paras)
    {
        if (model_name_ == "plane")
        {
            pcl::PointXYZRGBA p1, p2, p3;
            p1 = points.at(0);
            p2 = points.at(1);
            p3 = points.at(2);
            Eigen::Matrix3f A_matrix;
            A_matrix << p1.x, p1.y, p1.z, p2.x, p2.y, p2.z, p3.x, p3.y, p3.z; // solve Ax = b
            Eigen::Vector3f b_vector;
            b_vector << -1, -1, -1;
            Eigen::Vector3f x_vector = A_matrix.inverse() * b_vector;

            model_paras.para_1 = x_vector(0);
            model_paras.para_2 = x_vector(1);
            model_paras.para_3 = x_vector(2);
        }
    }

    void ransacFit()
    {
        int latest_inliers_num = 0;
        int iteration = 0;
        srand(time(0));
        int rand_index;
        ModelParameters current_model_paras;
        while (iteration < max_iteration_)
        {
            // rand 3 points
            std::vector<pcl::PointXYZRGBA> points(3);
            for (int point_index = 0; point_index < 3; ++point_index)
            {
                rand_index = std::rand() % cloud_->size(); // generate a random number
                points.at(point_index) = cloud_->points.at(rand_index); // save the rand point in points
            }
            
            // update model parameters
            calculateModelParas(points, current_model_paras);
            
            // calculate the inliers number
            int current_inliers_num = 0;
            for (int point_index = 0; point_index < cloud_->size(); ++point_index)
            {
                float distance = calculateDistance(cloud_->points.at(point_index), current_model_paras);
                if (distance < threshold_)
                {
                    ++current_inliers_num;
                }
            }

            // update the model parameters or not
            if (current_inliers_num > latest_inliers_num)
            {
                latest_inliers_num = current_inliers_num;
                model_paras_ = current_model_paras;
            }
            ++iteration;
        }

        inliers_->indices.clear();
        outliers_->indices.clear();
        for (int point_index = 0; point_index < cloud_->size(); ++point_index)
        {
                if (calculateDistance(cloud_->points.at(point_index), model_paras_) < threshold_ * 1.5)
                {
                    inliers_->indices.push_back(point_index);
                }
                else
                {
                    outliers_->indices.push_back(point_index);
                }
        }
    }

    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud_;
    ModelParameters model_paras_;
    std::string model_name_;
    int max_iteration_;
    std::string method_;
    pcl::PointIndices::Ptr inliers_ = boost::make_shared<pcl::PointIndices>();
    pcl::PointIndices::Ptr outliers_ = boost::make_shared<pcl::PointIndices>();
    float threshold_;

};