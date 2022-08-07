#pragma once
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <string>
#include <fstream>
#include <iostream>
#include <vector>

class ModelNet40Reader
{
public:
    ModelNet40Reader() = default;
    ~ModelNet40Reader() = default;
    void setInputFilePath(const std::string& file_path)
    {   file_path_ = file_path; }

    void setOutputFilePath(const std::string& pcd_file_path)
    {   pcd_file_path_ = pcd_file_path;}

    void setPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr output_cloud)
    {   output_cloud_ = output_cloud; }

    void getPointCloud(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr output_cloud)
    {   output_cloud = output_cloud_; }

    void read()
    {
        infile_.open(file_path_);
        if (infile_.is_open())
        {
            std::string line;
            std::string temp;
            std::vector<float> value(6); // 用来存储每一行的6个浮点数
            output_cloud_->width = 10000;
            output_cloud_->height = 1;
            output_cloud_->is_dense = false;
            output_cloud_->points.resize(output_cloud_->width * output_cloud_->height);
            for (size_t point_index = 0; point_index < 10000; ++point_index)
            {
                getline(infile_, line);
                std::stringstream ss(line);
                for (size_t i = 0; i < 6; ++i)
                {
                    getline(ss, temp, ',');
                    value.at(i) = std::stof(temp); // 每行通过逗号分隔符提取小字符串，将小字符串转化为 float 变量，存入 value 中
                }
                output_cloud_->points[point_index].x = value.at(0); // 给x坐标赋值
                output_cloud_->points[point_index].y = value.at(1); // 给y坐标赋值
                output_cloud_->points[point_index].z = value.at(2); // 给z坐标赋值
                output_cloud_->points[point_index].r = static_cast<uint8_t>(std::floor(128*(value.at(3)+1))); // 给r颜色赋值
                output_cloud_->points[point_index].g = static_cast<uint8_t>(std::floor(128*(value.at(4)+1))); // 给g颜色赋值
                output_cloud_->points[point_index].b = static_cast<uint8_t>(std::floor(128*(value.at(5)+1))); // 给b颜色赋值
            }
            infile_.close();
        }
        else
        {
            std::cout << "Could not open file " << file_path_ << std::endl;
        }
    }

    void save()
    {   pcl::io::savePCDFileASCII(pcd_file_path_, *output_cloud_); }

private:
    std::string file_path_;
    std::string pcd_file_path_;
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr output_cloud_;
    std::ifstream infile_;
};
