#include <chapter_3/cluster_algorithm.h>
#include <Eigen/Eigenvalues>
#include <math.h>
#include <ctime>
#include <cstdlib>


#define PI acos(-1)
struct EigenValueAndVector
{
    float eigenvalue;
    Eigen::VectorXf eigenvector;
};

bool less_based_on_eigenvalue(EigenValueAndVector a, EigenValueAndVector b) 
{   
    return a.eigenvalue < b.eigenvalue; 
}

void ClusterAlgorithm::cluster_KMeans_3dPointCloud()
{
    std::vector<Eigen::VectorXf> km_vectors;
    Eigen::VectorXf temp_vector3f;
    temp_vector3f.resize(3);
    for (int point_index = 0; point_index < cloud_->size(); ++point_index)
    {
        temp_vector3f(0) = cloud_->points.at(point_index).x;
        temp_vector3f(1) = cloud_->points.at(point_index).y;
        temp_vector3f(2) = cloud_->points.at(point_index).z;
        km_vectors.push_back(temp_vector3f);
    }
    cluster_KMeans_vectors(km_vectors);
} 

void ClusterAlgorithm::cluster_GMM()
{
    srand(time(0));
    int dim = 3;

    // use 1/cluster_num as the weight of each gaussian distribution
    std::vector<float> alphas(cluster_num_, 1/static_cast<float>(cluster_num_));
    // random select cluster_num points as initial mus
    std::vector<Eigen::Vector3f> mus; 
    Eigen::Vector3f temp_point;
    int rand_index;
    for (int cluster_index = 0; cluster_index < cluster_num_; ++cluster_index)
    {
        rand_index = std::rand() % cloud_->size();
        temp_point(0) = cloud_->points.at(rand_index).x;
        temp_point(1) = cloud_->points.at(rand_index).y;
        temp_point(2) = cloud_->points.at(rand_index).z;
        mus.push_back(temp_point); // initial mus complete
    }
    // use matrix I as the cov matrix of each gaussian distribution
    std::vector<Eigen::Matrix3f> sigmas(cluster_num_, Eigen::Matrix3f::Identity());

    // interation loop
    std::vector<Eigen::VectorXf> gammas(cloud_->size()); 
    int iteration = 0;
    while (iteration < max_iterations_)
    {
        // E step: Refer to 《机器学习》P210
        for (int point_index = 0; point_index < cloud_->size(); ++point_index)
        {
            // for each points, calculate posteriors of gaussian distribution
            Eigen::Vector3f temp_point(cloud_->points.at(point_index).x, cloud_->points.at(point_index).y, cloud_->points.at(point_index).z);
            gammas.at(point_index).resize(cluster_num_);
            // calculate the denominator firstly
            float temp_denominator = 0;
            for (int cluster_index = 0; cluster_index <cluster_num_; ++cluster_index)
            {
                temp_denominator += alphas.at(cluster_index) * gaussian_distribution(temp_point, mus.at(cluster_index),sigmas.at(cluster_index));
            }
            // calculate the numerators and posteriors of gaussian distribution
            float temp_numerator;
            for (int cluster_index = 0; cluster_index < cluster_num_; ++cluster_index)
            {
                temp_numerator = alphas.at(cluster_index) * gaussian_distribution(temp_point, mus.at(cluster_index),sigmas.at(cluster_index));
                gammas.at(point_index)(cluster_index) = temp_numerator / temp_denominator;
            }
        }

        // M step: Refer to 《机器学习》P210
        std::vector<float> effective_num(cluster_num_, 0);
        for (int point_index = 0; point_index < cloud_->size(); ++point_index)
        {
            for (int cluster_index = 0; cluster_index < cluster_num_; ++cluster_index)
            {
                effective_num.at(cluster_index) += gammas.at(point_index)(cluster_index);
            }
        }
        std::vector<Eigen::Vector3f> temp_mus_numerator(cluster_num_, Eigen::Vector3f::Zero());
        std::vector<Eigen::Matrix3f> temp_sigmas_numerator(cluster_num_, Eigen::Matrix3f::Zero());
        for (int cluster_index = 0; cluster_index < cluster_num_; ++cluster_index)
        {
            // update the mus
            for (int point_index = 0; point_index < cloud_->size(); ++point_index)
            {
                Eigen::Vector3f temp_point(cloud_->points.at(point_index).x, cloud_->points.at(point_index).y, cloud_->points.at(point_index).z);
                temp_mus_numerator.at(cluster_index) += gammas.at(point_index)(cluster_index) * temp_point;
            }
            mus.at(cluster_index) = temp_mus_numerator.at(cluster_index) / effective_num.at(cluster_index);
            // update the sigmas
            for (int point_index = 0; point_index < cloud_->size(); ++point_index)
            {
                Eigen::Vector3f temp_point(cloud_->points.at(point_index).x, cloud_->points.at(point_index).y, cloud_->points.at(point_index).z);
                temp_sigmas_numerator.at(cluster_index) += gammas.at(point_index)(cluster_index) * (temp_point - mus.at(cluster_index)) * (temp_point - mus.at(cluster_index)).transpose();
            }
            sigmas.at(cluster_index) = temp_sigmas_numerator.at(cluster_index) / effective_num.at(cluster_index);
            for (int dim_index = 0; dim_index < dim; ++dim_index)
            {
                if (sigmas.at(cluster_index)(dim_index, dim_index) < 0.01)
                    sigmas.at(cluster_index)(dim_index, dim_index) = 0.01;
            }
            // update the alphas
            alphas.at(cluster_index) = effective_num.at(cluster_index) / cloud_->size();
        }
        ++iteration;
    }

    // store the cluster result
    ClusterAlgorithm::cluster_indices_->resize(cluster_num_);
    Eigen::VectorXf::Index cluster_label;
    for (int point_index = 0; point_index < cloud_->size(); ++point_index)
    {
        //int cluster_label;
        gammas.at(point_index).maxCoeff(&cluster_label);
        ClusterAlgorithm::cluster_indices_->at(cluster_label).indices.push_back(point_index);
    }
} 

void ClusterAlgorithm::cluster_spectral_clustring()
{
    std::vector<Eigen::VectorXf> km_vectors;
    calculate_KM_input(km_vectors);
    cluster_KMeans_vectors(km_vectors);
} 

float ClusterAlgorithm::gaussian_distribution(Eigen::Vector3f x, Eigen::Vector3f mu, Eigen::Matrix3f sigma)
{
    return 1 / std::sqrt(std::pow(2 * PI, x.size()) * sigma.determinant()) * std::exp(-0.5 * (x - mu).transpose() * sigma.inverse() * (x - mu));
}

void ClusterAlgorithm::calculate_KM_input(std::vector<Eigen::VectorXf>& km_vectors)
{
    Eigen::MatrixXf weight_matrix;
    weight_matrix.resize(cloud_->size(), cloud_->size());
    for (int row_index = 0; row_index < cloud_->size(); ++row_index)
    {
        for (int col_index = 0; col_index < cloud_->size(); ++col_index)
        {
            if (col_index == row_index)
            {
                weight_matrix(row_index, col_index) = 0;
            }
            else if (col_index > row_index)
            {
                Eigen::Vector3f p1, p2;
                p1(0) = cloud_->points.at(row_index).x;
                p1(1) = cloud_->points.at(row_index).y;
                p1(2) = cloud_->points.at(row_index).z;
                p2(0) = cloud_->points.at(col_index).x;
                p2(1) = cloud_->points.at(col_index).y;
                p2(2) = cloud_->points.at(col_index).z;
                // using gaussian correlation to calculate the element of weight matrix
                weight_matrix(row_index, col_index) = exp(-1 * (p1 - p2).squaredNorm() / 2 / 0.5 / 0.5);
            }
            else
            {
                weight_matrix(row_index, col_index) = weight_matrix(col_index, row_index);
            }
        }
    }

    Eigen::MatrixXf d_matrix; 
    d_matrix.resize(cloud_->size(), cloud_->size());
    d_matrix.setIdentity();
    Eigen::VectorXf sum_vector = weight_matrix.colwise().sum();
    for (int row_index = 0; row_index < cloud_->size(); ++row_index)
    {
        d_matrix(row_index, row_index) = sum_vector(row_index); // calculate D Matrix
    }

    Eigen::MatrixXf laplacian_matrix;
    laplacian_matrix.resize(cloud_->size(), cloud_->size());
    laplacian_matrix = d_matrix - weight_matrix; // calculate L Matrix
    Eigen::MatrixXf laplacian_matrix_std = d_matrix.inverse() * laplacian_matrix; // normalize L matrix

    Eigen::EigenSolver<Eigen::MatrixXf> es(laplacian_matrix_std);
    Eigen::MatrixXf eigenvalues_full = es.pseudoEigenvalueMatrix(); // calculate the eigenvalues of L matrix
    eigenvalues_full.resize(cloud_->size(), cloud_->size());
    Eigen::MatrixXf eigenvectors_full = es.pseudoEigenvectors(); // calculate the eigenvectors of L matrix
    sortEigenvectorsByEigenvalues(eigenvalues_full, eigenvectors_full); // sort the eigenvectors based on eigenvalues

    Eigen::MatrixXf eigenvectors_min_eigenvalues = eigenvectors_full.block(0, 0, cloud_->size(), cluster_num_); // only take cluster_num_ column as matrix U
    for (int point_index = 0; point_index < cloud_->points.size(); ++point_index)
    {
        km_vectors.push_back(eigenvectors_min_eigenvalues.row(point_index).transpose()); // take each row of matrix U as a k dim vector
    }

}

void ClusterAlgorithm::cluster_KMeans_vectors(std::vector<Eigen::VectorXf>& km_vectors)
{
    srand(time(0));
    int dim = km_vectors.at(0).size();
    // random select K (cluster_num) points as the intial position of clusters 
    std::vector<Eigen::VectorXf> cluster_centroid;
    Eigen::VectorXf temp_vector;
    int rand_index;
    for (int cluster_index = 0; cluster_index < cluster_num_; ++cluster_index)
    {
        rand_index = std::rand() % km_vectors.size();
        temp_vector = km_vectors.at(rand_index);
        cluster_centroid.push_back(temp_vector);
    }
    // interation loop
    std::vector<int> cluster_labels(km_vectors.size(), -1);
    std::vector<float> distance(cluster_num_, -1.0); // For one point, the distance btw the point and the cluster_centroid
    int iteration = 0;
    while (iteration < max_iterations_)
    {
        // E step: calculate which cluster does each point belong to
        for (int point_index = 0; point_index < km_vectors.size(); ++point_index)
        {
            for (int cluster_index = 0; cluster_index < cluster_num_; ++cluster_index)
            {
                Eigen::VectorXf diff = cluster_centroid.at(cluster_index) - km_vectors.at(point_index);
                distance.at(cluster_index) = diff.norm();
            }
            cluster_labels.at(point_index) = (min_element(distance.begin(), distance.end())) - distance.begin(); // the cluster which has the minium distance with point
        }
        // M step: update the centroid of each cluster
        std::vector<Eigen::VectorXf> cluster_centroid_sum(cluster_num_);
        for (int cluster_index = 0; cluster_index < cluster_num_; ++cluster_index)
        {
            cluster_centroid_sum.at(cluster_index).resize(dim);
            cluster_centroid_sum.at(cluster_index).setZero();
        }
        std::vector<int> cluster_sum_num(cluster_num_, 0);
        for (int point_index = 0; point_index < km_vectors.size(); ++point_index)
        {
            int cluster_index = cluster_labels.at(point_index);
            cluster_centroid_sum.at(cluster_index) += km_vectors.at(point_index); 
            cluster_sum_num.at(cluster_index) += 1; // add 1 at the corresponding cluster
        }
        for (int cluster_index = 0; cluster_index < cluster_num_; ++cluster_index)
        {
            cluster_centroid.at(cluster_index) = cluster_centroid_sum.at(cluster_index) / cluster_sum_num.at(cluster_index);
        }
        ++iteration;
    }

    // store the cluster result
    ClusterAlgorithm::cluster_indices_->resize(cluster_num_);
    for (int point_index = 0; point_index < km_vectors.size(); ++point_index)
    {
        int cluster_index = cluster_labels.at(point_index);
        ClusterAlgorithm::cluster_indices_->at(cluster_index).indices.push_back(point_index);
    }
}

void ClusterAlgorithm::sortEigenvectorsByEigenvalues(Eigen::MatrixXf& eigenvalues, Eigen::MatrixXf& eigenvectors)
{
    std::vector<EigenValueAndVector> eigenvalue_and_vectors;
    int size = eigenvalues.col(0).size();
    EigenValueAndVector temp_eigenvalue_and_vector;
    for (int index = 0; index < size; ++index)
    {
        temp_eigenvalue_and_vector.eigenvalue = eigenvalues(index, index);
        temp_eigenvalue_and_vector.eigenvector = eigenvectors.col(index);
        eigenvalue_and_vectors.push_back(temp_eigenvalue_and_vector);
    }
    std::sort(eigenvalue_and_vectors.begin(), eigenvalue_and_vectors.end(), less_based_on_eigenvalue);
    
    for (int index = 0; index < eigenvalue_and_vectors.size(); ++index)
    {
        eigenvalues(index, index) = eigenvalue_and_vectors.at(index).eigenvalue;
        eigenvectors.col(index) = eigenvalue_and_vectors.at(index).eigenvector / eigenvalue_and_vectors.at(index).eigenvector.norm();
    }
}