#pragma once

#include <tuple>

#include <Eigen/Dense>
#include <toml++/toml.h>

class PreprocessingSet {
    public: 
        double pca;

        PreprocessingSet() : PreprocessingSet(0.0) {}
        PreprocessingSet(double pca) : pca(pca) {}
        PreprocessingSet(toml::table preprocessing_table);
};

Eigen::MatrixXd construct_pca_transformation(const Eigen::MatrixXd& data, const double energy_threshold);
Eigen::MatrixXd apply_pca_transformation(const Eigen::MatrixXd& data, const Eigen::MatrixXd& pca_transformation);