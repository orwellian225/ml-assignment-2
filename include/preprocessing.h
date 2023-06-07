#pragma once

#include <tuple>

#include <Eigen/Dense>
#include <toml++/toml.h>

class PreprocessingSet {
    public: 
        bool enabled_pca;
        Eigen::MatrixXd pca_transformation;

        PreprocessingSet();
        PreprocessingSet(toml::table preprocessing_table);

        Eigen::MatrixXd apply_pca_transformation(const Eigen::MatrixXd& data);
};

std::vector<std::string> split_string(std::string input, std::string delimiter);