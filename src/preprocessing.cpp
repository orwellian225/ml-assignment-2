#include <tuple>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <toml++/toml.h>
#include <fmt/core.h>
#include <fmt/ostream.h>

#include "preprocessing.h"

template <typename T>
struct fmt::formatter<
  T,
  std::enable_if_t<
    std::is_base_of_v<Eigen::DenseBase<T>, T>, char>> : ostream_formatter {};

std::vector<std::string> split_string(std::string input, std::string delimiter) {
    std::vector<std::string> result;

    size_t pos = 0;
    std::string token;
    while ((pos = input.find(delimiter)) != std::string::npos) {
        token = input.substr(0, pos);
        result.push_back(token);
        input.erase(0, pos + delimiter.length());
    }
    result.push_back(input); // Add the last element

    return result;
}

PreprocessingSet::PreprocessingSet() {
    enabled_pca = false;
}

PreprocessingSet::PreprocessingSet(toml::table preprocessing_table) {
    std::string pca_matrix_filepath_str = preprocessing_table["pca_matrix_file"].value<std::string>().value_or("NONE");
    enabled_pca = pca_matrix_filepath_str != "NONE";

    if (enabled_pca) {
        // Load the matrix
        std::filesystem::path pca_matrix_filepath(std::filesystem::current_path()/pca_matrix_filepath_str);
        std::ifstream pca_matrix_file(pca_matrix_filepath);
        std::string line;

        std::vector<std::vector<double>> temp_matrix(0);
        while (!pca_matrix_file.eof()) {
            getline(pca_matrix_file, line);

            if (line == "") { break; } // reached end of file, including a blank line

            std::vector<std::string> row_str = split_string(line, ",");
            std::vector<double> row_vals(row_str.size(), 0);
            for (size_t i = 0; i < row_str.size(); ++i) { 
                row_vals[i] = stod(row_str[i]); 
            }
            temp_matrix.push_back(row_vals);
        }


        pca_transformation = Eigen::MatrixXd(temp_matrix.size(), temp_matrix[0].size());
        for (size_t r = 0; r < temp_matrix.size(); ++r) {
            for (size_t c = 0; c < temp_matrix[0].size(); ++c) {
                pca_transformation(r, c) = temp_matrix[r][c];
            }
        }
    }
}

struct eigen_t {
    std::complex<double> value;
    Eigen::VectorXcd vec;

    bool operator < (const eigen_t& rhs) { return value.real() > rhs.value.real(); }
};

Eigen::MatrixXd PreprocessingSet::apply_pca_transformation(const Eigen::MatrixXd& data) {
    Eigen::MatrixXd centred_data = data.rowwise() - data.colwise().mean();
    return centred_data * pca_transformation;
}