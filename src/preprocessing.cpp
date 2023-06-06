#include <tuple>
#include <algorithm>

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

PreprocessingSet::PreprocessingSet(toml::table preprocessing_table) {
    pca = preprocessing_table["pca"].value<double>().value_or(0.0);
}

struct eigen_t {
    std::complex<double> value;
    Eigen::VectorXcd vec;

    bool operator < (const eigen_t& rhs) { return value.real() > rhs.value.real(); }
};

Eigen::MatrixXd construct_pca_transformation(const Eigen::MatrixXd& data, const double energy_threshold) {

    Eigen::MatrixXd centred_data = data.rowwise() - data.colwise().mean();
    Eigen::MatrixXd cov_matrix = 1.0 / (data.cols() - 1) * centred_data.transpose() * centred_data;

    Eigen::EigenSolver<Eigen::MatrixXd> solver(cov_matrix, true);
    Eigen::MatrixXcd eigen_vecs = solver.eigenvectors();
    Eigen::VectorXcd eigen_vals = solver.eigenvalues();

    std::vector<eigen_t> eigens(eigen_vecs.cols());
    for (size_t i = 0; i < eigen_vecs.cols(); ++i) {
        eigens[i] = eigen_t {
            eigen_vals(i),
            eigen_vecs.col(i),
        };
    }

    std::sort(eigens.begin(), eigens.end());
    for (size_t i = 0; i < eigens.size(); ++i) {
        eigen_vecs.col(i) = eigens[i].vec;
        eigen_vals(i) = eigens[i].value;
    }

    Eigen::VectorXd energy_sums(eigen_vals.size());
    for (size_t i = 0; i < eigen_vals.size(); ++i) {
        energy_sums(i) = eigen_vals.segment(0, i + 1).real().sum();
    }

    size_t threshold_idx = eigen_vals.size() - 1;
    double desired_energy = energy_threshold * eigen_vals.real().sum();
    for (size_t i = 0; i < eigen_vals.size(); ++i) {
        if (energy_sums(i) >=  desired_energy) {
            threshold_idx = i;
            break;
        }
    }

    Eigen::MatrixXd pca_transformation = eigen_vecs.block(0, 0, eigen_vecs.rows(), threshold_idx + 1).real();
    return pca_transformation;
}

Eigen::MatrixXd apply_pca_transformation(const Eigen::MatrixXd& data, const Eigen::MatrixXd& pca_transformation) {
    Eigen::MatrixXd centred_data = data.rowwise() - data.colwise().mean();
    return centred_data * pca_transformation;
}