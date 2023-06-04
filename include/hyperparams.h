#pragma once

#include <vector>
#include <string>

#include <toml++/toml.h>

struct hyperparams_t {
    double learning_rate;
    double regularisation_rate;
    double convergence_criteria;
    size_t batch_size;
    size_t num_epochs;

    std::string to_string();
};

class HyperparamSet {
    public:
        std::vector<double> learning_rates;
        std::vector<double> regularisation_rates;
        double convergence_criteria; 
        size_t batch_size; // Number of data points to use before updating the gradients
        size_t num_epochs; // Number of epochs to use in training
        
        HyperparamSet() : HyperparamSet({0.1}, {0.1}, 0.1, 1000, 10) {}
        HyperparamSet(std::vector<double> learning_rates, std::vector<double> regularisation_rates, double convergence_criteria, size_t batch_size, size_t num_epochs);
        HyperparamSet(toml::table hyperparam_table);
        ~HyperparamSet();

        size_t count_permutations();
        std::vector<hyperparams_t> construct_permutations();

};