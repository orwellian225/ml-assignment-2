#include <vector>

#include <toml++/toml.h>

#include "hyperparams.h"

HyperparamSet::HyperparamSet(std::vector<double> learning_rates, std::vector<double> regularisation_rates, double convergence_criteria, size_t batch_size, size_t num_epochs) {
    this->learning_rates = learning_rates;
    this->regularisation_rates = regularisation_rates;
    this->convergence_criteria = convergence_criteria;
    this->batch_size = batch_size;
    this->num_epochs = num_epochs;
}

HyperparamSet::HyperparamSet(toml::table hyperparam_table) {

    auto learning_rates_toml = hyperparam_table["learning_rates"].as_array();
    for (size_t i = 0; i < learning_rates_toml->size(); ++i) {
        learning_rates.push_back(learning_rates_toml->get_as<double>(i)->value_or(0.0));
    }

    auto regularisation_rates_toml = hyperparam_table["regularisation_rates"].as_array();
    for (size_t i = 0; i < regularisation_rates_toml->size(); ++i) {
        regularisation_rates.push_back(regularisation_rates_toml->get_as<double>(i)->value_or(0.0));
    }

    this->convergence_criteria = hyperparam_table["convergence_criteria"].value<double>().value_or(0.001);
    this->batch_size = hyperparam_table["batch_size"].value<size_t>().value_or(1000);
    this->num_epochs = hyperparam_table["num_epochs"].value<size_t>().value_or(1);
}

size_t HyperparamSet::count_permutations() {
   return learning_rates.size() * regularisation_rates.size();
}

std::vector<hyperparams_t> HyperparamSet::construct_permutations() {
    const size_t num_permutations = count_permutations();
    std::vector<hyperparams_t> permutations(num_permutations);

    size_t learning_idx = 0;
    size_t regularisation_idx = 0;
    for (size_t i = 0; i < num_permutations; ++i) {
        permutations[i] = hyperparams_t {
            learning_rates[learning_idx],
            regularisation_rates[regularisation_idx],
            convergence_criteria,
            batch_size,
            num_epochs,
        };

        ++learning_idx;

        if (learning_idx == learning_rates.size()) {
            learning_idx = 0;
            ++regularisation_idx;
        }
        if (regularisation_idx == regularisation_rates.size()) {
            regularisation_idx = 0;
        }
    }

    return permutations;
}