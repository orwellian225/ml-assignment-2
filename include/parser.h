#pragma once

#include <string>
#include <filesystem>
#include <vector>

#include <toml++/toml.h>

struct nnspec_t {
    std::string name;
    std::string author;

    struct {
        std::filesystem::path data_file;
        std::filesystem::path label_file;
        size_t feature_count;
        size_t label_count;
    } data;

    struct {
        std::vector<size_t> structure;

        struct {
            std::vector<double> learning_rates;
        } hyperparameters;
    } network;

    void print_spec_info();
};

nnspec_t load_network_spec(toml::table network_spec);
nnspec_t load_network_spec_from_file(std::filesystem::path filepath);
