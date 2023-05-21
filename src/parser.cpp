#include <filesystem>
#include <string>

#include <fmt/core.h>
#include <fmt/color.h>
#include <toml++/toml.h>

#include "parser.h"

nnspec_t load_network_spec(toml::table network_spec) {
    std::string name = network_spec["name"].value<std::string>().value_or("NO NAME");
    std::string author = network_spec["author"].value<std::string>().value_or("NO AUTHOR");

    auto data_node = network_spec["data"];
    std::filesystem::path data_file(data_node["data_file"].value<std::string>().value_or("NO_DATA"));
    std::filesystem::path label_file(data_node["label_file"].value<std::string>().value_or("NO_LABELS"));
    size_t feature_count = data_node["feature_count"].value<size_t>().value_or(0);
    size_t label_count = data_node["label_count"].value<size_t>().value_or(0);
    size_t size = data_node["size"].value<size_t>().value_or(0);

    auto network_node = network_spec["network"];
    auto structure_arr = network_node["structure"].as_array();
    std::vector<size_t> structure;
    for (size_t i = 0; i < structure_arr->size(); ++i) {
        structure.push_back((size_t)(structure_arr->get_as<int64_t>(i)->value_or(0)));
    }

    auto network_hyperparams = network_spec["network"]["hyperparameters"];
    auto learning_rates_arr = network_hyperparams["learning_rates"].as_array();
    std::vector<double> learning_rates;
    for (size_t i = 0; i < learning_rates_arr->size(); ++i) {
        learning_rates.push_back(learning_rates_arr->get_as<double>(i)->value_or(0.0));
    }

    return nnspec_t {
        name,
        author,
        {
            data_file,
            label_file,
            feature_count,
            label_count,
            size
        },
        { structure, { learning_rates } }
    };
}

nnspec_t load_network_spec_from_file(std::filesystem::path filepath) {
    auto network_spec = toml::parse_file(filepath.string());
    return load_network_spec(network_spec);
}

void nnspec_t::print_spec_info() {
    fmt::println("");
    fmt::print(fg(fmt::color::green), "Network "); fmt::print("{}\n", name);
    fmt::print(fg(fmt::color::green), "================================================================\n");
    fmt::print(fg(fmt::color::orange), "Structure "); fmt::print("{}\n", fmt::join(network.structure, " "));
    fmt::print(fg(fmt::color::orange), "Data "); fmt::print("{}\n", data.data_file.string());
    fmt::print(fg(fmt::color::orange), "Labels "); fmt::print("{}\n", data.label_file.string());
    fmt::print(fg(fmt::color::orange), "Data size "); fmt::println("{}", data.size);
    fmt::print(fg(fmt::color::orange), "Feature count "); fmt::println("{}", data.feature_count);
    fmt::print(fg(fmt::color::orange), "Label count "); fmt::println("{}", data.label_count);
    fmt::print(fg(fmt::color::green), "================================================================\n");
    fmt::println("");
}