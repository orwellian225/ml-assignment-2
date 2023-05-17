from typing import Dict, Union
import argparse as agp
import numpy as np
import data_ops as do

def calc_summary_stats(feature_matrix: np.array, feature: int) -> Dict[str, Union[float, int]]:
    feature_values = feature_matrix[:, feature]
    return {
        "Mean": np.mean(feature_values),
        "Median": np.mean(feature_values),
        "Variance": np.var(feature_values),
        "Standard Deviation": np.std(feature_values),
        "Minimum": np.min(feature_values),
        "Maximum": np.max(feature_values),
        "Range": np.max(feature_values) - np.min(feature_values)
    }

def calc_summary_stats(feature_matrix: np.array) -> Dict[str, np.array]:
    result = {
        "Mean": [],
        "Median": [],
        "Variance": [],
        "Standard Deviation": [],
        "Minimum": [],
        "Maximum": [],
        "Range": []
    }

    for i in range(len(feature_matrix[0])):
        feature_vals = feature_matrix[:, i]
        result["Mean"].append(np.mean(feature_vals))
        result["Median"].append(np.median(feature_vals))
        result["Variance"].append(np.var(feature_vals))
        result["Standard Deviation"].append(np.std(feature_vals))
        result["Minimum"].append(np.min(feature_vals))
        result["Maximum"].append(np.max(feature_vals))
        result["Range"].append(np.max(feature_vals) - np.min(feature_vals))

    return result

def main():
    parser = agp.ArgumentParser(description="Calculate the summary statistics of the specified data")
    parser.add_argument('data_filepath', metavar='df', type=str, help='The filepath of the text file holding the data')
    parser.add_argument('labels_filepath', metavar='lf', type=str, help='The filepath of the text file holding the labels')
    parser.add_argument('feature', metavar='i', type=int, help='The feature index to calculate the summary statistics for.\n-1 will calculate the statistics for all the features')
    args = parser.parse_args()

    matrix, _ = do.read_labelled_data(args.data_filepath, args.labels_filepath, show_log=True)

    if args.feature == -1:
        stats = calc_summary_stats(matrix)
    else:
        stats = calc_summary_stats(matrix, args.feature)

    print(f"Statistics for feature {args.feature}")
    for stat_name in stats.keys():
        print(f"\t{stat_name} {stats[stat_name]}")


if __name__ == "__main__":
    main()
