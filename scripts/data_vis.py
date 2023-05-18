import numpy as np
import matplotlib.pyplot as plt
import argparse as agp
import data_ops as do

label_colours = [
    "#ff0000",
    "#00ff00",
    "#0000ff",
    "#ffff00",
    "#ff00ff",
    "#00ffff",
    "#ffa500",
    "#00a5ff",
    "#2f4f4f",
    "#000000",
]

label_shapes = [
    "D",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "s",
    "*",
    "2",
]


def compare_features(feature_matrix: np.array, label_vector: np.array, feature_1: int, feature_2: int, bin_count=10, save_graph=False, name="None"):

    if feature_1 == feature_2:
        feat_vals = feature_matrix[:, feature_1]
        plt.hist(feat_vals, bins=bin_count)
    else:
        feat1_vals = feature_matrix[:, feature_1]
        feat2_vals = feature_matrix[:, feature_2]
        for i in range(len(feat1_vals)):
            plt.scatter(feat1_vals[i], feat2_vals[i], color=label_colours[label_vector[i]], marker=label_shapes[label_vector[i]])

    if save_graph:
        # Save the save
        plt.savefig(f"./data/vis/{name}_feature_comparison_{feature_1}_{feature_2}.pdf", bbox_inches="tight")
    else:
        # Show the graph
        plt.show()


def main():
    parser = agp.ArgumentParser(description="Compare two specified features")
    parser.add_argument('name', metavar='name', type=str, help="The name of the data ")
    parser.add_argument('data_filepath', metavar='df', type=str, help="The filepath of the data file")
    parser.add_argument('label_filepath', metavar='lf', type=str, help="The filepath of the label file")
    parser.add_argument("feature_1", metavar="f1", type=int, help="Feature ID of the first feature")
    parser.add_argument("feature_2", metavar="f2", type=int, help="Feature ID of the second feature")
    args = parser.parse_args()

    feature_matrix, label_vector = do.read_labelled_data(args.data_filepath, args.label_filepath, show_log=True)
    compare_features(feature_matrix, label_vector, args.feature_1, args.feature_2, name=args.name, save_graph=False)


if __name__ == "__main__":
    main()
