import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse as agp
import data_ops as do

label_colours = [
    "xkcd:red",
    "xkcd:green",
    "xkcd:blue",
    "xkcd:pink",
    "xkcd:purple",
    "xkcd:salmon",
    "xkcd:dark teal",
    "xkcd:tangerine",
    "xkcd:brown",
    "xkcd:black",
]


def compare_features(feature_matrix: np.array, label_vector: np.array, feature_1: int, feature_2: int, bin_count=10, save_graph=False, name="None"):

    if feature_1 == feature_2:
        feat_vals = feature_matrix[:, feature_1]
        N, bins, patches = plt.hist(feat_vals, bins=bin_count, histtype="bar", stacked=True)
    else:
        feat1_vals = feature_matrix[:, feature_1]
        feat2_vals = feature_matrix[:, feature_2]
        plt.scatter(feat1_vals, feat2_vals, c=label_vector, cmap=mpl.colors.ListedColormap(label_colours))

    if save_graph:
        # Save the save
        plt.savefig(f"./data/vis/{name}_feature_comparison_{feature_1}_{feature_2}.pdf", bbox_inches="tight")
    else:
        # Show the graph
        plt.show()

def compare_for_specific_label(label: int, feature_matrix: np.array, label_vector: np.array, feature_1: int, feature_2: int, bin_count=10, save_graph=False, name="None"):
    #FINDING ALL INDICES OF VECTOR ARRAY AND THEN DELETES THOSE ROWS
    i, = np.where(label_vector != label)
    feature_matrix = np.delete(feature_matrix, i, axis = 0)
    label_vector = np.delete(label_vector, i, axis=0)
    compare_features(feature_matrix, label_vector, feature_1, feature_2, name=name, save_graph=True)
    

def main():
    parser = agp.ArgumentParser(description="Compare two specified features")
    parser.add_argument('name', metavar='name', type=str, help="The name of the data ")
    parser.add_argument('data_filepath', metavar='df', type=str, help="The filepath of the data file")
    parser.add_argument('label_filepath', metavar='lf', type=str, help="The filepath of the label file")
    parser.add_argument("feature_1", metavar="f1", type=int, help="Feature ID of the first feature")
    parser.add_argument("feature_2", metavar="f2", type=int, help="Feature ID of the second feature")
    parser.add_argument("label", metavar='l', type =int, help="label to look at")
    args = parser.parse_args()

    feature_matrix, label_vector = do.read_labelled_data(args.data_filepath, args.label_filepath, show_log=True)
    
    if(args.label == -1):
        compare_features(feature_matrix, label_vector, args.feature_1, args.feature_2, name=args.name, save_graph=False)
    else:
        compare_for_specific_label(args.label,feature_matrix, label_vector, args.feature_1, args.feature_2, name=args.name, save_graph=False)
   


if __name__ == "__main__":
    main()
