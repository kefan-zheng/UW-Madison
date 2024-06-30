import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy


def load_data(filepath):
    with open(filepath, newline='') as csvfile:
        records = list(csv.DictReader(csvfile))
    csvfile.close()
    return records


def calc_features(row):
    return np.array(list(row.values())[2:]).astype('float64')


def hac(features):
    n = len(features)
    Z = np.zeros((n-1, 4))
    """
    initialize
    """
    class Node:
        def __init__(self, index, ele_num):
            self.index = index
            self.ele_num = ele_num

    node_map = {}
    for i in range(n):
        node_map[i] = Node(i, 1)

    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = np.linalg.norm(features[i] - features[j])
            distance_matrix[j, i] = distance_matrix[i, j]

    """
    iterate
    """
    remaining_iter_num = n
    while remaining_iter_num > 1:
        z_index = n - remaining_iter_num
        # get minimum
        row_index, col_index = -1, -1
        min_val = np.inf
        for i in range(len(distance_matrix)):
            for j in range(i+1, len(distance_matrix)):
                if distance_matrix[i, j] < min_val:
                    min_val = distance_matrix[i, j]
                    row_index, col_index = i, j

        # update result
        total_num = node_map[row_index].ele_num + node_map[col_index].ele_num
        Z[z_index, 0] = min(node_map[row_index].index, node_map[col_index].index)
        Z[z_index, 1] = max(node_map[row_index].index, node_map[col_index].index)
        Z[z_index, 2] = min_val
        Z[z_index, 3] = total_num

        # update node map
        node_map[row_index].index = n + z_index
        node_map[row_index].ele_num = total_num
        last_key = 0
        for key in node_map:
            if key >= col_index:
                if key+1 in node_map:
                    node_map[key] = node_map[key+1]
                else:
                    last_key = key

        del node_map[last_key]

        # update distance matrix
        for j in range(len(distance_matrix)):
            distance_matrix[row_index, j] = max(distance_matrix[row_index, j], distance_matrix[col_index, j])
            distance_matrix[j, row_index] = distance_matrix[row_index, j]
        distance_matrix = np.delete(distance_matrix, col_index, axis=0)
        distance_matrix = np.delete(distance_matrix, col_index, axis=1)

        remaining_iter_num -= 1

    return Z


def fig_hac(Z, names):
    fig = plt.figure()
    hierarchy.dendrogram(Z, labels=names, leaf_rotation=90)
    plt.tight_layout()
    return fig


def normalize_features(features):
    return [row for row in (features - np.mean(features, axis=0)) / np.std(features, axis=0)]


if __name__ == "__main__":
    data = load_data("countries.csv")
    country_names = [row["Country"] for row in data]
    features = [calc_features(row) for row in data]
    features_normalized = normalize_features(features)
    n = 20
    Z_raw = hac(features[:n])
    Z_normalized = hac(features_normalized[:n])
    # fig = fig_hac(Z_raw, country_names[:n])
    fig = fig_hac(Z_normalized, country_names[:n])
    plt.show()
