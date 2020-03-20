import csv
import numpy as np
import os

def read_dataset(dataset):
    if "pubmed" in dataset:
        return read_p(dataset)
    else:
        return read_cc(dataset)


def read_cc(dataset):
    folder = os.path.join("data", dataset)
    content_file = os.path.join(folder, dataset + ".content")
    cites_file = os.path.join(folder, dataset + ".cites")

    features = []
    neighbors = []
    labels = []
    keys = []
    keys_to_idx = {}

    with open(content_file) as content:
        rows = csv.reader(content, delimiter="\t")
        for r, row in enumerate(rows):
            key, fs, label = row[0], row[1:-1], row[-1]

            keys.append(key)
            features.append(fs)
            labels.append(label)
            neighbors.append([])

            keys_to_idx[key] = r

    with open(cites_file) as cites:
        rows = csv.reader(cites, delimiter="\t")
        for row in rows:
            cited = keys_to_idx[row[0]]
            citing = keys_to_idx[row[1]]
            neighbors[citing].append(cited)
    
    features = np.array(features)

    return features, neighbors, labels, keys

def read_p(dataset):
    folder = os.path.join(data, dataset)
    ...


if __name__ == "__main__":
    read_cc("cora")