import os
import csv
import numpy as np

data = 'dataset'


def read_dataset(dataset):
    print("reading " + dataset + " dataset")
    if "pubmed" in dataset:
        return read_p(dataset)
    else:
        return read_cc(dataset)


def read_cc(dataset):
    folder = os.path.join(data, dataset)
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

    valid_edges = 0
    invalid_edges = 0
    with open(cites_file) as cites:
        rows = csv.reader(cites, delimiter="\t")
        for row in rows:
            if row[0] not in keys_to_idx or row[1] not in keys_to_idx:
                invalid_edges += 1
                continue
            cited = keys_to_idx[row[0]]
            citing = keys_to_idx[row[1]]
            neighbors[citing].append([cited, 1])
            valid_edges += 1

    print("valid edges:", valid_edges)
    print("invalid edges:", invalid_edges)

    features = np.array(features)

    return features, neighbors, labels, keys


def read_p(dataset):
    features = []
    neighbors = []
    labels = []
    keys = []

    t_dict = {}
    folder = os.path.join(data, dataset)
    nodes = os.path.join(folder, 'data/Pubmed-Diabetes.NODE.paper.tab')
    edges = os.path.join(folder, 'data/Pubmed-Diabetes.DIRECTED.cites.tab')

    with open(nodes) as csvFile:
        for idx, row in enumerate(csvFile):
            if (idx < 2):
                if ('cat' in row):
                    indices = row.replace('\n', '').replace(
                        'numeric:w-', '').replace(':0.0', '').split('\t')[1:]
            else:
                node_feature = np.zeros(500)
                node = row.replace('\n', '').replace('w-', '').split('\t')
                labels.append(int(node[1].replace('label=', '')))
                keys.append(node[0])
                for i in range(2, len(node)-1):
                    k, v = node[i].split('=')
                    node_feature[indices.index(k)] = v
                features.append(node_feature)

    neighbors = [[] for i in range(len(features))]

    with open(edges) as csvFile:
        for idx, row in enumerate(csvFile):
            if (idx >= 2):
                node = row.replace('\n', '').replace('paper:', '').split('\t')
                node.remove('|')
                neighbors[keys.index(node[1])].append(
                    [keys.index(node[2]), int(node[0])])

    return np.asarray(features), neighbors, labels, keys


def permute(features, neighbors, labels, keys, seed=None):

    np.random.seed(seed=seed)
    permutation = np.random.permutation(len(keys))
    inv_permutation = np.argsort(permutation)
    labels = [labels[i] for i in permutation]
    keys = [keys[p] for p in permutation]
    features = features[permutation]
    for n in neighbors:
        for edge in n:
            edge[0] = inv_permutation[edge[0]]

    neighbors = [neighbors[p] for p in permutation]
    return features, neighbors, labels, keys


def split(dataset):
    n_classes = 3
    if dataset == "cora":
        n_classes = 7
    elif dataset == "citeseer":
        n_classes = 6
    
    train_size = 20*n_classes
    return np.arange(train_size), np.arange(train_size, train_size+500), np.arange(train_size+500, train_size+1500)


if __name__ == '__main__':

    features, neighbors, labels, keys = read_dataset("cora")
    permute(features, neighbors, labels, keys, 1234)
    train_idx, val_idx, test_idx = split("cora")
