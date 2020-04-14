import csv, os, pickle as pkl, sys

import networkx as nx
import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import eigs

DIR_NAME = os.path.dirname(os.path.realpath(__file__))
data = DIR_NAME + '/data'


def read_dataset(dataset, yang_splits=False):
    if yang_splits:
        return read_dataset_yang_splits(dataset)

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
    classes = set()

    with open(content_file) as content:
        rows = csv.reader(content, delimiter="\t")
        for r, row in enumerate(rows):
            key, fs, label = row[0], row[1:-1], row[-1]

            keys.append(key)
            features.append(np.array(fs, dtype=np.float32))
            labels.append(label)
            neighbors.append([])

            classes.add(label)
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
            neighbors[citing].append(np.array([cited, 1]))
            valid_edges += 1

    print("valid edges:", valid_edges)
    print("invalid edges:", invalid_edges)

    features = np.array(features)
    neighbors = np.array(neighbors)

    classes = sorted(classes)
    labels = int_enc(labels, classes)
    o_h_labels = one_hot_enc(len(classes), labels)

    return features, neighbors, labels, o_h_labels, keys


def int_enc(labels, classes):
    int_labels = []
    for l in range(len(labels)):
        int_label = classes.index(labels[l])
        int_labels.append(int_label)
    return np.array(int_labels)

def read_p(dataset):

    features = []
    neighbors = []
    labels = []
    keys = []
    keys_to_idx = {}
    classes = set()

    t_dict = {}
    folder = os.path.join(data, dataset)
    nodes = os.path.join(folder, 'data/Pubmed-Diabetes.NODE.paper.tab')
    edges = os.path.join(folder, 'data/Pubmed-Diabetes.DIRECTED.cites.tab')

    with open(nodes) as csvFile:
        rows = list(csvFile)
        indices = rows[1].replace('\n', '').replace('numeric:w-', '').replace(':0.0', '').split('\t')[1:]
        keys_to_word_idx = {indices[i]: i for i in range(len(indices))}
        rows = rows[2:]
        for row in rows:
            node_feature = np.zeros(500)

            node = row.replace('\n', '').replace('w-', '').split('\t')
            label = int(node[1].replace('label=', ''))
            labels.append(label)
            classes.add(label)
            keys_to_idx[node[0]] = len(keys)
            keys.append(node[0])

            for i in range(2, len(node)-1):
                k, v = node[i].split('=')
                node_feature[keys_to_word_idx[k]] = v
            features.append(node_feature)

    neighbors = [[] for i in range(len(features))]

    with open(edges) as csvFile:
        rows = list(csvFile)[2:]
        for row in rows:
            node = row.replace('\n', '').replace('paper:', '').split('\t')
            node.remove('|')
            neighbors[keys_to_idx[node[1]]].append(np.array([keys_to_idx[node[2]], int(node[0])]))

    labels = int_enc(labels, sorted(classes))
    o_h_labels = one_hot_enc(n_classes=3, labels=labels)

    return np.array(features, dtype=np.float32), np.array(neighbors), labels, o_h_labels, keys


def one_hot_enc(n_classes, labels):
    o_h_labels = []
    for l in range(len(labels)):
        label = labels[l]
        o_h_label = np.zeros(n_classes)
        o_h_label[label] = 1
        o_h_labels.append(o_h_label)

    return np.array(o_h_labels).astype(np.int32)


def permute(features, neighbors, labels, o_h_labels, keys):

    permutation = np.random.permutation(len(keys))
    inv_permutation = np.argsort(permutation)
    labels = labels[permutation]
    o_h_labels = o_h_labels[permutation]
    keys = [keys[p] for p in permutation]
    features = features[permutation]
    for n in neighbors:
        for edge in n:
            edge[0] = inv_permutation[edge[0]]

    neighbors = [neighbors[p] for p in permutation]

    return features, neighbors, labels, o_h_labels, keys

def normalize_features(features):
    rowsum = np.array(features.sum(1))
    rowsum[np.where(rowsum == 0)] = np.inf
    normalized_features = features / np.broadcast_to(rowsum, features.T.shape).T
    #normalized_features[np.isinf(normalized_features)] = 0
    return normalized_features

def get_num_classes(dataset):
    if dataset == "citeseer":
        return 6
    elif dataset == "cora":
        return 7
    elif dataset == "pubmed":
        return 3

def split(dataset, labels):
    n_classes = get_num_classes(dataset)
    size = len(labels)
    counters = np.zeros(n_classes)

    train_size = 20*n_classes
    mask_train = np.zeros(size, dtype=bool)
    mask_val = np.zeros(size, dtype=bool)
    t = v = i = 0
    while t < train_size:
        label = labels[i]
        if counters[label] < 20:  # 20 nodes per class in the training set
            mask_train[i] = True
            counters[label] += 1
            t += 1
        elif v < 500:
            mask_val[i] = True
            v += 1
        i += 1

    mask_val[np.arange(train_size + v, train_size+500)] = True

    mask_test = np.zeros(size, dtype=bool)
    mask_test[np.arange(size-1000, size)] = True

    return mask_train, mask_val, mask_test

def adjacency_matrix(neighbors, self_loops=False, weighted=False):
    num_nodes = len(neighbors)
    row_ind = []
    col_ind = []
    values = []

    for n, adjacency_list in enumerate(neighbors):
        if self_loops:
            row_ind.append(n); col_ind.append(n); values.append(1)
        for edge in adjacency_list:
            neighbor = edge[0]
            weight = edge[1] if weighted else 1
            row_ind.append(n); col_ind.append(neighbor); values.append(weight)
            row_ind.append(neighbor); col_ind.append(n); values.append(weight)
            # the adjacency matrix must se symmetric
            
            # TODO: symmetrize non-DAGs (i.e. treat the case of two edges between a pair of nodes)

    return sparse.csr_matrix((values, (row_ind, col_ind)), shape=[num_nodes, num_nodes])

def degree_matrix(A):
    D = np.diag(np.sum(A, axis=1))
    return D

def semi_inverse_degree_matrix(A):
    D_minus_half = sparse.diags( np.power(np.sum(A, axis=0), -1/2), [0], shape=A.shape )
    return D_minus_half

def normalized_laplacian_matrix(A):
    n = A.shape[0]
    D_minus_half = semi_inverse_degree_matrix(A)
    norm_L = sparse.identity(n) - D_minus_half.dot(A).dot(D_minus_half)
    return norm_L

def scaled_normalized_laplacian_matrix(A):
    n = A.shape[0]
    norm_L = normalized_laplacian_matrix(A)
    lambda_max = eigs(norm_L, k=1, which='LM', return_eigenvectors=True)[0] # Largest-Magnitude eigenvalue of norm_L
    scaled_norm_L = float(2/lambda_max.real) * norm_L - sparse.identity(n)
    return scaled_norm_L

def renormalization_matrix(A):
    n = A.shape[0]
    renorm_A = A + sparse.identity(n)
    renorm_D_minus_half = semi_inverse_degree_matrix(renorm_A)
    renormalized_matrix = renorm_D_minus_half.dot(A).dot(renorm_D_minus_half)
    return renormalized_matrix

def parse_index_file(filename): # directly from GCN Github code
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l): # directly from GCN Github code
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def read_dataset_yang_splits(dataset):  # adapted from GCN Github code
    print("reading " + dataset + " dataset with Yang splits")
    data_folder = os.path.join(data, "yang_splits")
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_folder, "ind.{}.{}".format(dataset, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_folder, "ind.{}.test.index".format(dataset)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sparse.vstack((allx, tx)).astype(np.float32).toarray()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    o_h_labels = np.vstack((ally, ty)) # one-hot encoded
    o_h_labels[test_idx_reorder, :] = o_h_labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, o_h_labels.shape[0])
    val_mask = sample_mask(idx_val, o_h_labels.shape[0])
    test_mask = sample_mask(idx_test, o_h_labels.shape[0])

    return features, o_h_labels, adj, train_mask, val_mask, test_mask

# if __name__ == '__main__':
def main():
    dataset = "pubmed"
    # np.random.seed(0)

    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset)
    features = normalize_features(features)
    permute(features, neighbors, labels, o_h_labels, keys)
    train_idx, val_idx, test_idx = split(dataset, labels)
    A = adjacency_matrix(neighbors)

    features, o_h_labels, adj, train_mask, val_mask, test_mask = read_dataset(dataset, yang_splits=True)

    

if __name__ == '__main__':
    main()
