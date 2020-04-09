import argparse
import numpy as np

import tensorflow as tf
from add_parent_path import add_parent_path

from models import Planetoid_I, Planetoid_T

with add_parent_path():
    from metrics import UnlabeledLoss
    from utils import *


def main(modality, dataset_name, embedding_dim,
        random_walk_length, window_size,
        neg_sample, sample_context_rate,
        data_seed, net_seed, checkpoint_path):
    
    print("Planetoid-{:s}!".format(modality))
    
    # reproducibility
    np.random.seed(data_seed)
    tf.random.set_seed(net_seed)

    print("reading dataset")
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset_name)
    num_classes = len(set(labels))

    print("shuffling dataset")
    features, neighbors, labels, o_h_labels, keys = permute(features, neighbors, labels, o_h_labels, keys)
    
    print("obtaining masks")
    mask_train, mask_val, mask_test = split(dataset_name, labels)

    print("calculating adjacency matrix")
    A = adjacency_matrix(neighbors)

    # Define model, loss, metrics and optimizers
    if modality == "I":
        model = Planetoid_I(A, o_h_labels, embedding_dim, random_walk_length, window_size, neg_sample, sample_context_rate)
    elif modality == "T":
        model = Planetoid_T(A, o_h_labels, embedding_dim, random_walk_length, window_size, neg_sample, sample_context_rate)

    L_s = tf.keras.losses.CategoricalCrossentropy("test_loss")

    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="val_acc")
    test_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

    model.load_weights(checkpoint_path+'Planetoid_ckpts/cp.ckpt')

    model.test(features, o_h_labels, mask_test, L_s, test_accuracy, test_loss)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test Planetoid')

    # modality can be I inductive T transductive
    parser.add_argument("-m", "--modality", help="model to use", default="I", choices=["I", "T"])
    
    # dataset choice
    parser.add_argument("-d", "--dataset", help="dataset to use", default="citeseer", choices=["citeseer", "cora", "pubmed"])
    
    # network hyperparameters
    parser.add_argument("-emb", "--embedding-dim", help="node embedding size", default=50, type=int)

    # sampling algorithm (Alg.1) hyper-parameters
    parser.add_argument("-q", "--random-walk-length", help="random walk length", default=10, type=int)
    parser.add_argument("-w", "--window-size", help="window size", default=3, type=int)
    parser.add_argument("-r1", "--neg-sample-rate", help="negative sample rate", default=5/6, type=float)
    parser.add_argument("-r2", "--sample-context-rate", help="context sample with label rate", default=5/6, type=float)

    # reproducibility
    parser.add_argument("-ds", "--data-seed", help="seed to set in numpy before shuffling dataset", default=0, type=int)
    parser.add_argument("-ns", "--net-seed", help="seed to set in tensorflow before creating the neural network", default=0, type=int)

    # save model weights
    parser.add_argument("-cp", "--checkpoint-path", help="path for loading model checkpoint")

    args = parser.parse_args()
    
    main(args.modality, args.dataset, args.embedding_dim, 
        args.random_walk_length, args.window_size, 
        args.neg_sample_rate, args.sample_context_rate,
        args.data_seed, args.net_seed, args.checkpoint_path)


