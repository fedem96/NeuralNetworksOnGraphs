import argparse
import numpy as np

import tensorflow as tf
from add_parent_path import add_parent_path

from models import Planetoid_I, Planetoid_T

with add_parent_path():
    from metrics import UnlabeledLoss
    from utils import *


DIR_NAME = os.path.dirname(os.path.realpath(__file__))


def main(modality, dataset_name, embedding_dim, yang_splits,
        random_walk_length, window_size,
        neg_sample, data_seed, 
        checkpoint_path, verbose,
        tsne):
    
    print("Planetoid-{:s}!".format(modality))

    # reproducibility
    np.random.seed(data_seed)

    if yang_splits:
        features, o_h_labels, A, mask_train, mask_val, mask_test = read_dataset(dataset_name, yang_splits=True)
        labels = np.array([np.argmax(l) for l in o_h_labels], dtype=np.int32)
    else:
        if verbose > 0: print("reading dataset")
        features, neighbors, labels, o_h_labels, keys = read_dataset(dataset_name)

        if verbose > 0: print("shuffling dataset")
        features, neighbors, labels, o_h_labels, keys = permute(features, neighbors, labels, o_h_labels, keys)
        
        if verbose > 0: print("obtaining masks")
        mask_train, mask_val, mask_test = split(dataset_name, labels)

        if verbose > 0: print("calculating adjacency matrix")
        A = adjacency_matrix(neighbors)

    num_classes = get_num_classes(dataset_name)

    # Define model, loss, metrics and optimizers
    if modality == "I":
        labeled_iters = 10000
        model = Planetoid_I(mask_test, A, o_h_labels, embedding_dim, random_walk_length, window_size, neg_sample, mask_train, labeled_iters)
    elif modality == "T":
        labeled_iters = 2000    
        model = Planetoid_T(A, o_h_labels, embedding_dim, random_walk_length, window_size, neg_sample, mask_train, labeled_iters)

    L_s = tf.keras.losses.CategoricalCrossentropy("loss_s")

    if verbose > 0: print("test the model on test set")

    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="bw_test_accuracy")
    test_loss = tf.keras.metrics.Mean('bw_test_loss', dtype=tf.float32)

    wpath = os.path.join(checkpoint_path,'cp.ckpt')
    model.load_weights(wpath).expect_partial()
    
    t_loss, t_acc = model.test(features, o_h_labels, mask_test, L_s, test_accuracy)

    print("Test acc {:.3f}" .format(t_acc))

    if tsne:
        if verbose > 0: print("calculating t-SNE plot")
        # tsne of the hidden rapresentations: 
        if modality == "T":
            # - instances embeddings for the transductive one
            intermediate_output = model.get_manifold(np.where(mask_test)[0])
        else:
            # - par embeddings for the the inductive model;
            intermediate_output = model.get_manifold(features[mask_test])
        
        plot_tsne(intermediate_output, labels[mask_test], len(o_h_labels[0]), 'Planetoid-'+modality)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test Planetoid')

    # modality can be I inductive T transductive
    parser.add_argument("-m", "--modality", help="model to use", default="I", choices=["I", "T"])
    
    # dataset choice
    parser.add_argument("-d", "--dataset", help="dataset to use", default="cora", choices=["citeseer", "cora", "pubmed"])
    parser.add_argument("-y", "--yang-splits", help="whether to use Yang splits or not", default=False, action='store_true')
    
    # network hyperparameters
    parser.add_argument("-emb", "--embedding-dim", help="node embedding size", default=50, type=int)

    # sampling algorithm (Alg.1) hyper-parameters
    parser.add_argument("-q", "--random-walk-length", help="random walk length", default=10, type=int)
    parser.add_argument("-w", "--window-size", help="window size", default=3, type=int)
    parser.add_argument("-r1", "--neg-sample-rate", help="negative sample rate", default=0, type=float)

    # reproducibility
    parser.add_argument("-ds", "--data-seed", help="seed to set in numpy before shuffling dataset", default=0, type=int)

    # save model weights
    parser.add_argument("-cp", "--checkpoint-path", help="path for loading model checkpoint", type=str)

    # verbose
    parser.add_argument("-v", "--verbose", help="useful prints", default=1, type=int)

    # tsne
    parser.add_argument("-t", "--tsne", help="whether to make t-SNE plot or not", default=False, action='store_true')

    args = parser.parse_args()
    print(args.checkpoint_path)
    args.checkpoint_path = args.checkpoint_path.encode("ascii").decode("utf-8")
    print(args.checkpoint_path)
    
    main(args.modality, args.dataset, args.embedding_dim, args.yang_splits,
        args.random_walk_length, args.window_size, 
        args.neg_sample_rate, args.data_seed,
        args.checkpoint_path, args.verbose,
        args.tsne)


