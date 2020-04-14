import argparse
import numpy as np

import tensorflow as tf
from add_parent_path import add_parent_path

from models import Planetoid_I, Planetoid_T

with add_parent_path():
    from metrics import UnlabeledLoss
    from utils import *


def main(modality, dataset_name, yang_splits,
        embedding_dim, epochs, pretrain_batch,
        supervised_batch, unsupervised_batch, supervised_batch_size,
        unsupervised_batch_size, learning_rate_supervised, 
        learning_rate_unsupervised, patience,
        random_walk_length, window_size, 
        neg_sample, sample_context_rate,
        data_seed, net_seed, checkpoint_path, verbose):
    
    print("Planetoid-{:s}!".format(modality))

    # reproducibility
    np.random.seed(data_seed)
    tf.random.set_seed(net_seed)

    if yang_splits:
        features, o_h_labels, A, mask_train, mask_val, mask_test = read_dataset(dataset_name, yang_splits=True)
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
        model = Planetoid_I(mask_test, A, o_h_labels, embedding_dim, random_walk_length, window_size, neg_sample, sample_context_rate, mask_train, labeled_iters)
    elif modality == "T":
        labeled_iters = 2000
        model = Planetoid_T(A, o_h_labels, embedding_dim, random_walk_length, window_size, neg_sample, sample_context_rate, mask_train, labeled_iters)

    L_s = tf.keras.losses.CategoricalCrossentropy("loss_s")
    if neg_sample > 0:
        L_u = UnlabeledLoss(unsupervised_batch_size)
    else:
        L_u = tf.keras.losses.SparseCategoricalCrossentropy("loss_u")

    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="bw_accuracy")
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name="bw_val_accuracy")
    train_loss = tf.keras.metrics.Mean('bw_loss', dtype=tf.float32)
    train_loss_u = tf.keras.metrics.Mean('bw_loss_u', dtype=tf.float32)
    val_loss = tf.keras.metrics.Mean('bw_val_loss', dtype=tf.float32)

    optimizer_u = tf.keras.optimizers.SGD(learning_rate=learning_rate_unsupervised)   
    optimizer_s = tf.keras.optimizers.SGD(learning_rate=learning_rate_supervised)      

    if verbose > 0: print("pre-train model")
    # Pretrain iterations on graph context
    model.pretrain_step(features, mask_test, L_u, optimizer_u, train_loss_u, pretrain_batch, unsupervised_batch_size)

    if verbose > 0: print("begin training")
    model.train(features, o_h_labels, mask_train, mask_val, mask_test, epochs, L_s, L_u, optimizer_u, optimizer_s, train_accuracy, val_accuracy, train_loss, 
            train_loss_u, val_loss, supervised_batch, unsupervised_batch, supervised_batch_size, unsupervised_batch_size, patience, checkpoint_path, verbose)

    if verbose > 0: print("test the model on test set")

    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="bw_test_accuracy")
    test_loss = tf.keras.metrics.Mean('bw_test_loss', dtype=tf.float32)

    t_loss, t_acc = model.test(features, o_h_labels, mask_test, L_s, test_accuracy, test_loss)
    print("Test acc {:.3f}" .format(t_acc))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Planetoid')

    # modality can be I inductive T transductive
    parser.add_argument("-m", "--modality", help="model to use", default="I", choices=["I", "T"])
    
    # dataset choice
    parser.add_argument("-d", "--dataset", help="dataset to use", default="cora", choices=["citeseer", "cora", "pubmed"])
    parser.add_argument("-y", "--yang-splits", help="whether to use Yang splits or not", default=True, action='store_true')
    
    # network hyperparameters
    parser.add_argument("-emb", "--embedding-dim", help="node embedding size", default=50, type=int)

    # optimization parameters
    parser.add_argument("-e", "--epochs", help="training epochs", default=10, type=int)
    parser.add_argument("-it", "--pretrain-batch", help="pretraining batches number", default=10400, type=int)
    parser.add_argument("-t1", "--supervised-batch", help="supervised batch number at each epoch", default=1.0, type=float)
    parser.add_argument("-t2", "--unsupervised-batch", help="unsupervised batch number at each epoch", default=0.1, type=float)
    parser.add_argument("-n1", "--supervised-batch-size", help="supervised mini-batch size", default=200, type=int)
    parser.add_argument("-n2", "--unsupervised-batch-size", help="unsupervised mini-batch size", default=20, type=int)    
    parser.add_argument("-lrs", "--learning-rate-supervised", help="supervised learning rate", default=1e-1, type=float)
    parser.add_argument("-lru", "--learning-rate-unsupervised", help="unsupervised learning rate", default=1e-3, type=float)
    parser.add_argument("-p", "--patience", help="patience for early stop", default=-1, type=int)
    
    # sampling algorithm (Alg.1) hyper-parameters
    parser.add_argument("-q", "--random-walk-length", help="random walk length", default=10, type=int)
    parser.add_argument("-w", "--window-size", help="window size", default=3, type=int)
    parser.add_argument("-r1", "--neg-sample-rate", help="negative sample rate", default=0, type=float)
    parser.add_argument("-r2", "--sample-context-rate", help="context sample with label rate", default=0.038, type=float)

    # reproducibility
    parser.add_argument("-ds", "--data-seed", help="seed to set in numpy before shuffling dataset", default=0, type=int)
    parser.add_argument("-ns", "--net-seed", help="seed to set in tensorflow before creating the neural network", default=0, type=int)

    # save model weights
    parser.add_argument("-cp", "--checkpoint-path", help="path for model checkpoints", default=None)

    # verbose
    parser.add_argument("-v", "--verbose", help="useful prints", default=1, type=int)

    args = parser.parse_args()
    
    main(args.modality, args.dataset, args.yang_splits,
        args.embedding_dim, args.epochs, args.pretrain_batch,
        args.supervised_batch, args.unsupervised_batch, args.supervised_batch_size,
        args.unsupervised_batch_size, args.learning_rate_supervised, 
        args.learning_rate_unsupervised, args.patience,
        args.random_walk_length, args.window_size, 
        args.neg_sample_rate, args.sample_context_rate,
        args.data_seed, args.net_seed, args.checkpoint_path, args.verbose)


