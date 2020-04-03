import argparse
import numpy as np

import tensorflow as tf
from add_parent_path import add_parent_path

from models import Planetoid_I, Planetoid_T

with add_parent_path():
    from metrics import UnlabeledLoss
    from utils import *


def main(modality, dataset_name, 
        embedding_dim, epochs, pretrain_batch,
        supervised_batch, unsupervised_batch, supervised_batch_size,
        unsupervised_batch_size, learning_rate_supervised, 
        learning_rate_unsupervised, random_walk_length, 
        window_size, neg_sample, sample_context_rate,
        data_seed, net_seed):
    
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

    # Define model, loss, metrics and optimizers
    if modality == "I":
        model = Planetoid_I(neighbors, o_h_labels, embedding_dim, random_walk_length, window_size, neg_sample, sample_context_rate)
    elif modality == "T":
        model = Planetoid_T(neighbors, o_h_labels, embedding_dim, random_walk_length, window_size, neg_sample, sample_context_rate)

    L_s = tf.keras.losses.CategoricalCrossentropy()
    L_u = UnlabeledLoss(unsupervised_batch_size)

    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name="train_acc")
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name="test_acc")
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_loss_u = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)

    optimizer_u = tf.keras.optimizers.SGD(learning_rate=learning_rate_unsupervised)       # , momentum=0.99)
    optimizer_s = tf.keras.optimizers.SGD(learning_rate=learning_rate_supervised)       # , momentum=0.99)

    print("pre-train model")

    # Pretrain iterations on graph context
    model.pretrain_step(features, mask_test, L_u, optimizer_u, train_loss_u, pretrain_batch, unsupervised_batch_size)

    print("begin training")
    model.train(features, o_h_labels, mask_train, mask_test, epochs, L_s, L_u, optimizer_u, optimizer_s, train_accuracy, test_accuracy, 
        train_loss, train_loss_u, test_loss, supervised_batch, unsupervised_batch, supervised_batch_size, unsupervised_batch_size)

    # TODO: test


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train Planetoid')

    # modality can be I inductive T transductive
    parser.add_argument("-m", "--modality", help="model to use", default="I", choices=["I", "T"])
    
    # dataset choice
    parser.add_argument("-d", "--dataset", help="dataset to use", default="citeseer", choices=["citeseer", "cora", "pubmed"])
    
    # network hyperparameters
    parser.add_argument("-emb", "--embedding-dim", help="number of training epochs", default=50, type=int)

    # optimization parameters
    parser.add_argument("-e", "--epochs", help="number of training epochs", default=10, type=int)
    parser.add_argument("-it", "--pretrain-batch", help="number of pretraining batches", default=400, type=int)
    parser.add_argument("-t1", "--supervised-batch", help="number of training epochs", default=1.0, type=float)
    parser.add_argument("-t2", "--unsupervised-batch", help="number of training epochs", default=0.1, type=float)
    parser.add_argument("-n1", "--supervised-batch-size", help="number of training epochs", default=200, type=int)
    parser.add_argument("-n2", "--unsupervised-batch-size", help="number of training epochs", default=20, type=int)    
    parser.add_argument("-lrs", "--learning-rate-supervised", help="supervised learning rate", default=1e-1, type=float)
    parser.add_argument("-lru", "--learning-rate-unsupervised", help="unsupervised learning rate", default=1e-3, type=float)
    
    # sampling algorithm parameters
    parser.add_argument("-q", "--random-walk-length", help="number of training epochs", default=10, type=int)
    parser.add_argument("-w", "--window-size", help="number of training epochs", default=3, type=int)
    parser.add_argument("-r1", "--neg-sample-rate", help="number of training epochs", default=5/6, type=float)
    parser.add_argument("-r2", "--sample-context-rate", help="number of training epochs", default=5/6, type=float)

    # reproducibility
    parser.add_argument("-ds", "--data-seed", help="seed to set in numpy before shuffling dataset", default=0, type=int)
    parser.add_argument("-ns", "--net-seed", help="seed to set in tensorflow before creating the neural network", default=0, type=int)

    args = parser.parse_args()
    
    main(args.modality, args.dataset, 
        args.embedding_dim, args.epochs, args.pretrain_batch,
        args.supervised_batch, args.unsupervised_batch, args.supervised_batch_size,
        args.unsupervised_batch_size, args.learning_rate_supervised, 
        args.learning_rate_unsupervised, args.random_walk_length, 
        args.window_size, args.neg_sample_rate, args.sample_context_rate,
        args.data_seed, args.net_seed)


