import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from add_parent_path import add_parent_path

from models import ChebNet

with add_parent_path():
    from metrics import *
    from utils import *

def main(dataset_name,
        dropout_rate, K, hidden_units,
        l2_weight,
        data_seed,
        model_path, verbose,
        tsne):
    
    # reproducibility
    np.random.seed(data_seed)

    if verbose > 0: print("reading dataset")
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset_name)
    num_classes = len(set(labels))

    if verbose > 0: print("shuffling dataset")
    features, neighbors, labels, o_h_labels, keys = permute(features, neighbors, labels, o_h_labels, keys)
    features = normalize_features(features)

    if verbose > 0: print("obtaining masks")
    mask_train, mask_val, mask_test = split(dataset_name, labels)
    y_train = np.multiply(o_h_labels, np.broadcast_to(mask_train.T, o_h_labels.T.shape).T )
    y_val   = np.multiply(o_h_labels, np.broadcast_to(mask_val.T,   o_h_labels.T.shape).T )
    y_test  = np.multiply(o_h_labels, np.broadcast_to(mask_test.T,  o_h_labels.T.shape).T )

    if verbose > 0: print("calculating adjacency matrix")
    A = adjacency_matrix(neighbors)
    if verbose > 0: print("calculating scaled normalized laplacian matrix")
    scaled_norm_L = scaled_normalized_laplacian_matrix(A)

    num_nodes = A.shape[0]
    num_features = len(features[0])

    if verbose > 0: print("defining model")
    model = ChebNet(scaled_norm_L, K, num_classes, dropout_rate, hidden_units)
    model.compile(
        loss=lambda y_true, y_pred: masked_loss(y_true, y_pred, 'categorical_crossentropy') + l2_weight * tf.nn.l2_loss(model.trainable_weights[0]), # regularize first layer only
        #optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[masked_accuracy],
        # run_eagerly=True
    )
    model.build(features.shape)
    model.summary()
    model.load_weights(os.path.join(model_path, "ckpt")).expect_partial()

    if verbose > 0: print("test the model on test set")
    loss, accuracy = model.evaluate(features, y_test, batch_size=num_nodes, verbose=0)
    print("accuracy on test: " + str(accuracy))

    if tsne:
        if verbose > 0: print("calculating t-SNE plot")
        intermediate_layer_model = tf.keras.Sequential([model.layers[0], model.layers[1]])
        intermediate_output = intermediate_layer_model.predict(features, batch_size=num_nodes)
        plot_tsne(intermediate_output[mask_test], labels[mask_test], len(o_h_labels[0]), 'ChebNet')

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a ChebNet')

    # dataset choice
    parser.add_argument("-d", "--dataset", help="dataset to use", default="pubmed", choices=["citeseer", "cora", "pubmed"])

    # network hyperparameters
    parser.add_argument("-dr", "--dropout-rate", help="dropout rate for dropout layers (fraction of the input units to drop)", default=0.5, type=float)
    parser.add_argument("-K", "--num-polynomials", help="number of Chebychev polynomials (there will be used polynomials from order 0 to K-1)", default=4, type=int)
    parser.add_argument("-hu", "--hidden-units", help="number of Chebychev filters in the first layer", default=16, type=int)

    # optimization hyperparameters
    parser.add_argument("-l2w", "--l2-weight", help="l2 weight for regularization of first layer", default=5e-4, type=float)

    # reproducibility
    parser.add_argument("-ds", "--data-seed", help="seed to set in numpy before shuffling dataset", default=0, type=int)

    # save model to file
    parser.add_argument("-cp", "--checkpoint-path", help="path where to save the weights", default=None)

    # verbose
    parser.add_argument("-v", "--verbose", help="useful prints", default=1, type=int)

    # tsne
    parser.add_argument("-t", "--tsne", help="whether to make t-SNE plot or not", default=False, action='store_true')

    args = parser.parse_args()
    main(args.dataset,
        args.dropout_rate, args.num_polynomials, args.hidden_units,
        args.l2_weight,
        args.data_seed,
        args.checkpoint_path, args.verbose,
        args.tsne)