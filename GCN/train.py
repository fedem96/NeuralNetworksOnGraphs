import argparse

import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from add_parent_path import add_parent_path

from models import GCN

with add_parent_path():
    from metrics import *
    from utils import *


# TODO: refactor code
def main(dataset_name, yang_splits,
        dropout_rate, hidden_units,
        training_epochs, learning_rate, l2_weight, patience,
        data_seed, net_seed,
        model_path, verbose):
    
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

    # add self loops to adj matrix
    A = A + sp.eye(A.shape[0])
    num_classes = get_num_classes(dataset_name)
    features = normalize_features(features)

    y_train = np.multiply(o_h_labels, np.broadcast_to(mask_train.T, o_h_labels.T.shape).T )
    y_val   = np.multiply(o_h_labels, np.broadcast_to(mask_val.T,   o_h_labels.T.shape).T )
    y_test  = np.multiply(o_h_labels, np.broadcast_to(mask_test.T,  o_h_labels.T.shape).T )

    if verbose > 0: print("calculating renormalized matrix")
    renormalized_matrix = renormalization_matrix(A)

    num_nodes = A.shape[0]
    num_features = len(features[0])

    if verbose > 0: print("defining model")
    model = GCN(renormalized_matrix, num_classes, dropout_rate, hidden_units)
    model.compile(
        loss=lambda y_true, y_pred: masked_loss(y_true, y_pred, 'categorical_crossentropy') + l2_weight * tf.nn.l2_loss(model.trainable_weights[0]), # regularize first layer only
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[masked_accuracy],
        # run_eagerly=True
    )
    model.build(features.shape)
    if verbose > 0: model.summary()

    if verbose > 0: print("begin training")
    callbacks = []
    callbacks.append(EarlyStoppingAvg(monitor='val_loss', mode='min', min_delta=0, patience=patience, restore_best_weights=True, verbose=verbose))
    callbacks.append(TensorBoard(log_dir='logs'))
    if model_path is not None:
        callbacks.append(ModelCheckpoint(monitor='val_loss', mode='min', filepath=model_path, save_best_only=True, save_weights_only=True, verbose=verbose))
    # input_shape: (num_nodes, num_features) -> output_shape: (num_nodes, num_classes)
    model.fit(features, y_train, epochs=training_epochs, batch_size=len(features), shuffle=False, validation_data=(features, y_val), callbacks=callbacks, verbose=verbose)
    if model_path is not None:
        model.load_weights(model_path)

    file_writer = tf.summary.create_file_writer("./logs/results/")
    file_writer.set_as_default()

    # log best performances on train and val set
    loss, accuracy = model.evaluate(features, y_train, batch_size=len(features), verbose=0)
    print("accuracy on training: " + str(accuracy))
    tf.summary.scalar('bw_loss', data=loss, step=1)
    tf.summary.scalar('bw_accuracy', data=accuracy, step=1)

    v_loss, v_accuracy = model.evaluate(features, y_val, batch_size=len(features), verbose=0)
    print("accuracy on validation: " + str(v_accuracy))
    tf.summary.scalar('bw_val_loss', data=v_loss, step=1)
    tf.summary.scalar('bw_val_accuracy', data=v_accuracy, step=1)
    tf.summary.scalar('bw_epoch', data=callbacks[0].stopped_epoch, step=1)
    
    if verbose > 0: print("test the model on test set")
    t_loss, t_accuracy = model.evaluate(features, y_test, batch_size=len(features), verbose=0)
    print("accuracy on test: " + str(t_accuracy))
    tf.summary.scalar('bw_test_loss', data=t_loss, step=1)
    tf.summary.scalar('bw_test_accuracy', data=t_accuracy, step=1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train GCN')

    # dataset choice
    parser.add_argument("-d", "--dataset", help="dataset to use", default="cora", choices=["citeseer", "cora", "pubmed"])
    parser.add_argument("-y", "--yang-splits", help="whether to use Yang splits or not", default=False, action='store_true')

    # network hyperparameters
    parser.add_argument("-dr", "--dropout-rate", help="dropout rate for dropout layers (fraction of the input units to drop)", default=0.5, type=float)
    parser.add_argument("-hu", "--hidden-units", help="number of Graph Convolutional filters in the first layer", default=16, type=int)

    # optimization hyperparameters
    parser.add_argument("-e", "--epochs", help="number of training epochs", default=200, type=int)
    parser.add_argument("-lr", "--learning-rate", help="starting learning rate of Adam optimizer", default=0.01, type=float)
    parser.add_argument("-l2w", "--l2-weight", help="l2 weight for regularization of first layer", default=5e-4, type=float)
    parser.add_argument("-p", "--patience", help="patience for early stop", default=10, type=int)

    # reproducibility
    parser.add_argument("-ds", "--data-seed", help="seed to set in numpy before shuffling dataset", default=0, type=int)
    parser.add_argument("-ns", "--net-seed", help="seed to set in tensorflow before creating the neural network", default=0, type=int)

    # save model to file
    parser.add_argument("-cp", "--checkpoint-path", help="path where to save the weights", default=None)

    # verbose
    parser.add_argument("-v", "--verbose", help="useful prints", default=1, type=int)

    args = parser.parse_args()
    main(args.dataset, args.yang_splits,
        args.dropout_rate, args.hidden_units,
        args.epochs, args.learning_rate, args.l2_weight, args.patience,
        args.data_seed, args.net_seed,
        args.checkpoint_path, args.verbose)