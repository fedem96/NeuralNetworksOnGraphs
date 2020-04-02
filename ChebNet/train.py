import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from add_parent_path import add_parent_path

from models import ChebNet

with add_parent_path():
    from metrics import *
    from utils import *

# TODO: refactor code
def main(dataset_name,
        dropout_rate, K, hidden_units,
        training_epochs, learning_rate, l2_weight,
        data_seed, net_seed):
    
    # reproducibility
    np.random.seed(data_seed)
    tf.random.set_seed(net_seed)

    print("reading dataset")
    features, neighbors, labels, o_h_labels, keys = read_dataset(dataset_name)
    num_classes = len(set(labels))

    print("shuffling dataset")
    features, neighbors, labels, o_h_labels, keys = permute(features, neighbors, labels, o_h_labels, keys)
    features = normalize_features(features)

    print("obtaining masks")
    mask_train, mask_val, mask_test = split(dataset_name, labels)
    y_train = np.multiply(o_h_labels, np.broadcast_to(mask_train.T, o_h_labels.T.shape).T )
    y_val   = np.multiply(o_h_labels, np.broadcast_to(mask_val.T,   o_h_labels.T.shape).T )
    y_test  = np.multiply(o_h_labels, np.broadcast_to(mask_test.T,  o_h_labels.T.shape).T )

    print("calculating adjacency matrix")
    A = adjacency_matrix(neighbors)
    print("calculating scaled normalized laplacian matrix")
    scaled_norm_L = scaled_normalized_laplacian_matrix(A)

    num_nodes = A.shape[0]
    num_features = len(features[0])

    print("defining model")
    model = ChebNet(scaled_norm_L, K, num_classes, dropout_rate, hidden_units, learning_rate, l2_weight)

    print("begin training")
    tb = TensorBoard(log_dir='logs') #TODO: change dir
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto') # TODO: check if same as paper
    # input_shape: (num_nodes, num_features) -> output_shape: (num_nodes, num_classes)
    model.fit(features, y_train, epochs=training_epochs, batch_size=len(features), shuffle=False, validation_data=(features, y_val), callbacks=[tb, es])

    y_pred = model.predict(features, len(features))
    print("validation accuracy:", float(masked_accuracy(y_val, y_pred)))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a ChebNet')

    # dataset choice
    parser.add_argument("-d", "--dataset", help="dataset to use", default="cora", choices=["citeseer", "cora", "pubmed"])

    # network hyperparameters
    parser.add_argument("-dr", "--dropout-rate", help="dropout rate for dropout layers (fraction of the input units to drop)", default=0.5, type=float)
    parser.add_argument("-K", "--num-polynomials", help="number of Chebychev polynomials (there will be used polynomials from order 0 to K-1)", default=3, type=int)
    parser.add_argument("-hu", "--hidden-units", help="number of Chebychev filters in the first layer", default=16, type=int)

    # optimization hyperparameters
    parser.add_argument("-e", "--epochs", help="number of training epochs", default=10, type=int)
    parser.add_argument("-lr", "--learning-rate", help="starting learning rate of Adam optimizer", default=0.01, type=float)
    parser.add_argument("-l2w", "--l2-weight", help="l2 weight for regularization of first layer", default=5e-4, type=float)

    # reproducibility
    parser.add_argument("-ds", "--data-seed", help="seed to set in numpy before shuffling dataset", default=0, type=int)
    parser.add_argument("-ns", "--net-seed", help="seed to set in tensorflow before creating the neural network", default=0, type=int)

    args = parser.parse_args()
    main(args.dataset,
        args.dropout_rate, args.num_polynomials, args.hidden_units,
        args.epochs, args.learning_rate, args.l2_weight,
        args.data_seed, args.net_seed)