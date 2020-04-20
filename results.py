import argparse
import numpy as np
import pandas as pd

# outlier if far from the mean more than n*std
def reject_outliers(data, n=3):
    m = np.mean(data)
    s = np.std(data)
    return data[np.where(np.abs(data - m) < n * s)]

def compute_statistics(df, seed_value):

    statistics = np.zeros((2,2))

    opt_values = df[df["data-seed"]==seed_value]["bw_test_accuracy"].values
    prev_shape = opt_values.shape
    opt_values = reject_outliers(opt_values)
    outs_opt = -opt_values.shape[0] + prev_shape[0]
    print("opt seed outliers", outs_opt)


    statistics[0][0] = np.mean(opt_values) * 100 # transform to percentages
    statistics[0][1] = np.std(opt_values)  * 100
     
    net_values = df[df["net-seed"]==seed_value]["bw_test_accuracy"].values
    prev_shape = net_values.shape
    net_values = reject_outliers(net_values)
    outs_net = -net_values.shape[0] + prev_shape[0]
   
    print("net seed outliers", outs_net)

    statistics[1][0] = np.mean(net_values) * 100
    statistics[1][1] = np.std(net_values)  * 100

    return statistics

def report_statistics(f_path, seed_value, nrows):
    
    # read the first 60 csv lines (last experiments)
    df = pd.read_csv(f_path, sep =',', usecols=["operation", "data-seed", "net-seed", "dataset", "bw_test_accuracy"], nrows=nrows)
    # extract model name from csv
    model_name = df["operation"][0].split('|')[0]

    print(model_name)
    
    pub_values = df[df["dataset"]=="pubmed"].drop(["operation", "dataset"],1)
    st = compute_statistics(pub_values, seed_value)

    print("Pubmed: \n  Optimization seed: {:.2f} ± {:.2f}% " .format(st[0][0], st[0][1]), end=' ')
    print("Data seed: {:.2f} ± {:.2f}%" .format(st[1][0], st[1][1]))

    cora_values = df[df["dataset"]=="cora"].drop(["operation", "dataset"],1)
    st = compute_statistics(cora_values, seed_value)

    print("Cora: \n  Optimization seed: {:.2f} ± {:.2f}% " .format(st[0][0], st[0][1]), end=' ')
    print("Data seed: {:.2f} ± {:.2f}%" .format(st[1][0], st[1][1]))

    cite_values = df[df["dataset"]=="citeseer"].drop(["operation", "dataset"],1)
    st = compute_statistics(cite_values, seed_value)

    print("Citeseer: \n  Optimization seed: {:.2f} ± {:.2f}% " .format(st[0][0], st[0][1]), end=' ')
    print("Data seed: {:.2f} ± {:.2f}%" .format(st[1][0], st[1][1]))
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train GCN')
    parser.add_argument("-p", "--path", help="path to csv result file")
    parser.add_argument("-v", "--value", help="fixed seed value", default=5687, type=int)
    parser.add_argument("-r", "--runs", help="number of experimental runs", default=None, type=int)
    args = parser.parse_args()
    report_statistics(args.path, args.value, args.runs)
