import argparse
import numpy as np
import pandas as pd

def compute_statistics(df, seed_value):

    statistics = np.zeros((2,2))

    opt_values = df[df["data-seed"]==seed_value]["best_test_accuracy"].values

    statistics[0][0] = np.mean(opt_values)
    statistics[0][1] = np.std(opt_values)
    
    net_values = df[df["net-seed"]==seed_value]["best_test_accuracy"].values

    statistics[1][0] = np.mean(net_values)
    statistics[1][1] = np.std(net_values)

    return statistics

def report_statistics(f_path, seed_value):
    
    # read the first 60 csv lines (last experiments)
    df = pd.read_csv(f_path, sep =',', usecols=["operation", "data-seed", "net-seed", "dataset", "best_test_accuracy"], nrows=60)
    # extract model name from csv
    model_name = df["operation"][0].split('|')[0]

    print(model_name)
    
    pub_values = df[df["dataset"]=="pubmed"].drop(["operation", "dataset"],1)
    st = compute_statistics(pub_values, seed_value)

    print("Pubmed: \n  Optimization seed: {:.2f} ± {:.2f}% " .format(st[0][0]*100, st[0][1]), end=' ')
    print("Data seed: {:.2f} ± {:.2f}%" .format(st[1][0]*100, st[1][1]))

    cora_values = df[df["dataset"]=="cora"].drop(["operation", "dataset"],1)
    st = compute_statistics(cora_values, seed_value)

    print("Cora: \n  Optimization seed: {:.2f} ± {:.2f}% " .format(st[0][0]*100, st[0][1]), end=' ')
    print("Data seed: {:.2f} ± {:.2f}%" .format(st[1][0]*100, st[1][1]))

    cite_values = df[df["dataset"]=="citeseer"].drop(["operation", "dataset"],1)
    st = compute_statistics(cite_values, seed_value)

    print("Citeseer: \n  Optimization seed: {:.2f} ± {:.2f}% " .format(st[0][0]*100, st[0][1]), end=' ')
    print("Data seed: {:.2f} ± {:.2f}%" .format(st[1][0]*100, st[1][1]))
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train GCN')
    parser.add_argument("-p", "--path", help="path to csv result file")
    parser.add_argument("-v", "--value", help="fixed seed value", default=5687, type=int)
    args = parser.parse_args()
    report_statistics(args.path, args.value)