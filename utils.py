import os, csv
import numpy as np

DIR_NAME = os.path.dirname(os.path.realpath(__file__))
data = DIR_NAME + '/data'

def read_dataset(dataset):
    if "pubmed" in dataset:
        return read_p(dataset)
    else:
        return read_cc(dataset)


def read_cc(dataset):
    folder = os.path.join(data, dataset)
    ...

def read_p(dataset):
    features = []
    neighbors = []
    labels = []
    keys = []

    t_dict = {}
    folder = os.path.join(data, dataset)
    nodes = os.path.join(folder, 'data/Pubmed-Diabetes.NODE.paper.tab')
    edges = os.path.join(folder, 'data/Pubmed-Diabetes.DIRECTED.cites.tab')

    with open(nodes) as csvFile:       
        for idx, row in enumerate(csvFile):             
            if (idx<2):
                if ('cat' in row):
                    indices = row.replace('\n','').replace('numeric:w-','').replace(':0.0','').split('\t')[1:]
            else:
                node_feature = np.zeros(500)
                node = row.replace('\n','').replace('w-','').split('\t')
                labels.append(int(node[1].replace('label=','')))
                keys.append(node[0])
                for i in range(2,len(node)-1):
                    k, v = node[i].split('=')
                    node_feature[indices.index(k)] = v        
                features.append(node_feature)

    neighbors = [[] for i in range(len(features))]

    with open(edges) as csvFile:
        for idx, row in enumerate(csvFile):
            if (idx>2):
                node = row.replace('\n','').replace('paper:','').split('\t')
                node.remove('|')
                neighbors[keys.index(node[1])].append([node[2],int(node[0])])

    return features, neighbors, labels, keys
    

    
if __name__ == '__main__':
    read_dataset('pubmed')