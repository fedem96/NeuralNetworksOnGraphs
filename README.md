# **Neural Networks on Graphs**

Deep learning has revolutionized many machine learning tasks in recent years, ranging from image classification
and video processing to speech recognition and natural language understanding. The data in these tasks are typically represented
in the Euclidean space. However, there is an increasing number of applications where data are generated from non-Euclidean domains and are represented as graphs with complex relationships and interdependency between objects. [[1]](#Wu)

There are many different approaches and algorithms that enable the use of Neural Networks on Graph data, the survey [[1]](#Wu) summarizes them. Good papers to read as an introduction to neural networks on graphs are [[2]](#Frasconi) and [[3]](#Gori).

# This project

In this project, we want to compare four different algorithms:

* Planetoid (2016) [[4]](#Yang)
* ChebNet (2016) [[5]](#Defferrard)
* GCN (2017) [[6]](#Kipf)
* GAT (2018) [[7]](#Velickovic)

on three different datasets [[8]](#Datasets):

* Citeseer [[9]](#Citeseer)
* Cora [[10]](#Cora)
* Pubmed Diabetes

reproducing experimental results reported in Tab.2 of [[7]](#Velickovic), i.e. calculating node-classification accuracies.

We use the same data split as in [[4](#Yang), [5](#Defferrard), [6](#Kipf), [7](#Velickovic)], which consists of 20 training nodes per class, 500 validation nodes, and 1000 test nodes: remaining nodes are used during training without their labels, i.e. we are in a semi-supervised setting.

We make 100 runs for each algorithm, changing the seed for the initialization of weights and reporting the average values with their standard deviation.

Besides, we set a fixed seed for weights initialization, and repeat 100 additional runs while changing the random data splits seed.

# Experimental results
For each algorithm, we report:
1. Original results reported in [[7]](#Velickovic)
2. Our results, 100 runs: same data splits as in [[4](#Yang), [5](#Defferrard), [6](#Kipf), [7](#Velickovic)], changing seeds for weights initialization
3. Our results, 100 runs: fixed weights initialization seed, changing seeds for data splits generation


| **Method**                | **Cora**    | **Citeseer** | **Pubmed**  |
|---------------------------|-------------|--------------|-------------|
|||||
| **Planetoid**<sup>1</sup> | 75.7%       | 64.7%        | 77.2%       |
| **Planetoid**<sup>2</sup> | 73.1 ± 0.8% | 62.3 ± 1.1%  | 73.7 ± 0.8% |
| **Planetoid**<sup>3</sup> | 72.2 ± 0.7% | 63.7 ± 0.7%  | 73.4 ± 0.2% |
|||||
| **ChebNet**<sup>1</sup>   | 81.2%       | 69.8%        | 74.4%       |
| **ChebNet**<sup>2</sup>   | 82.0 ± 0.6% | 70.5 ± 0.7%  | 75.2 ± 1.8% |
| **ChebNet**<sup>3</sup>   | 78.9 ± 1.8% | 68.2 ± 1.9%  | 73.4 ± 2.4% |
|||||
| **GCN**<sup>1</sup>       | 81.5%       | 70.3%        | 79.0%       |
| **GCN**<sup>2</sup>       | 80.6 ± 0.6% | 68.7 ± 0.9%  | 78.3 ± 0.5% |
| **GCN**<sup>3</sup>       | 79.2 ± 1.7% | 68.0 ± 1.8%  | 76.2 ± 2.5% |
|||||
| **GAT**<sup>1</sup>       | 83.0 ± 0.7% | 72.5 ± 0.7%  | 79.0 ± 0.3% |
| **GAT**<sup>2</sup>       | 83.1 ± 0.4% | 71.7 ± 0.7%  | 77.7 ± 0.4% |
| **GAT**<sup>3</sup>       | 81.0 ± 1.7% | 69.7 ± 1.7%  | 77.4 ± 2.4% |



# Reproducing experiments

## Download repository
* `git clone https://github.com/fedem96/NeuralNetworksOnGraphs.git`
* `cd NeuralNetworksOnGraphs`

## Install dependecies
* `pip install -r requirements.txt`

## Download (original) datasets
* `mkdir data`

### Citeseer
* `wget https://linqs-data.soe.ucsc.edu/public/lbc/citeseer.tgz`
* `tar -xf citeseer.tgz -C data`

### Cora
* `wget https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz`
* `tar -xf cora.tgz -C data`

### Pubmed
* `wget https://linqs-data.soe.ucsc.edu/public/Pubmed-Diabetes.tgz`
* `tar -xf Pubmed-Diabetes.tgz -C data`
* `mv data/Pubmed-Diabetes data/pubmed`

## Run the experiments
We use the open source library Guild AI[[11]](#GuildAI) to simplify experiments reproducibility.

### Planetoid
* `guild run planetoid-t:evaluate100`
* `guild run planetoid-i:evaluate100`

### ChebNet
* `guild run chebnet:evaluate100`

### GCN
* `guild run gcn:evaluate100`

### GAT
* `guild run gat:evaluate100`

## Calculate means and standard deviations

### Planetoid
* `guild compare -o planetoid-t --csv -> results/planetoid-t.csv`
* `guild compare -o planetoid-i --csv -> results/planetoid-i.csv`
* `python3 results.py -p results/planetoid-t.csv`
* `python3 results.py -p results/planetoid-i.csv`

### ChebNet
* `guild compare -o chebnet --csv -> results/chebnet.csv`
* `python3 results.py -p results/chebnet.csv`

### GCN
* `guild compare -o gcn --csv -> results/gcn.csv`
* `python3 results.py -p results/gcn.csv`

### ChebNet
* `guild compare -o gat --csv -> results/gat.csv`
* `python3 results.py -p results/gat.csv`


# References
<a name="Wu">[1]</a> Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang and P. S. Yu. *A Comprehensive Survey on Graph Neural Networks* (2019)  
<a name="Frasconi">[2]</a> P. Frasconi, M. Gori and A. Sperduti. *A general framework for adaptive processing of data structures* (1998)  
<a name="Gori">[3]</a> M. Gori, G. Monfardini and F. Scarselli. *A new model for learning in
graph domains* (2005)  
<a name="Yang">[4]</a> Z. Yang, W. Cohen and R. Salakhudinov. *Revisiting semi-supervised learning with graph embeddings* (2016)  
<a name="Defferrard">[5]</a> M. Defferrard, X. Bresson and P. Vandergheynst. *Convolutional
neural networks on graphs with fast localized spectral filtering* (2016)  
<a name="Kipf">[6]</a> T. N. Kipf and M. Welling. *Semi-supervised classification with graph
convolutional networks* (2017)  
<a name="Velickovic">[7]</a> P. Velickovic, G. Cucurull, A. Casanova, A. Romero, P. Lio and
Y. Bengio. *Graph attention networks* (2017)  
<a name="Datasets">[8]</a> Datasets: https://linqs.soe.ucsc.edu/data  
<a name="Citeseer">[9]</a> C. L. Giles, K. Bollacker and S. Lawrence. *Citeseer: An automatic citation indexing system* (1998)  
<a name="Cora">[10]</a> A. McCallum, K. Nigam, J. Rennie, and K. Seymore. *Automating
the construction of internet portals with machine learning* (2000)  
<a name="GuildAI">[11]</a> https://guild.ai/
