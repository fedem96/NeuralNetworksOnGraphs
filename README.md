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
* Pubmed Diabetes <!-- [[11]](#Pubmed) -->

reproducing experimental results reported in Tab.2 of [[7]](#Velickovic), i.e. calculating node-classification accuracies.

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

... (IN PROGRESS) ...

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
<!-- <a name="Pubmed">[11]</a> -->

# WORK IN PROGRESS...
