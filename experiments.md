## Reproduce Experiments
We use the open source library [Guild AI](https://guild.ai/) to simplify experiments reproducibility.

You can run every models defined in the [yaml file](guild.yml) as follows:

```sh
$ guild run model:train
```

or 

```sh
$ python model/train.py 
```

To reproduce our experimental results 

### Planetoid
```sh
$ guild run planetoid:evaluate100
```

### ChebNet
```sh
$ guild run chebnet:evaluate100
```

### GCN
```sh
$ guild run gcn:evaluate100
```

### GAT
```sh
$ guild run gat:evaluate100
```


## Calculate means and standard deviations

### Planetoid
```sh
$ guild compare -o planetoid --csv -> results/planetoid.csv
$ python3 results.py -p results/planetoid.csv
```

### ChebNet
```sh
$ guild compare -o chebnet --csv -> results/chebnet.csv
$ python3 results.py -p results/chebnet.csv
```

### GCN
```sh
$ guild compare -o gcn --csv -> results/gcn.csv
$ python3 results.py -p results/gcn.csv
```

### GAT
```sh
$ guild compare -o gat --csv -> results/gat.csv
$ python3 results.py -p results/gat.csv
```

## Test the models
In the test scripts you can test the models and create the t-SNE plot of the learned hidden space. 


### Example

* train and save a model  
`$ python3 GAT/train.py --dataset cora --checkpoint-path GAT_ckpt`
* test the model and make t-SNE plot   
`$ python3 GAT/test.py --dataset cora --checkpoint-path GAT_ckpt --tsne`

**T.B.N.:** 
- *--checkpoint-path* specifies the path wherethe model will be saved/loaded;
- *--tsne* specifies to create the t-SNE plot.