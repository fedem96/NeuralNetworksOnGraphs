def read_dataset(dataset):
    if "pubmed" in dataset:
        return read_p(dataset)
    else
        return read_cc(dataset)


def read_cc(dataset):
    folder = os.path.join(data, dataset)
    ...

def read_p(dataset):
    folder = os.path.join(data, dataset)
    ...
