from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.transforms import NormalizeFeatures


def get_dataset(dataset_name):
    if dataset_name == 'PPI':
        dataset = PPI(f'../data/inductive/{dataset_name}', transform=NormalizeFeatures())
    else:
        dataset = Planetoid(f'../data/transductive/{dataset_name}', dataset_name, transform=NormalizeFeatures())

    # print(f"""Loaded: {dataset_name}
    # Classes: {dataset.num_classes}
    # Feature size: {dataset.num_node_features}""")
    return dataset
