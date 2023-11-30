from torch_geometric.datasets import Planetoid, PPI
from torch_geometric.transforms import NormalizeFeatures


def get_dataset(dataset_name):
    """
    Download and save the data of `dataset_name` to `../data` and subsequently return it after saving.

    :param dataset_name: Specify which dataset to load. Valid options are in {'Cora', 'CiteSeer', 'Pubmed', 'PPI'}.

    :return: The dataset.
    :raises ValueError: If `dataset_name` is not in {'Cora', 'CiteSeer', 'Pubmed', 'PPI'}.
    """

    if dataset_name not in {'Cora', 'CiteSeer', 'Pubmed', 'PPI'}:
        raise ValueError(f"Invalid dataset name {dataset_name}")

    # Download the dataset and normalize the features.
    if dataset_name == 'PPI':
        dataset = PPI(f'../data/inductive/{dataset_name}', transform=NormalizeFeatures())
    else:
        dataset = Planetoid(f'../data/transductive/{dataset_name}', dataset_name, transform=NormalizeFeatures())

    return dataset
