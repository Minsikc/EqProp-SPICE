from typing import Iterable
from pytorch_lightning import LightningDataModule
import torch
import numpy as np
from torch.utils.data import DataLoader
from pl_bolts.datamodules import *
from pl_bolts.datamodules.vision_datamodule import VisionDataModule

class miniMNIST16DataModule(MNISTDataModule):
    def __init__(self, *args, **kwargs):
        data_dir = '.data'
        super().__init__(data_dir, *args, **kwargs)

def load_datamodule(name:str, **kwargs):
    from torchvision import transforms
    DATASETS = {'MNIST': MNISTDataModule, 'CIFAR-10': CIFAR10DataModule}
    assert name in DATASETS, NotImplementedError(f'{name} is not in DATASETS. which are {DATASETS}')
    
    DMcls = DATASETS.get(name, None)
    DMcls:VisionDataModule
    dm = DMcls(data_dir=f'./data/{name}', batch_size=1, drop_last = True, **kwargs)
    
    
    dm.default_transforms = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((16,16)),
        transforms.Normalize((0.1307,), (0.3081/std_deviation,))]) if kwargs['transform'] is None else kwargs['transform']

    return dm

def load_dataset(name, **kwargs):
    if name =='XOR':
        return load_XOR()
    elif name =='EFL':
        return load_EFL()
    elif name =='MNIST':
        return load_MNIST()
    elif name =='mini_MNIST':
        # selected_classes = kwargs.get('selected_classes') or [0,1,2,3,4]
        # samples_per_cls = kwargs.get('samples_per_cls') or 600
        # std_deviation = kwargs.get('std_deviation') or 1
        return load_mini_MNIST(**kwargs)
    elif name =='mini_FMNIST':
        # selected_classes = kwargs.get('selected_classes') or [0,1,2,3,4]
        # samples_per_cls = kwargs.get('samples_per_cls') or 600
        # std_deviation = kwargs.get('std_deviation') or 1
        return load_mini_FMNIST(**kwargs)
    else:
        raise ValueError("Dataset \"{}\" undefined".format(name))

def pass_kwargs(func):
    """Pass kwargs to a function.

    Args:
        func (function): function to be called
        **kwargs: keyword arguments to be passed to the function

    Returns:
        function: function with kwargs passed in
    """
    def wrapper(**kwargs):
        samples_per_cls = kwargs.get('samples_per_cls') or 600
        selected_classes = kwargs.get('selected_classes') or [0,1,2,3,4]
        std_deviation = kwargs.get('std_deviation') or 1
        return func(selected_classes, samples_per_cls, std_deviation)
    return wrapper

def load_XOR() -> TensorDataset:
    """load XOR dataset

    Returns:
        TensorDataset: _description_
    """
    from torch.utils.data import TensorDataset
    X = torch.FloatTensor([[[-2, -2]], [[-2, 2]], [[2, -2]], [[2, 2]]])
    Y = torch.FloatTensor([[1,0], [0,1], [1,0], [0,1]])
    dataset = TensorDataset(X, Y)
    return dataset

def load_EFL() -> TensorDataset:
    """_summary_

    Returns:
        TensorDataset: _description_
    """
    from torch.utils.data import TensorDataset
    li = [1,1,1,1,1]
    c = [1,0,0,0,0]
    E = np.array([li,c,li,c,li])
    F = np.array([li,c,li,c,c])
    L = np.array([c,c,c,c,li])
    EFL = np.array([E,F,L])
    X = torch.from_numpy(EFL)
    Y = torch.FloatTensor([[1,0,0], [0,1,0], [0,0,1]])
    return TensorDataset(X, Y)


def load_MNIST(std_deviation=1):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((28,28)),
        transforms.Normalize((0.1307,), (0.3081/std_deviation,))])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform  
    )

    return dataset

@pass_kwargs
def load_mini_FMNIST(selected_classes, samples_per_cls, std_deviation):
    """load 16-by-16 resized 5 classes subset of FMNIST

    Returns:
        torchvision.dataset: subset of FMNIST
    """
    from torchvision import datasets, transforms
    import numpy as np

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((16,16)),
        transforms.Normalize((0.2859,), (0.3530/std_deviation,))])  # origin: (0,1) -> 

    dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform  
    )

    return make_subset(dataset, samples_per_cls, selected_classes)

@pass_kwargs
def load_mini_MNIST(selected_classes, samples_per_cls=600, std=1):
    """load 16-by-16 resized 5 classes subset of MNIST

    Returns:
        torchvision.dataset: subset of MNIST
    """
    from torchvision import datasets, transforms
    import numpy as np

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((16,16)),
        transforms.Normalize((0.1307,), (0.3081/std,))])  # origin: (0,1) -> 

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform  
    )
    
    return make_subset(dataset, samples_per_cls, selected_classes)

def make_subset(dataset, samples_per_cls:int, selected_classes:Iterable):
    """
    Split the indices in a stratified way
    and make a subset of the dataset
    """
    from torch.utils.data import Subset
    idx = []
    for cls in selected_classes: 
        idx += list(np.where((dataset.targets == cls) == True)[0][:samples_per_cls])
    subset = Subset(dataset, idx)
    return subset
