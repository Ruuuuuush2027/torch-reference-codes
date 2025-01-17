"""
Contains functionality for creating PyTorch DataLoader
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(train_dir, test_dir, transform: transforms.Compose,
                       batch_size: int, num_workers = NUM_WORKERS):
    """Creates training and testing loaders

    Takes in a training directory and a testing directory, turns into PyTorch
    Datasets and Dataloaders

    Args:
        train_dir: path to training directory
        test_dir: path to testing directory
        transform: the transform to perform on training and testing data
        batch_size: number of samples per batch
        num_workers: num_workers per data loader
    
    Returns:
        A tuple of (train_loader, test_loader, class_names).
        Class_names is a list of classes
    """
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                    transform=transform) # transforms to perform on data (images)

    test_data = datasets.ImageFolder(root=test_dir,
                                    transform=transform)
    class_names = train_data.classes
    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size, # how many samples per batch?
                                num_workers=num_workers, # how many subprocesses to use for data loading? (higher = more)
                                shuffle=True, # shuffle the data?
                                pin_memory = True) 

    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False, # don't usually need to shuffle testing data
                                pin_memory = True) 
    return train_dataloader, test_dataloader, class_names
