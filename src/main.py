from distillation import distillation_experiment
import os
from datasets import MNISTDataset, FashionMNISTDataset, KMNISTDataset, IMDBDataset, EMNISTLettersDataset, OracleMNISTDataset
from util import load_or_create
import numpy as np
import random
# set seeds
np.random.seed(0)
random.seed(0)

"""
So far, these are the best params:

distilled:
    (kmnist_dataset, "KMNIST", 
        {
            "teacher": { "C": 2000, "T": 100, "s": 8.2, "epochs": 60 },
            "student": { "C": 200, "T": 100, "s": 8.2, "epochs": 120 },
            "temperature": 4.0,
            "alpha": 0.5,
        },
        {"overwrite": False}
    ),
    (mnist_dataset, "MNIST", 
        {
            "teacher": { "C": 1000, "T": 10, "s": 4.0, "epochs": 60 },
            "student": { "C": 100, "T": 10, "s": 4.0, "epochs": 120 },
            "temperature": 3.0,
        },
        {"overwrite": False}
    ),
    (emnist_dataset, "EMNIST", 
        {
            "teacher": { "C": 1000, "T": 100, "s": 4.0, "epochs": 60 },
            "student": { "C": 100, "T": 100, "s": 4.0, "epochs": 120 },
            "temperature": 4.0,
            "alpha": 0.5,
        },
        {"overwrite": False}
    ),  
"""

if __name__ == "__main__":
    # load datasets.
    print("Loading datasets...")
    kmnist_dataset = load_or_create(os.path.join("data", "kmnist_dataset.pkl"), KMNISTDataset)
    mnist_dataset = load_or_create(os.path.join("data", "mnist_dataset.pkl"), MNISTDataset)
    fashion_mnist_dataset = load_or_create(os.path.join("data", "fashion_mnist_dataset.pkl"), FashionMNISTDataset)
    imdb_dataset = load_or_create(os.path.join("data", "imdb_dataset.pkl"), IMDBDataset)
    emnist_dataset = load_or_create(os.path.join("data", "emnist_dataset.pkl"), EMNISTLettersDataset)
    oracle_mnist_dataset = load_or_create(os.path.join("data", "oracle_mnist_dataset.pkl"), OracleMNISTDataset)

    print("Datasets loaded")
        
    #run distilled experiments
    # this goes (dataset, name, params, kwargs)
    one_off_dir = os.path.join("results")
    distilled_experiments = [
        (kmnist_dataset, "KMNIST", 
            {
                "teacher": { "C": 2000, "T": 100, "s": 8.2, "epochs": 60 },
                "student": { "C": 200, "T": 100, "s": 8.2, "epochs": 120 },
                "temperature": 4.0,
                "alpha": 0.5,
            },
            {"overwrite": False}
        ),
        (mnist_dataset, "MNIST", 
            {
                "teacher": { "C": 1000, "T": 10, "s": 4.0, "epochs": 60 },
                "student": { "C": 100, "T": 10, "s": 4.0, "epochs": 120 },
                "temperature": 3.0,
                "alpha": 0.5,
            },
            {"overwrite": False}
        ),
        (emnist_dataset, "EMNIST", 
            {
                "teacher": { "C": 1000, "T": 100, "s": 4.0, "epochs": 60 },
                "student": { "C": 100, "T": 100, "s": 4.0, "epochs": 120 },
                "temperature": 4.0,
                "alpha": 0.5,
            },
            {"overwrite": False}
        ),  
        (imdb_dataset, "IMDB", 
            {
                "teacher": { "C": 10000, "T": 8000, "s": 4.0, "epochs": 60 },
                "student": { "C": 2000, "T": 8000, "s": 4.0, "epochs": 120 },
                "temperature": 3.0,
                "alpha": 0.5,
            },
            {"overwrite": False, "make_activation_maps": False}
        ),
    ]
    
    print("Running distilled experiments")
    for dataset, name, params, kwargs in distilled_experiments:
        kwargs["folderpath"] = one_off_dir
        kwargs["save_all"] = True
        distillation_experiment(dataset, name, params, **kwargs)
        
    