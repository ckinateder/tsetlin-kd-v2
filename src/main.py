from distillation import distillation_experiment, plot_results
import os
from datasets import MNISTDataset, FashionMNISTDataset, KMNISTDataset, IMDBDataset, EMNISTLettersDataset, OracleMNISTDataset
from util import load_or_create, load_json, save_json
from __init__ import *
import pandas as pd
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
    (imdb_dataset, "IMDB", 
        {
            "teacher": { "C": 8000, "T": 6000, "s": 7.0, "epochs": 30 },
            "student": { "C": 4000, "T": 6000, "s": 7.0, "epochs": 60 },
            "temperature": 3.0,
            "alpha": 0.5,
        },
        {"overwrite": False, "make_activation_maps": False}
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

    print("Datasets loaded")
        
    #run distilled experiments
    # this goes (dataset, name, params, kwargs)
    one_off_dir = os.path.join("results")
    distilled_experiments = [
        (mnist_dataset, "MNIST-top-indices", 
            {
                "teacher": { "C": 1000, "T": 10, "s": 4.0, "epochs": 60 },
                "student": { "C": 100, "T": 10, "s": 4.0, "epochs": 120 },
                "temperature": 3.0,
                "alpha": 0.5,
                "z": 0.2,
            },
            {"overwrite": False}
        ),
        (emnist_dataset, "EMNIST-top-indices", 
            {
                "teacher": { "C": 1000, "T": 100, "s": 4.0, "epochs": 60 },
                "student": { "C": 100, "T": 100, "s": 4.0, "epochs": 120 },
                "temperature": 4.0,
                "alpha": 0.5,
                "z": 0.2,
            },
            {"overwrite": False}
        ),  
        (kmnist_dataset, "KMNIST-top-indices", 
            {
                "teacher": { "C": 2000, "T": 100, "s": 8.2, "epochs": 60 },
                "student": { "C": 200, "T": 100, "s": 8.2, "epochs": 120 },
                "temperature": 4.0,
                "alpha": 0.5,
                "z": 0.2,
            },
            {"overwrite": False}
        ),
        (imdb_dataset, "IMDB-top-indices", 
            {
                "teacher": { "C": 8000, "T": 6000, "s": 7.0, "epochs": 30 },
                "student": { "C": 4000, "T": 6000, "s": 7.0, "epochs": 60 },
                "temperature": 3.0,
                "alpha": 0.5,
                "z": 0.2,
            },
            {"overwrite": False, "make_activation_maps": False}
        ),
    ]
    
    print("Running distilled experiments")
    for dataset, name, params, kwargs in distilled_experiments:
        kwargs["folderpath"] = one_off_dir
        kwargs["save_all"] = True
        distillation_experiment(dataset, name, params, **kwargs)
        