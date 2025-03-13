from distillation import distribution_distillation_experiment, plot_results, clause_distillation_experiment
from activation_maps import visualize_activation_maps
import os
from datasets import MNISTDataset, FashionMNISTDataset, KMNISTDataset, IMDBDataset, EMNISTLettersDataset, OracleMNISTDataset
from util import load_or_create, load_json, save_json, load_pkl
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

def remake_plots(parent_dir):
    for fpath in os.listdir(parent_dir):
        print(fpath)
        if not os.path.exists(os.path.join(parent_dir, fpath, OUTPUT_JSON_PATH)):
            continue
        output = load_json(os.path.join(parent_dir, fpath, OUTPUT_JSON_PATH))
        if "downsample" in output["params"]:
            plot_results(output, os.path.join(parent_dir, fpath), output["params"]["downsample"])
        else:
            plot_results(output, os.path.join(parent_dir, fpath))

if __name__ == "__main__":
    # load datasets.
    print("Loading datasets...")
    kmnist_dataset = load_or_create(os.path.join("data", "kmnist_dataset.pkl"), KMNISTDataset)
    mnist_dataset = load_or_create(os.path.join("data", "mnist_dataset.pkl"), MNISTDataset)
    fashion_mnist_dataset = load_or_create(os.path.join("data", "fashion_mnist_dataset.pkl"), FashionMNISTDataset)
    imdb_dataset = load_or_create(os.path.join("data", "imdb_dataset.pkl"), IMDBDataset)
    emnist_dataset = load_or_create(os.path.join("data", "emnist_dataset.pkl"), EMNISTLettersDataset)
    print("Datasets loaded")
        
    one_off_dir = os.path.join("results")
    clause_dir = os.path.join(one_off_dir, "clause")
    distribution_dir = os.path.join(one_off_dir, "distribution")

    print(f"Remaking all plots in {distribution_dir}...")
    remake_plots(distribution_dir)
    print(f"Remaking all plots in {clause_dir}...")
    remake_plots(clause_dir)
    
    clause_distilled_experiments = [
        (mnist_dataset, "MNIST", 
            {
                "teacher": { "C": 800, "T": 10, "s": 7.0, "epochs": 120 },
                "student": { "C": 100, "T": 10, "s": 7.0, "epochs": 240 },
                "downsample": 0.15,
            },
            {"overwrite": False}
        ),
        (kmnist_dataset, "KMNIST", 
            {
                "teacher": { "C": 400, "T": 100, "s": 5, "epochs": 120 },
                "student": { "C": 100, "T": 100, "s": 5, "epochs": 240 },
                "downsample": 0.22,
            },
            {"overwrite": False}
        ),
        (imdb_dataset, "IMDB", 
            {
                "teacher": { "C": 10000, "T": 6000, "s": 5.0, "epochs": 30 },
                "student": { "C": 2000, "T": 6000, "s": 5.0, "epochs": 90 },
                "downsample": 0.25,
            },
            {"overwrite": False}
        ),
    ]

    print("Running clause-based distilled experiments")
    for dataset, name, params, kwargs in clause_distilled_experiments:
        kwargs["folderpath"] = clause_dir
        kwargs["save_all"] = True
        clause_distillation_experiment(dataset, name, params, **kwargs)

    #run distilled experiments
    # this goes (dataset, name, params, kwargs)
    distribution_distilled_experiments = [
        (mnist_dataset, "MNIST", 
            {
                "teacher": { "C": 1000, "T": 10, "s": 4.0, "epochs": 120 },
                "student": { "C": 100, "T": 10, "s": 4.0, "epochs": 240 },
                "temperature": 3.0,
                "alpha": 0.5,
                "z": 0.3,
            },
            {"overwrite": False}
        ),
        (kmnist_dataset, "KMNIST", 
            {
                "teacher": { "C": 2000, "T": 100, "s": 8.2, "epochs": 120 },
                "student": { "C": 200, "T": 100, "s": 8.2, "epochs": 240 },
                "temperature": 4.0,
                "alpha": 0.5,
                "z": 0.3,
            },
            {"overwrite": False}
        ),
        (emnist_dataset, "EMNIST", 
            {
                "teacher": { "C": 1000, "T": 100, "s": 7.0, "epochs": 120 },
                "student": { "C": 100, "T": 100, "s": 7.0, "epochs": 240 },
                "temperature": 4.0,
                "alpha": 0.5,
                "z": 0.2,
            },
            {"overwrite": False}
        ),  
        (imdb_dataset, "IMDB", 
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
    
    print("Running distribution-based distilled experiments")
    for dataset, name, params, kwargs in distribution_distilled_experiments:
        kwargs["folderpath"] = distribution_dir
        kwargs["save_all"] = True
        distribution_distillation_experiment(dataset, name, params, **kwargs)

   
    exit()
    
    print("Updating charts")
    # activation maps
    for fpath in os.listdir(one_off_dir):
        output = load_json(os.path.join(one_off_dir, fpath, OUTPUT_JSON_PATH))
        if output["experiment_name"] == "IMDB":
            continue
        # get dataset from output
        dataset = locals()[output["experiment_name"].lower()+"_dataset"]
        distilled_model = load_pkl(os.path.join(one_off_dir, fpath, "distilled.pkl"))
        teacher_model = load_pkl(os.path.join(one_off_dir, fpath, "teacher_baseline.pkl"))
        student_model = load_pkl(os.path.join(one_off_dir, fpath, "student_baseline.pkl"))

        samples = np.random.randint(0, len(dataset.X_train), size=4)
        plot_results(output, os.path.join(one_off_dir, fpath))
        visualize_activation_maps(teacher_model, student_model, distilled_model, 
                                dataset.X_train[samples], dataset.Y_train[samples], dataset.image_shape, os.path.join(one_off_dir, fpath, output["experiment_name"]+"_activation_maps.png"))
