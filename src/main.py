from distillation import distribution_distillation_experiment, plot_results
from activation_maps import visualize_activation_maps
import os
from stats import info_theory_experiment
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
    
    #run distilled experiments
    # this goes (dataset, name, params, kwargs)
    distilled_experiments = [
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
                "teacher": { "C": 1000, "T": 100, "s": 4.0, "epochs": 120 },
                "student": { "C": 100, "T": 100, "s": 4.0, "epochs": 240 },
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
    
    print("Running distilled experiments")
    for dataset, name, params, kwargs in distilled_experiments:
        kwargs["folderpath"] = one_off_dir
        kwargs["save_all"] = True
        distribution_distillation_experiment(dataset, name, params, **kwargs)
   
    exit()
    
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
        print(samples)
        visualize_activation_maps(teacher_model, student_model, distilled_model, 
                                dataset.X_train[samples], dataset.Y_train[samples], dataset.image_shape, os.path.join(one_off_dir, fpath, output["experiment_name"]+"_activation_maps.png"))

    print("Updating charts")
    # update all charts
    for fpath in os.listdir(one_off_dir):
        # modify output
        print(fpath)
        output = load_json(os.path.join(one_off_dir, fpath, OUTPUT_JSON_PATH))
        results = pd.DataFrame(output["results"])
        output["analysis"]["avg_acc_test_distilled"] = results[ACC_TEST_DISTILLED].mean()
        output["analysis"]["std_acc_test_distilled"] = results[ACC_TEST_DISTILLED].std()

        output["analysis"]["avg_acc_train_distilled"] = results[ACC_TRAIN_DISTILLED].mean()
        output["analysis"]["std_acc_train_distilled"] = results[ACC_TRAIN_DISTILLED].std()

        post_teacher_results = results.iloc[output["params"]["teacher"]["epochs"]:]
        output["analysis"]["avg_time_train_distilled"] = post_teacher_results[TIME_TRAIN_DISTILLED].mean()
        output["analysis"]["avg_time_test_distilled"] = post_teacher_results[TIME_TEST_DISTILLED].mean()

        output["analysis"]["inference_time_teacher"] = post_teacher_results[TIME_TEST_TEACHER].mean()
        output["analysis"]["inference_time_student"] = post_teacher_results[TIME_TEST_STUDENT].mean()
        output["analysis"]["inference_time_distilled"] = post_teacher_results[TIME_TEST_DISTILLED].mean()
        save_json(output, os.path.join(one_off_dir, fpath, OUTPUT_JSON_PATH))

        output = load_json(os.path.join(one_off_dir, fpath, OUTPUT_JSON_PATH))
        
        plot_results(output, os.path.join(one_off_dir, fpath))
