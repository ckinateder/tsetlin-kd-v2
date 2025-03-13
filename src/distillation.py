import numpy as np
from time import time
import pandas as pd
import os
import matplotlib.pyplot as plt
from util import save_json, load_pkl, save_pkl, make_dir, rm_file, load_json
from datetime import datetime
from tqdm import tqdm
from pyTsetlinMachineParallel.tm import MultiClassTsetlinMachine
from datasets import Dataset
from __init__ import *
from prodict import Prodict
from activation_maps import visualize_activation_maps

DISTRIB_DISTILLED_DEFAULTS = {
    "teacher": {
        "C": 1000,
        "T": 10,
        "s": 5,
        "epochs": 30,
    },
    "student": {
        "C": 100,
        "T": 10,
        "s": 5,
        "epochs": 60,
    },
    "temperature": 4.0,
    "alpha": 0.5,
    "z": 0.2,
    "weighted_clauses": True,
    "number_of_state_bits": 8,
}

CLAUSE_DISTILLED_DEFAULTS = {
    "teacher": {
        "C": 1000,
        "T": 10,
        "s": 5,
        "epochs": 30,
    },
    "student": {
        "C": 100,
        "T": 10,
        "s": 5,
        "epochs": 60,
    },
    "downsample": 0.0,
    "weighted_clauses": True,
    "number_of_state_bits": 8,
}


def train_step(model: MultiClassTsetlinMachine, 
               X_train: np.ndarray, 
               Y_train: np.ndarray, 
               X_test: np.ndarray, 
               Y_test: np.ndarray, 
               epoch: int, 
               soft_labels: 
               np.ndarray | None = None, 
               temperature: float | None = None,
               alpha: float | None = None) -> tuple[float, float, float, float]:
    """
    Train a model for one epoch and return test accuracy and timing information.

    Args:
        model: The TsetlinMachine model to train
        X_train: Training data features
        Y_train: Training data labels 
        X_test: Test data features
        Y_test: Test data labels
        soft_labels: Soft labels for the training data
        temperature: Temperature for the soft labels
        alpha: Alpha for the soft labels
    Returns:
        tuple[float, float, float]: Test accuracy percentage, training time, and testing time
    """
    tqdm.write(f'Epoch {epoch:>3}:', end=' ')

    start_training = time()
    if soft_labels is None or temperature is None or alpha is None:
        model.fit(X_train, Y_train, epochs=1, incremental=True)
    else:
        model.fit_soft(X_train, Y_train, epochs=1, incremental=True, soft_labels=soft_labels, temperature=temperature, alpha=alpha)
    stop_training = time()
    tqdm.write(f'Training time: {stop_training-start_training:.2f} s', end=' ')

    # test the model on the training set - not included in timing
    train_prediction = model.predict(X_train)
    train_result = 100*(train_prediction == Y_train).mean()
    tqdm.write(f'Training accuracy: {train_result:.2f}%', end=' ')

    start_testing = time()
    prediction = model.predict(X_test)
    test_result = 100*(prediction == Y_test).mean()
    stop_testing = time()
    tqdm.write(f'Testing time: {stop_testing-start_testing:.2f} s, Test accuracy: {test_result:.2f}%')
    return train_result, test_result, stop_training-start_training, stop_testing-start_testing

def validate_params(params: dict, experiment_name: str, distillation_type: str) -> str:
    """
    Validate the parameters for the experiment.
    Example valid params:
        {
            "teacher": {
                "C": 1000,
                "T": 10,
                "s": 5,
                "epochs": 30,
            },
            "student": {
                "C": 100,
                "T": 10,
                "s": 5,
                "epochs": 60,
            },
            "temperature": 4.0,
            "alpha": 0.5,
            "z": 0.2,
            "weighted_clauses": True,
            "number_of_state_bits": 8,
        }

    Args:
        params (dict): Parameters for the experiment
        experiment_name (str): Name of the experiment

    Returns:
        str: id of the experiment
    """
    # check that the parameters are valid
    assert "teacher" in params, "teacher parameters are required"
    assert "student" in params, "student parameters are required"
    assert all(key in params["teacher"] for key in ["C", "T", "s", "epochs"]), "teacher parameters are required"
    assert all(key in params["student"] for key in ["C", "T", "s", "epochs"]), "student parameters are required"
    assert "weighted_clauses" in params, "weighted_clauses is required"
    assert "number_of_state_bits" in params, "number_of_state_bits is required"
    params["combined_epochs"] = params["teacher"]["epochs"] + params["student"]["epochs"]
    
    if distillation_type == "distribution":
        assert "temperature" in params, "temperature is required"
        assert params["temperature"] > 0, "temperature must be greater than 0"
        assert "alpha" in params, "alpha is required"
        assert params["alpha"] >= 0 and params["alpha"] <= 1, "alpha must be between 0 and 1"
        assert "z" in params, "z is required"
        assert params["z"] > 0 and params["z"] < 1, "z must be between 0 and 1"

        # generate experiment id
        exid = f"{experiment_name.replace(' ', '-')}_tC{params['teacher']['C']}_sC{params['student']['C']}_" \
            f"tT{params['teacher']['T']}_sT{params['student']['T']}_ts{params['teacher']['s']}_ss{params['student']['s']}_" \
            f"te{params['teacher']['epochs']}_se{params['student']['epochs']}_temp{params['temperature']}_a{params['alpha']}_z{params['z']}" 

    elif distillation_type == "clause":
        assert "downsample" in params, "downsample is required"
        assert params["downsample"] >= 0 and params["downsample"] <= 1, "downsample must be between 0 and 1"

        # generate experiment id
        exid = f"{experiment_name.replace(' ', '-')}_tC{params['teacher']['C']}_sC{params['student']['C']}_" \
            f"tT{params['teacher']['T']}_sT{params['student']['T']}_ts{params['teacher']['s']}_ss{params['student']['s']}_" \
            f"te{params['teacher']['epochs']}_se{params['student']['epochs']}_ds{params['downsample']}" 
    else:
        raise ValueError(f"Invalid distillation type: {distillation_type}")

    return exid

def plot_results(output: dict, fpath: str):
    # load results
    results = pd.DataFrame(output["results"])
    experiment_name = output["experiment_name"]

    # Plot configuration
    alpha = 0.7
    distilled_color = "blue"
    teacher_color = "orange"
    student_color = "green"
    line_thickness = 1

    # Set font to Times New Roman for all plots
    plt.rcParams['font.family'] = 'CMU Serif'
    plt.rcParams['font.serif'] = ['CMU Serif']
    plt.rcParams['mathtext.fontset'] = 'cm'  # For math text
    plt.rcParams['font.size'] = 14

    # plot test results and save
    analysis = output["analysis"]
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plt.axhline(analysis["avg_acc_test_distilled"], color=distilled_color, linestyle=":", alpha=0.4, label="_Distilled Avg")
    plt.axhline(analysis["avg_acc_test_teacher"], color=teacher_color, linestyle=":", alpha=0.4, label="_Teacher Avg")
    plt.axhline(analysis["avg_acc_test_student"], color=student_color, linestyle=":", alpha=0.4, label="_Student Avg")
    plt.plot(results[ACC_TEST_DISTILLED], label="Distilled", linewidth=line_thickness)
    plt.plot(results[ACC_TEST_TEACHER], label="Teacher", alpha=alpha, linewidth=line_thickness)
    plt.plot(results[ACC_TEST_STUDENT], label="Student", alpha=alpha, linewidth=line_thickness)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    if len(results) < 100:
        plt.xticks(range(0, len(results), 10))
    else:
        plt.xticks(range(0, len(results), ((len(results)//100)+1)*10))
    plt.legend(loc="lower right")
    plt.grid(linestyle='dotted')
    plt.savefig(os.path.join(fpath, experiment_name+"_"+TEST_ACCURACY_PNG_PATH))
    plt.close()
    
    # plot train results and save
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plt.axhline(analysis["avg_acc_train_distilled"], color=distilled_color, linestyle=":", alpha=0.4, label="_Distilled Avg")
    plt.axhline(analysis["avg_acc_train_teacher"], color=teacher_color, linestyle=":", alpha=0.4, label="_Teacher Avg")
    plt.axhline(analysis["avg_acc_train_student"], color=student_color, linestyle=":", alpha=0.4, label="_Student Avg")
    plt.plot(results[ACC_TRAIN_DISTILLED], label="Distilled", linewidth=line_thickness)
    plt.plot(results[ACC_TRAIN_TEACHER], label="Teacher", alpha=alpha, linewidth=line_thickness)
    plt.plot(results[ACC_TRAIN_STUDENT], label="Student", alpha=alpha, linewidth=line_thickness)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    if len(results) < 100:
        plt.xticks(range(0, len(results), 10))
    else:
        plt.xticks(range(0, len(results), ((len(results)//100)+1)*10))
    plt.legend(loc="lower right")
    plt.grid(linestyle='dotted')
    plt.savefig(os.path.join(fpath, experiment_name+"_"+TRAIN_ACCURACY_PNG_PATH))
    plt.close()

    # plot bar chart of test time for each model
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plt.grid(linestyle='dotted', zorder=0)
    labels = ["Teacher", "Student", "Distilled"]
    data = [analysis["avg_time_test_teacher"], analysis["avg_time_test_student"], analysis["avg_time_test_distilled"]]
    colors = [teacher_color, student_color, distilled_color]
    plt.bar(labels, data, color=colors, zorder=10)
    # get y tick size
    yticks = plt.yticks()[0]
    offset = yticks[0] * 0.1
    plt.yticks(np.arange(0, yticks.max()*1.1, yticks[1]-yticks[0]))
    # add text on top of each bar
    for i, label in enumerate(labels):
        plt.text(i, data[i]+offset, f"{data[i]:.3f} s", ha="center", va="bottom")
    # take default y ticks and set 10% higher
    #plt.xlabel("Model")
    plt.ylabel("Inference Time (s)")
    plt.savefig(os.path.join(fpath, experiment_name+"_"+TEST_TIME_PNG_PATH))
    plt.close()

    # plot bar chart of training time for each model
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plt.grid(linestyle='dotted', zorder=0)
    labels = ["Teacher", "Student", "Distilled"]
    data = [analysis["avg_time_train_teacher"], analysis["avg_time_train_student"], analysis["avg_time_train_distilled"]]
    colors = [teacher_color, student_color, distilled_color]
    plt.bar(labels, data, color=colors, zorder=10)
    # get y tick size
    yticks = plt.yticks()[0]
    offset = yticks[0] * 0.1
    plt.yticks(np.arange(0, yticks.max()*1.1, yticks[1]-yticks[0]))
    # add text on top of each bar
    for i, label in enumerate(labels):
        plt.text(i, data[i]+offset, f"{data[i]:.3f} s", ha="center", va="bottom")
    # take default y ticks and set 10% higher
    #plt.xlabel("Model")
    plt.ylabel("Training Time (s)")
    plt.savefig(os.path.join(fpath, experiment_name+"_"+TRAIN_TIME_PNG_PATH))
    plt.close()

def distribution_distillation_experiment(
    dataset: Dataset,
    experiment_name: str,
    params: dict = DISTRIB_DISTILLED_DEFAULTS,
    folderpath: str = DEFAULT_FOLDERPATH,
    save_all: bool = False,
    overwrite: bool = False,
    make_activation_maps: bool = True,
) -> dict:
    """
    Run a distillation experiment comparing teacher, student, and distilled models.

    Note on baseline_teacher_model and baseline_student_model:
    This is really only for the downsample experiment where we only want to change the downsampling parameters.
    This lets us use the same teacher and student models for all downsampling experiments.
    Remember, the training looks like this:
        student_model trained on original data for combined_epochs
        teacher_model trained on original data for combined_epochs, but a checkpoint is saved after teacher_epochs
        distilled_model trained on output of teacher_model (transformed and downsampled) for student_epochs

    Args:
        dataset (Dataset): The dataset to use for the experiment
        experiment_name (str): Name of the experiment
        params (dict, optional): Parameters for the experiment. Defaults to DISTILLED_DEFAULTS.
        folderpath (str, optional): Path to save experiment results. Defaults to DEFAULT_FOLDERPATH.
        save_all (bool, optional): Whether to save all models. Defaults to False. If True, saves 
            all models to the experiment directory with paths teacher_baseline.pkl, student_baseline.pkl, distilled.pkl

    Returns:
        tuple: Tuple containing:
            - Dictionary containing experiment results including:
                - Teacher, student and distilled model accuracies
                - Training and testing times
                - Number of clauses dropped during distillation
                - Total experiment time
            - pd.DataFrame: Results dataframe
    """
    exp_start = time()
    print(f"Starting distribution-based distillation experiment {experiment_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # check that the data is valid
    dataset.validate_lengths()
    X_train, Y_train, X_test, Y_test = dataset.get_data()

    # fill in missing parameters with defaults
    for key, value in DISTRIB_DISTILLED_DEFAULTS.items():
        if key not in params:
            print(f"Parameter {key} not specified, using default value {value}")
            params[key] = value

    print(f"Using params: {params}")

    # get experiment id
    experiment_id = validate_params(params, experiment_name, "distribution")
    print(f"Experiment ID: {experiment_id}")

    params = Prodict.from_dict(params)
    # create an experiment directory
    if not overwrite and os.path.exists(os.path.join(folderpath, experiment_id)):
        # Check if experiment files exist, and if save_all, check model files exist too
        basic_files_exist = all(os.path.exists(os.path.join(folderpath, experiment_id, f)) 
                              for f in [OUTPUT_JSON_PATH, RESULTS_CSV_PATH])
        
        model_files_exist = not save_all or all(os.path.exists(os.path.join(folderpath, experiment_id, f)) 
                                               for f in [TEACHER_BASELINE_MODEL_PATH, 
                                                       STUDENT_BASELINE_MODEL_PATH,
                                                       DISTILLED_MODEL_PATH])
        
        if basic_files_exist and model_files_exist:
            print(f"Experiment {experiment_id} already exists, replotting and skipping")
            # load the results
            results = pd.read_csv(os.path.join(folderpath, experiment_id, RESULTS_CSV_PATH))
            output = load_json(os.path.join(folderpath, experiment_id, OUTPUT_JSON_PATH))
            # plot results
            plot_results(output, os.path.join(folderpath, experiment_id))
            return output, results
        else:
            print(f"Experiment {experiment_id} already exists, but some files are missing, continuing") 

    make_dir(os.path.join(folderpath, experiment_id), overwrite=True)
    teacher_model_path = os.path.join(folderpath, experiment_id, TEACHER_CHECKPOINT_PATH)

    # create models
    baseline_student_tm = MultiClassTsetlinMachine(
        params.student.C, params.student.T, params.student.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)
    baseline_teacher_tm = MultiClassTsetlinMachine(
        params.teacher.C, params.teacher.T, params.teacher.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)
    distilled_tm = MultiClassTsetlinMachine(
        params.student.C, params.student.T, params.student.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)

    # create a results dataframe
    results = pd.DataFrame(columns=RESULTS_COLUMNS, index=range(params.combined_epochs))
    
    # train baselines
    # train baseline teacher
    print(f"Creating a baseline teacher with {params.teacher.C} clauses and training on original data")
    start = time()
    bt_pbar = tqdm(range(params.combined_epochs), desc="Teacher", leave=False, dynamic_ncols=True)
    best_acc = 0
    best_acc_epoch = 0
    for i in bt_pbar:
        train_result, test_result, train_time, test_time = train_step(baseline_teacher_tm, X_train, Y_train, X_test, Y_test, i)
        results.loc[i, ACC_TRAIN_TEACHER], results.loc[i, TIME_TRAIN_TEACHER] = train_result, train_time
        results.loc[i, ACC_TEST_TEACHER], results.loc[i, TIME_TEST_TEACHER] = test_result, test_time
        bt_pbar.set_description(f"Teacher: {results[ACC_TEST_TEACHER].mean():.2f} +/- {results[ACC_TEST_TEACHER].std():.2f}%")

        if i <= params.teacher.epochs - 1 and test_result > best_acc:
            save_pkl(baseline_teacher_tm, teacher_model_path)
            best_acc = test_result
            best_acc_epoch = i

        if i == params.teacher.epochs - 1:
            tqdm.write(f"Saved teacher model to {teacher_model_path} @ epoch {best_acc_epoch} (best acc: {best_acc:.2f}%)")

    bt_pbar.close()
    end = time()
    print(f'Baseline teacher training time: {end-start:.2f} s')

    # copy first teacher_epochs results to distilled results
    results.loc[:params.teacher.epochs, ACC_TEST_DISTILLED] = results.loc[:params.teacher.epochs, ACC_TEST_TEACHER]
    results.loc[:params.teacher.epochs, ACC_TRAIN_DISTILLED] = results.loc[:params.teacher.epochs, ACC_TRAIN_TEACHER]
    results.loc[:params.teacher.epochs, TIME_TRAIN_DISTILLED] = results.loc[:params.teacher.epochs, TIME_TRAIN_TEACHER]
    results.loc[:params.teacher.epochs, TIME_TEST_DISTILLED] = results.loc[:params.teacher.epochs, TIME_TEST_TEACHER]

    # train baseline student
    print(f"Creating a baseline student with {params.student.C} clauses and training on original data")
    start = time()
    bs_pbar = tqdm(range(params.combined_epochs), desc="Student", leave=False, dynamic_ncols=True)
    for i in bs_pbar:
        train_result, test_result, train_time, test_time = train_step(baseline_student_tm, X_train, Y_train, X_test, Y_test, i)
        results.loc[i, ACC_TRAIN_STUDENT], results.loc[i, TIME_TRAIN_STUDENT] = train_result, train_time
        results.loc[i, ACC_TEST_STUDENT], results.loc[i, TIME_TEST_STUDENT] = test_result, test_time
        bs_pbar.set_description(f"Student: {results[ACC_TEST_STUDENT].mean():.2f} +/- {results[ACC_TEST_STUDENT].std():.2f}%")

    bs_pbar.close()
    end = time()
    print(f'Baseline student training time: {end-start:.2f} s')

    print(f"Loading teacher model from {teacher_model_path}, trained for {params.teacher.epochs} epochs")
    teacher_tm = load_pkl(teacher_model_path)
    if not save_all:
        rm_file(teacher_model_path) # remove the teacher model file. we don't need it anymore

    # GET soft labels
    print(f"Initializing student with {params.student.C} clauses from teacher and z={params.z}")
    distilled_tm.init_from_teacher(teacher_tm, X_train, Y_train, clauses_per_class=params.student.C, z=params.z)
    print(f"Generating soft labels from teacher")
    soft_labels = teacher_tm.get_soft_labels(X_train)
    print(f"Soft labels generated")

    start = time()
    print(f"Training distilled model for {params.student.epochs} epochs")
    dt_pbar = tqdm(range(params.teacher.epochs, params.combined_epochs), desc="Distilled", leave=False, dynamic_ncols=True)
    for i in dt_pbar:
        train_result, test_result, train_time, test_time = train_step(distilled_tm, X_train, Y_train, X_test, Y_test, i, soft_labels, params.temperature, params.alpha)
        results.loc[i, ACC_TRAIN_DISTILLED], results.loc[i, TIME_TRAIN_DISTILLED] = train_result, train_time
        results.loc[i, ACC_TEST_DISTILLED], results.loc[i, TIME_TEST_DISTILLED] = test_result, test_time
        dt_pbar.set_description(f"Distilled: {results[ACC_TEST_DISTILLED].mean():.2f} +/- {results[ACC_TEST_DISTILLED].std():.2f}%")

    dt_pbar.close()
    end = time()

    print(f'Teacher-student training time: {end-start:.2f} s')

    if save_all:
        save_pkl(baseline_teacher_tm, os.path.join(folderpath, experiment_id, TEACHER_BASELINE_MODEL_PATH))
        save_pkl(baseline_student_tm, os.path.join(folderpath, experiment_id, STUDENT_BASELINE_MODEL_PATH))
        save_pkl(distilled_tm, os.path.join(folderpath, experiment_id, DISTILLED_MODEL_PATH))

    total_time = time() - exp_start

    # THIS IS DONE BECAUSE the teacher model will skew inference time when it doesn't actually affect reality
    post_teacher_results = results.iloc[params.teacher.epochs:]

    output = {
        "analysis": {
            # average accuracy on the test set
            "avg_acc_test_teacher": results[ACC_TEST_TEACHER].mean(), 
            "avg_acc_test_student": results[ACC_TEST_STUDENT].mean(),
            "avg_acc_test_distilled": post_teacher_results[ACC_TEST_DISTILLED].mean(),

            # standard deviation of accuracy on the test set
            "std_acc_test_teacher": results[ACC_TEST_TEACHER].std(),
            "std_acc_test_student": results[ACC_TEST_STUDENT].std(),
            "std_acc_test_distilled": post_teacher_results[ACC_TEST_DISTILLED].std(),

            # average accuracy on the training set
            "avg_acc_train_teacher": results[ACC_TRAIN_TEACHER].mean(),
            "avg_acc_train_student": results[ACC_TRAIN_STUDENT].mean(),
            "avg_acc_train_distilled": post_teacher_results[ACC_TRAIN_DISTILLED].mean(),\

            # standard deviation of accuracy on the training set
            "std_acc_train_teacher": results[ACC_TRAIN_TEACHER].std(),
            "std_acc_train_student": results[ACC_TRAIN_STUDENT].std(),
            "std_acc_train_distilled": post_teacher_results[ACC_TRAIN_DISTILLED].std(),

            # final accuracy on the test set
            "final_acc_test_distilled": results[ACC_TEST_DISTILLED].iloc[-1],
            "final_acc_test_teacher": results[ACC_TEST_TEACHER].iloc[-1],
            "final_acc_test_student": results[ACC_TEST_STUDENT].iloc[-1],

            # final accuracy on the training set
            "final_acc_train_distilled": results[ACC_TRAIN_DISTILLED].iloc[-1],
            "final_acc_train_teacher": results[ACC_TRAIN_TEACHER].iloc[-1],
            "final_acc_train_student": results[ACC_TRAIN_STUDENT].iloc[-1],

            # sum of all training epoch times
            "sum_time_train_teacher": results[TIME_TRAIN_TEACHER].sum(),
            "sum_time_train_student": results[TIME_TRAIN_STUDENT].sum(),
            "sum_time_train_distilled": results[TIME_TRAIN_DISTILLED].sum(),

            # sum of all test set evaluation times
            "sum_time_test_teacher": results[TIME_TEST_TEACHER].sum(),
            "sum_time_test_student": results[TIME_TEST_STUDENT].sum(),
            "sum_time_test_distilled": results[TIME_TEST_DISTILLED].sum(),

            # average time for each training epoch
            "avg_time_train_teacher": results[TIME_TRAIN_TEACHER].mean(),
            "avg_time_train_student": results[TIME_TRAIN_STUDENT].mean(),
            "avg_time_train_distilled": post_teacher_results[TIME_TRAIN_DISTILLED].mean(),

            # average time for each test set evaluation
            "avg_time_test_teacher": results[TIME_TEST_TEACHER].mean(),
            "avg_time_test_student": results[TIME_TEST_STUDENT].mean(),
            "avg_time_test_distilled": post_teacher_results[TIME_TEST_DISTILLED].mean(),

            # inference time for each epoch
            "inference_time_teacher": post_teacher_results[TIME_TEST_TEACHER].mean(),
            "inference_time_student": post_teacher_results[TIME_TEST_STUDENT].mean(),
            "inference_time_distilled": post_teacher_results[TIME_TEST_DISTILLED].mean(),

            "total_time": total_time,
        },
        "data": {
            "X_train": X_train.shape,
            "Y_train": Y_train.shape,
            "X_test": X_test.shape,
            "Y_test": Y_test.shape,
            "num_classes": len(np.unique(Y_train)),
        },
        "params": params.to_dict(),
        "experiment_name": experiment_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id": experiment_id,
        "results": results.to_dict(),
    }

    # save output
    fpath = os.path.join(folderpath, experiment_id)
    save_json(output, os.path.join(fpath, OUTPUT_JSON_PATH))
    results.to_csv(os.path.join(fpath, RESULTS_CSV_PATH))

    # make activation maps
    if make_activation_maps:
        try:
            # plot activation maps
            samples = np.random.randint(0, len(X_test), size=4)
            visualize_activation_maps(baseline_teacher_tm, baseline_student_tm, distilled_tm, 
                                    X_test[samples], Y_test[samples], dataset.image_shape, os.path.join(fpath, 
                                    experiment_name+"_"+ACTIVATION_MAPS_PNG_PATH))
        except Exception as e:
            print(f"Error making activation maps: {e}")
            print("Make sure the dataset has a valid image shape")

    # plot results
    plot_results(output, fpath)

    return output, results

def get_downsample_indices(X:np.ndarray, downsample: float, symmetric: bool = True) -> np.ndarray:
    """
    Downsample clauses by removing those that are too active or too inactive.

    This function performs clause pruning by removing clauses that activate too frequently or too rarely.
    Clauses that are too active (above over threshold) are considered too specific and not generalizable.
    Clauses that are too inactive (below under threshold) are considered too general and not specific enough.

    This function returns the *indices* of where to drop - it does not actually drop the clauses

    Args:
        X (np.ndarray): Data transformed by teacher TM's clauses
        downsample (float): Drop clauses that are activated in (1 - downsample)*100% of the time. 
            If downsample is 0.05, then any clause that is activated in 95% of the time is dropped.
        symmetric (bool): If True, ALSO drop clauses that are inactive in (1 - downsample)*100% of the time.
            This doesn't usually make a difference to the distilled model's performance.

    Returns:
        np.ndarray: Indices of clauses to drop
    """
    # drop clauses that are too active or too inactive
    # this is a form of data mining where we seek out the clauses that are most informative
    # we drop the clauses that are too active because they are too specific and not generalizable
    # this should reduce the number of clauses and make the student learn faster
    # this works pretty well with downsample = 0.05
    assert downsample >= 0 and downsample < 1, "Downsample should be a float between 0 and 1"
    sums = np.sum(X, axis=0) # shape is (num_classes*num_clauses)
    normalized_sums = sums / X.shape[0] # get the sum of each clause over all samples divided by the number of samples

    # find where to drop
    over_clauses = np.where(normalized_sums > (1 - downsample))[0] # clauses that are activated in (1 - downsample)*100% of the time
    under_clauses = np.where(normalized_sums < downsample)[0] # clauses that are inactive (1 - downsample)*100% of the time
    if symmetric:
        clauses_to_drop = np.concatenate([over_clauses, under_clauses])
    else:
        clauses_to_drop = over_clauses

    return clauses_to_drop

def downsample_clauses(X_train_transformed:np.ndarray, X_test_transformed:np.ndarray, downsample: float, symmetric: bool = True) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Downsample clauses by removing those that are too active or too inactive.

    This function performs clause pruning by removing clauses that activate too frequently or too rarely.
    Clauses that are too active (above over threshold) are considered too specific and not generalizable.
    Clauses that are too inactive (below under threshold) are considered too general and not specific enough.

    Args:
        X_train_transformed (np.ndarray): Training data transformed by teacher TM's clauses
        X_test_transformed (np.ndarray): Test data transformed by teacher TM's clauses  
        downsample (float): Drop clauses that are activated in (1 - downsample)*100% of the time. 
            If downsample is 0.05, then any clause that is activated in 95% of the time is dropped.
        symmetric (bool): If True, ALSO drop clauses that are inactive in (1 - downsample)*100% of the time.
            This doesn't usually make a difference to the distilled model's performance.

    Returns:
        tuple: Contains:
            - X_train_reduced (np.ndarray): Training data with pruned clauses removed
            - X_test_reduced (np.ndarray): Test data with pruned clauses removed
            - num_clauses_dropped (int): Number of clauses that were pruned
    """
    # drop clauses that are too active or too inactive
    # this is a form of data mining where we seek out the clauses that are most informative
    # we drop the clauses that are too active because they are too specific and not generalizable
    # this should reduce the number of clauses and make the student learn faster
    # this works pretty well with downsample = 0.05
    clauses_to_drop = get_downsample_indices(X_train_transformed, downsample, symmetric=symmetric)

    X_train_reduced = np.delete(X_train_transformed, clauses_to_drop, axis=1) # delete the clauses from the training data
    X_test_reduced = np.delete(X_test_transformed, clauses_to_drop, axis=1) # delete the clauses from the testing data

    num_clauses_dropped = len(clauses_to_drop)
    reduction_percentage = 100*(num_clauses_dropped/X_train_transformed.shape[1]) # calculate the percentage of clauses dropped

    print(f"Dropped {num_clauses_dropped} clauses from {X_train_transformed.shape[1]} clauses, {reduction_percentage:.2f}% reduction")
    
    return X_train_reduced, X_test_reduced, num_clauses_dropped


def clause_distillation_experiment(
    dataset: Dataset,
    experiment_name: str,
    params: dict = CLAUSE_DISTILLED_DEFAULTS,
    folderpath: str = DEFAULT_FOLDERPATH,
    save_all: bool = False,
    overwrite: bool = False,
    make_activation_maps: bool = True,
) -> dict:
    """
    Run a distillation experiment comparing teacher, student, and distilled models.

    Note on baseline_teacher_model and baseline_student_model:
    This is really only for the downsample experiment where we only want to change the downsampling parameters.
    This lets us use the same teacher and student models for all downsampling experiments.
    Remember, the training looks like this:
        student_model trained on original data for combined_epochs
        teacher_model trained on original data for combined_epochs, but a checkpoint is saved after teacher_epochs
        distilled_model trained on output of teacher_model (transformed and downsampled) for student_epochs

    Args:
        dataset (Dataset): The dataset to use for the experiment
        experiment_name (str): Name of the experiment
        params (dict, optional): Parameters for the experiment. Defaults to DISTILLED_DEFAULTS.
        folderpath (str, optional): Path to save experiment results. Defaults to DEFAULT_FOLDERPATH.
        save_all (bool, optional): Whether to save all models. Defaults to False. If True, saves 
            all models to the experiment directory with paths teacher_baseline.pkl, student_baseline.pkl, distilled.pkl

    Returns:
        tuple: Tuple containing:
            - Dictionary containing experiment results including:
                - Teacher, student and distilled model accuracies
                - Training and testing times
                - Number of clauses dropped during distillation
                - Total experiment time
            - pd.DataFrame: Results dataframe
    """
    exp_start = time()
    print(f"Starting clause-based distillation experiment {experiment_name} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # check that the data is valid
    dataset.validate_lengths()
    X_train, Y_train, X_test, Y_test = dataset.get_data()

    # fill in missing parameters with defaults
    for key, value in CLAUSE_DISTILLED_DEFAULTS.items():
        if key not in params:
            print(f"Parameter {key} not specified, using default value {value}")
            params[key] = value

    print(f"Using params: {params}")

    # get experiment id
    experiment_id = validate_params(params, experiment_name, "clause")
    print(f"Experiment ID: {experiment_id}")

    params = Prodict.from_dict(params)
    # create an experiment directory
    if not overwrite and os.path.exists(os.path.join(folderpath, experiment_id)):
        # Check if experiment files exist, and if save_all, check model files exist too
        basic_files_exist = all(os.path.exists(os.path.join(folderpath, experiment_id, f)) 
                              for f in [OUTPUT_JSON_PATH, RESULTS_CSV_PATH])
        
        model_files_exist = not save_all or all(os.path.exists(os.path.join(folderpath, experiment_id, f)) 
                                               for f in [TEACHER_BASELINE_MODEL_PATH, 
                                                       STUDENT_BASELINE_MODEL_PATH,
                                                       DISTILLED_MODEL_PATH])
        
        if basic_files_exist and model_files_exist:
            print(f"Experiment {experiment_id} already exists, replotting and skipping")
            # load the results
            results = pd.read_csv(os.path.join(folderpath, experiment_id, RESULTS_CSV_PATH))
            output = load_json(os.path.join(folderpath, experiment_id, OUTPUT_JSON_PATH))
            # plot results
            plot_results(output, os.path.join(folderpath, experiment_id))
            return output, results
        else:
            print(f"Experiment {experiment_id} already exists, but some files are missing, continuing") 

    make_dir(os.path.join(folderpath, experiment_id), overwrite=True)
    teacher_model_path = os.path.join(folderpath, experiment_id, TEACHER_CHECKPOINT_PATH)

    # create models
    baseline_student_tm = MultiClassTsetlinMachine(
        params.student.C, params.student.T, params.student.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)
    baseline_teacher_tm = MultiClassTsetlinMachine(
        params.teacher.C, params.teacher.T, params.teacher.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)
    distilled_tm = MultiClassTsetlinMachine(
        params.student.C, params.student.T, params.student.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)

    # create a results dataframe
    results = pd.DataFrame(columns=RESULTS_COLUMNS, index=range(params.combined_epochs))
    
    # train baselines
    # train baseline teacher
    print(f"Creating a baseline teacher with {params.teacher.C} clauses and training on original data")
    start = time()
    bt_pbar = tqdm(range(params.combined_epochs), desc="Teacher", leave=False, dynamic_ncols=True)
    best_acc = 0
    best_acc_epoch = 0
    for i in bt_pbar:
        train_result, test_result, train_time, test_time = train_step(baseline_teacher_tm, X_train, Y_train, X_test, Y_test, i)
        results.loc[i, ACC_TRAIN_TEACHER], results.loc[i, TIME_TRAIN_TEACHER] = train_result, train_time
        results.loc[i, ACC_TEST_TEACHER], results.loc[i, TIME_TEST_TEACHER] = test_result, test_time
        bt_pbar.set_description(f"Teacher: {results[ACC_TEST_TEACHER].mean():.2f} +/- {results[ACC_TEST_TEACHER].std():.2f}%")

        if i <= params.teacher.epochs - 1 and test_result > best_acc:
            save_pkl(baseline_teacher_tm, teacher_model_path)
            best_acc = test_result
            best_acc_epoch = i

        if i == params.teacher.epochs - 1:
            tqdm.write(f"Saved teacher model to {teacher_model_path} @ epoch {best_acc_epoch} (best acc: {best_acc:.2f}%)")

    bt_pbar.close()
    end = time()
    print(f'Baseline teacher training time: {end-start:.2f} s')

    # copy first teacher_epochs results to distilled results
    results.loc[:params.teacher.epochs, ACC_TEST_DISTILLED] = results.loc[:params.teacher.epochs, ACC_TEST_TEACHER]
    results.loc[:params.teacher.epochs, ACC_TRAIN_DISTILLED] = results.loc[:params.teacher.epochs, ACC_TRAIN_TEACHER]
    results.loc[:params.teacher.epochs, TIME_TRAIN_DISTILLED] = results.loc[:params.teacher.epochs, TIME_TRAIN_TEACHER]
    results.loc[:params.teacher.epochs, TIME_TEST_DISTILLED] = results.loc[:params.teacher.epochs, TIME_TEST_TEACHER]

    # train baseline student
    print(f"Creating a baseline student with {params.student.C} clauses and training on original data")
    start = time()
    bs_pbar = tqdm(range(params.combined_epochs), desc="Student", leave=False, dynamic_ncols=True)
    for i in bs_pbar:
        train_result, test_result, train_time, test_time = train_step(baseline_student_tm, X_train, Y_train, X_test, Y_test, i)
        results.loc[i, ACC_TRAIN_STUDENT], results.loc[i, TIME_TRAIN_STUDENT] = train_result, train_time
        results.loc[i, ACC_TEST_STUDENT], results.loc[i, TIME_TEST_STUDENT] = test_result, test_time
        bs_pbar.set_description(f"Student: {results[ACC_TEST_STUDENT].mean():.2f} +/- {results[ACC_TEST_STUDENT].std():.2f}%")

    bs_pbar.close()
    end = time()
    print(f'Baseline student training time: {end-start:.2f} s')

    print(f"Loading teacher model from {teacher_model_path}, trained for {params.teacher.epochs} epochs")
    teacher_tm = load_pkl(teacher_model_path)
    if not save_all:
        rm_file(teacher_model_path) # remove the teacher model file. we don't need it anymore

    # Transform the training and testing data
    print(f"Getting offline clause outputs from teacher model")
    s = time()
    X_train_transformed = teacher_tm.transform(X_train)
    e = time()
    train_transform_time = e-s
    print(f"Got clause outputs for X_train in {train_transform_time:.2f}s")
    s = time()
    X_test_transformed = teacher_tm.transform(X_test)
    e = time()
    test_transform_time = e-s
    print(f"Got clause outputs for X_test in {test_transform_time:.2f}s")

    # downsample clauses
    if params.downsample > 0:
        print(f"Downsampling clauses with downsample rate {params.downsample}")
        X_train_downsampled, X_test_downsampled, num_clauses_dropped = downsample_clauses(X_train_transformed, X_test_transformed, params.downsample, symmetric=True)
        reduction_percentage = 100*(num_clauses_dropped/X_train_transformed.shape[1]) # calculate the percentage of clauses dropped
        if num_clauses_dropped == X_train_transformed.shape[1]:
            print(f"Every clause was dropped, skipping distillation")
            return None, results
    else:
        X_train_downsampled = X_train_transformed
        X_test_downsampled = X_test_transformed

    start = time()
    print(f"Training distilled model for {params.student.epochs} epochs")
    dt_pbar = tqdm(range(params.teacher.epochs, params.combined_epochs), desc="Distilled", leave=False, dynamic_ncols=True)
    for i in dt_pbar:
        train_result, test_result, train_time, test_time = train_step(distilled_tm, X_train_downsampled, Y_train, X_test_downsampled, Y_test, i)
        results.loc[i, ACC_TRAIN_DISTILLED], results.loc[i, TIME_TRAIN_DISTILLED] = train_result, train_time
        results.loc[i, ACC_TEST_DISTILLED], results.loc[i, TIME_TEST_DISTILLED] = test_result, test_time
        dt_pbar.set_description(f"Distilled: {results[ACC_TEST_DISTILLED].mean():.2f} +/- {results[ACC_TEST_DISTILLED].std():.2f}%")

    dt_pbar.close()
    end = time()

    print(f'Teacher-student training time: {end-start:.2f} s')

    if save_all:
        save_pkl(baseline_teacher_tm, os.path.join(folderpath, experiment_id, TEACHER_BASELINE_MODEL_PATH))
        save_pkl(baseline_student_tm, os.path.join(folderpath, experiment_id, STUDENT_BASELINE_MODEL_PATH))
        save_pkl(distilled_tm, os.path.join(folderpath, experiment_id, DISTILLED_MODEL_PATH))

    total_time = time() - exp_start

    # THIS IS DONE BECAUSE the teacher model will skew inference time when it doesn't actually affect reality
    post_teacher_results = results.iloc[params.teacher.epochs:]

    output = {
        "analysis": {
            # average accuracy on the test set
            "avg_acc_test_teacher": results[ACC_TEST_TEACHER].mean(), 
            "avg_acc_test_student": results[ACC_TEST_STUDENT].mean(),
            "avg_acc_test_distilled": post_teacher_results[ACC_TEST_DISTILLED].mean(),

            # standard deviation of accuracy on the test set
            "std_acc_test_teacher": results[ACC_TEST_TEACHER].std(),
            "std_acc_test_student": results[ACC_TEST_STUDENT].std(),
            "std_acc_test_distilled": post_teacher_results[ACC_TEST_DISTILLED].std(),

            # average accuracy on the training set
            "avg_acc_train_teacher": results[ACC_TRAIN_TEACHER].mean(),
            "avg_acc_train_student": results[ACC_TRAIN_STUDENT].mean(),
            "avg_acc_train_distilled": post_teacher_results[ACC_TRAIN_DISTILLED].mean(),\

            # standard deviation of accuracy on the training set
            "std_acc_train_teacher": results[ACC_TRAIN_TEACHER].std(),
            "std_acc_train_student": results[ACC_TRAIN_STUDENT].std(),
            "std_acc_train_distilled": post_teacher_results[ACC_TRAIN_DISTILLED].std(),

            # final accuracy on the test set
            "final_acc_test_distilled": results[ACC_TEST_DISTILLED].iloc[-1],
            "final_acc_test_teacher": results[ACC_TEST_TEACHER].iloc[-1],
            "final_acc_test_student": results[ACC_TEST_STUDENT].iloc[-1],

            # final accuracy on the training set
            "final_acc_train_distilled": results[ACC_TRAIN_DISTILLED].iloc[-1],
            "final_acc_train_teacher": results[ACC_TRAIN_TEACHER].iloc[-1],
            "final_acc_train_student": results[ACC_TRAIN_STUDENT].iloc[-1],

            # sum of all training epoch times
            "sum_time_train_teacher": results[TIME_TRAIN_TEACHER].sum(),
            "sum_time_train_student": results[TIME_TRAIN_STUDENT].sum(),
            "sum_time_train_distilled": results[TIME_TRAIN_DISTILLED].sum(),

            # sum of all test set evaluation times
            "sum_time_test_teacher": results[TIME_TEST_TEACHER].sum(),
            "sum_time_test_student": results[TIME_TEST_STUDENT].sum(),
            "sum_time_test_distilled": results[TIME_TEST_DISTILLED].sum(),

            # average time for each training epoch
            "avg_time_train_teacher": results[TIME_TRAIN_TEACHER].mean(),
            "avg_time_train_student": results[TIME_TRAIN_STUDENT].mean(),
            "avg_time_train_distilled": post_teacher_results[TIME_TRAIN_DISTILLED].mean(),

            # average time for each test set evaluation
            "avg_time_test_teacher": results[TIME_TEST_TEACHER].mean(),
            "avg_time_test_student": results[TIME_TEST_STUDENT].mean(),
            "avg_time_test_distilled": post_teacher_results[TIME_TEST_DISTILLED].mean(),

            # inference time for each epoch
            "inference_time_teacher": post_teacher_results[TIME_TEST_TEACHER].mean(),
            "inference_time_student": post_teacher_results[TIME_TEST_STUDENT].mean(),
            "inference_time_distilled": post_teacher_results[TIME_TEST_DISTILLED].mean(),

            "total_time": total_time,
        },
        "data": {
            "X_train": X_train.shape,
            "Y_train": Y_train.shape,
            "X_test": X_test.shape,
            "Y_test": Y_test.shape,
            "num_classes": len(np.unique(Y_train)),
        },
        "downsample_info": {
            "num_clauses_dropped": num_clauses_dropped,
            "L_D": X_train_downsampled.shape[1],
            "reduction_percentage": reduction_percentage,
        },
        "params": params.to_dict(),
        "experiment_name": experiment_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "id": experiment_id,
        "results": results.to_dict(),
    }

    # save output
    fpath = os.path.join(folderpath, experiment_id)
    save_json(output, os.path.join(fpath, OUTPUT_JSON_PATH))
    results.to_csv(os.path.join(fpath, RESULTS_CSV_PATH))

    # plot results
    plot_results(output, fpath)

    return output, results