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
    "_agg_num": 0,
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
    
    if params["_agg_num"] > 0:
        exid += f"_n{params['_agg_num']}"

    return exid

def plot_results(output: dict, fpath: str, downsample: float | None = None):
    """
    Plot the results of a distillation experiment.
    
    This function generates and saves plots showing the accuracy of teacher, baseline, and student models
    on both test and training datasets. It also plots inference times for comparison.
    
    Args:
        output (dict): Dictionary containing experiment results, including metrics and analysis.
        fpath (str): File path where the plots should be saved.
        downsample (float | None, optional): Downsampling rate used in clause-based distillation.
            If provided, additional plots for the downsampled model will be generated. Defaults to None.
    
    Returns:
        None: The function saves plots to the specified file path but does not return any values.
    """
    # load results
    results = pd.DataFrame(output["results"])
    experiment_name = output["experiment_name"]
    analysis = output["analysis"]

    # Plot configuration
    alpha = 0.7
    avg_alpha = 0.7
    minor_alpha = 0.2
    colors = {
        "student": "tab:blue",
        "teacher": "tab:orange",
        "baseline": "tab:green",
        "student_ds": "tab:purple"
    }
    display_names = {
        "teacher": "Teacher",
        "baseline": "Baseline",
        "student": "Student",
        "student_ds": "Student DS",
    }
    line_thickness = 1.2
    default_font_size = 14
    legend_font_size = 13
    font_family = "CMU Serif"

    # Set font to Times New Roman for all plots
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.serif'] = font_family
    plt.rcParams['mathtext.fontset'] = 'cm'  # For math text
    plt.rcParams['font.size'] = default_font_size

    def get_label_position(x, y, xlim, ylim, offset=15):
        """Calculate optimal label position to avoid plot overflow."""
        x_min, x_max = xlim
        y_min, y_max = ylim
        x_range = x_max - x_min
        x_rel = (x - x_min) / x_range
        
        # More conservative positioning that accounts for boundaries
        if x_rel < 0.3:  # Point is on the left side
            x_offset = offset
            y_offset = -offset/2  # Slightly above
            ha = 'left'
        elif x_rel > 0.7:  # Point is on the right side
            x_offset = -offset
            y_offset = offset/2  # Slightly below
            ha = 'right'
        else:  # Point is in the middle
            # Choose the side with more space
            if x - x_min > x_max - x:
                x_offset = -offset
                y_offset = -offset/2  # Slightly below
                ha = 'right'
            else:
                x_offset = offset
                y_offset = offset/2  # Slightly above
                ha = 'left'
        
        va = 'center'
            
        return x_offset, y_offset, ha, va

    def setup_grid():
        """Setup grid with major and minor lines."""
        plt.grid(linestyle='dotted', which='major')
        plt.grid(linestyle='dotted', which='minor', alpha=minor_alpha)
        plt.minorticks_on()

    def plot_accuracy_curves(metric_type: str):
        """Plot accuracy curves for a given metric type (train/test)."""
        plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)

        # Plot average lines
        for model in ["student", "teacher", "baseline"]:
            if downsample is not None and model == "student":
                plt.axhline(analysis[f"avg_acc_{metric_type}_student_ds"],
                          color=colors["student_ds"], linestyle=":",
                          alpha=avg_alpha, label="_Student DS Avg")
            plt.axhline(analysis[f"avg_acc_{metric_type}_{model}"],
                      color=colors[model], linestyle=":",
                      alpha=avg_alpha, label=f"_{display_names[model]} Avg")

        # Plot curves
        for model in ["student", "teacher", "baseline"]:
            if downsample is not None and model == "student":
                plt.plot(results[f"acc_{metric_type}_student_ds"],
                        label="Student DS", color=colors["student_ds"],
                        linewidth=line_thickness)
            plt.plot(results[f"acc_{metric_type}_{model}"],
                    label=display_names[model], color=colors[model],
                    alpha=alpha if model != "student" else 1.0,
                    linewidth=line_thickness)
        
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        if len(results) < 100:
            plt.xticks(range(0, len(results), 10))
        else:
            plt.xticks(range(0, len(results), ((len(results)//100)+1)*10))
        plt.legend(loc="lower right", fontsize=legend_font_size)
        setup_grid()
        plt.savefig(os.path.join(fpath, experiment_name+f"_{metric_type}_accuracy.png"))
        plt.close()

    def plot_time_bars(metric_type: str):
        """Plot time bars for a given metric type (train/test)."""
        plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
        plt.grid(linestyle='dotted', zorder=0, axis="y")

        labels = ["Teacher", "Baseline", "Student"]
        data = [analysis[f"avg_time_{metric_type}_teacher"], analysis[f"avg_time_{metric_type}_baseline"], analysis[f"avg_time_{metric_type}_student"]]
        bar_colors = [colors["teacher"], colors["baseline"], colors["student"]]

        if downsample is not None:
            data.append(analysis[f"avg_time_{metric_type}_student_ds"])
            labels.append("Student DS")
            bar_colors.append(colors["student_ds"])

        plt.bar(labels, data, color=bar_colors, zorder=10)
        yticks = plt.yticks()[0]
        offset = yticks[0] * 0.1
        plt.yticks(np.arange(0, yticks.max()*1.1, yticks[1]-yticks[0]))
        
        for i, (label, value) in enumerate(zip(labels, data)):
            plt.text(i, value+offset, f"{value:.3f} s", ha="center", va="bottom")
        
        plt.ylabel(f"{metric_type.capitalize()} Time (s)")
        plt.savefig(os.path.join(fpath, experiment_name+f"_{metric_type}_time.png"))
        plt.close()

    def plot_efficiency(metric_type: str, polygon_alpha: float = 0.08):
        """Plot efficiency curves for a given metric type (train/test)."""
        plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
        setup_grid()
        
        # Create data points for each model
        models = [
            ("Teacher", analysis[f"avg_acc_{metric_type}_teacher"],
             analysis[f"avg_time_{metric_type}_teacher"], colors["teacher"]),
            ("Baseline", analysis[f"avg_acc_{metric_type}_baseline"],
             analysis[f"avg_time_{metric_type}_baseline"], colors["baseline"]),
            ("Student", analysis[f"avg_acc_{metric_type}_student"],
             analysis[f"avg_time_{metric_type}_student"], colors["student"])
        ]

        if downsample is not None:
            models.append(("Student DS", analysis[f"avg_acc_{metric_type}_student_ds"],
                          analysis[f"avg_time_{metric_type}_student_ds"], colors["student_ds"]))
        
        # Plot points first
        for name, acc, time, color in models:
            plt.scatter(time, acc, color=color, s=100, zorder=10)
        
        # Extract teacher and student data for the line
        teacher_data = next(m for m in models if m[0] == "Teacher")
        student_data = next(m for m in models if m[0] == "Baseline")
        
        teacher_time, teacher_acc = teacher_data[2], teacher_data[1]
        student_time, student_acc = student_data[2], student_data[1]
        
        # Get the current axis limits
        xlim = plt.xlim()
        ylim = plt.ylim()
        
        # Add some padding to the plot limits
        x_padding = (xlim[1] - xlim[0]) * 0.1
        y_padding = (ylim[1] - ylim[0]) * 0.1
        xlim = (xlim[0] - x_padding, xlim[1] + x_padding)
        ylim = (ylim[0] - y_padding, ylim[1] + y_padding)
        plt.xlim(xlim)
        plt.ylim(ylim)

        # Calculate the slope and intercept of the line connecting teacher and student
        if teacher_time != student_time:  # Avoid division by zero
            slope = (teacher_acc - student_acc) / (teacher_time - student_time)
            intercept = teacher_acc - slope * teacher_time
            
            # Create points for the line that covers the entire plot
            x_line = np.array([xlim[0], xlim[1]])
            y_line = slope * x_line + intercept
            
            # Plot the line connecting teacher and student
            plt.plot(x_line, y_line, '--', color='black', linewidth=1.5, zorder=5)
            
            # Create polygons for shading the regions
            if slope > 0:
                # Line goes from bottom-left to top-right
                # Positive gains region (left of the line)
                left_polygon = plt.Polygon([
                    [xlim[0], ylim[0]],  # Bottom-left corner
                    [xlim[0], ylim[1]],  # Top-left corner
                    [xlim[1], slope * xlim[1] + intercept],  # Right intersection
                    [xlim[0], slope * xlim[0] + intercept],  # Left intersection
                ], color='green', alpha=polygon_alpha, zorder=1)
                
                # Negative gains region (right of the line)
                right_polygon = plt.Polygon([
                    [xlim[0], slope * xlim[0] + intercept],  # Left intersection
                    [xlim[1], slope * xlim[1] + intercept],  # Right intersection
                    [xlim[1], ylim[0]],  # Bottom-right corner
                ], color='red', alpha=polygon_alpha, zorder=1)
            else:
                # Line goes from top-left to bottom-right
                # Positive gains region (left of the line)
                left_polygon = plt.Polygon([
                    [xlim[0], ylim[0]],  # Bottom-left corner
                    [xlim[1], ylim[0]],  # Bottom-right corner
                    [xlim[1], slope * xlim[1] + intercept],  # Right intersection
                    [xlim[0], slope * xlim[0] + intercept],  # Left intersection
                ], color='green', alpha=polygon_alpha, zorder=1)
                
                # Negative gains region (right of the line)
                right_polygon = plt.Polygon([
                    [xlim[0], slope * xlim[0] + intercept],  # Left intersection
                    [xlim[1], slope * xlim[1] + intercept],  # Right intersection
                    [xlim[1], ylim[1]],  # Top-right corner
                    [xlim[0], ylim[1]],  # Top-left corner
                ], color='red', alpha=polygon_alpha, zorder=1)
            
            # Add the polygons to the plot
            plt.gca().add_patch(left_polygon)
            plt.gca().add_patch(right_polygon)
            
            # Add legend text explaining the regions
            plt.text(0.05, 0.95, "Positive Gains", transform=plt.gca().transAxes, 
                    fontsize=12, color='green', horizontalalignment='left', verticalalignment='top')
            plt.text(0.95, 0.05, "Negative Gains", transform=plt.gca().transAxes,
                    fontsize=12, color='red', horizontalalignment='right', verticalalignment='bottom')
        
        # Function to check if a point is near the line
        def is_near_line(x, y, threshold=0.1):
            if teacher_time == student_time:  # Vertical line case
                return abs(x - teacher_time) < threshold
            else:
                # Distance from point to line: |ax + by + c|/sqrt(a² + b²)
                a = slope
                b = -1
                c = intercept
                distance = abs(a*x + b*y + c) / np.sqrt(a**2 + b**2)
                return distance < threshold
                
        # Add labels with positioning that avoids the line
        for name, acc, time, color in models:
            # Get default position
            x_offset, y_offset, ha, va = get_label_position(time, acc, xlim, ylim)
            
            # Adjust if near the line
            if 'slope' in locals() and is_near_line(time, acc, threshold=0.15):
                # Calculate relative position more precisely
                x_rel = (time - xlim[0]) / (xlim[1] - xlim[0])
                
                if x_rel < 0.3:  # Point is on the left side
                    x_offset = 12
                    ha = 'left'
                elif x_rel > 0.7:  # Point is on the right side
                    x_offset = -12
                    ha = 'right'
                else:  # Point is in the middle
                    # Choose the side with more space
                    if time - xlim[0] > xlim[1] - time:
                        x_offset = -12
                        ha = 'right'
                    else:
                        x_offset = 12
                        ha = 'left'
            # Special case for Downsampled point
            #if name == "Distilled w/ PCD":
            #    x_offset*=1.5
            
            plt.annotate(name, (time, acc), 
                        xytext=(x_offset, y_offset), 
                        textcoords='offset points',
                        fontsize=legend_font_size,
                        ha=ha,
                        va=va,
                        zorder=15)
        
        plt.xlabel(f"Average {metric_type.capitalize()} Time (s)")
        plt.ylabel(f"Average {metric_type.capitalize()} Accuracy (%)")
        plt.savefig(os.path.join(fpath, experiment_name+f"_{metric_type}_efficiency.png"))
        plt.close()

    # Generate all plots
    for metric_type in ["train", "test"]:
        plot_accuracy_curves(metric_type)
        plot_time_bars(metric_type)
        plot_efficiency(metric_type)

def distribution_distillation_experiment(
    dataset: Dataset,
    experiment_name: str,
    params: dict = DISTRIB_DISTILLED_DEFAULTS,
    folderpath: str = DEFAULT_FOLDERPATH,
    save_all: bool = False,
    overwrite: bool = False,
    make_activation_maps: bool = True,
    plot_if_exists: bool = True,
) -> dict:
    """
    Run a distribution-based distillation experiment comparing teacher, baseline, and student models.

    Training flow:
        1. Teacher model trained on original data for teacher_epochs (checkpoint saved)
        2. Baseline model trained on original data for combined_epochs (no KD)
        3. Student model initialized with top-z teacher clauses, then trained for student_epochs
           using soft labels from teacher (KD with temperature and alpha)

    Args:
        dataset (Dataset): The dataset to use for the experiment
        experiment_name (str): Name of the experiment
        params (dict, optional): Parameters for the experiment. Defaults to DISTRIB_DISTILLED_DEFAULTS.
        folderpath (str, optional): Path to save experiment results. Defaults to DEFAULT_FOLDERPATH.
        save_all (bool, optional): Whether to save all models. Defaults to False. If True, saves
            all models to the experiment directory with paths teacher_baseline.pkl, baseline.pkl, student.pkl
        overwrite (bool, optional): Whether to overwrite existing results. Defaults to False.
        make_activation_maps (bool, optional): Whether to generate activation map visualizations. Defaults to True.
        plot_if_exists (bool, optional): Whether to regenerate plots if experiment exists. Defaults to True.

    Returns:
        dict: Dictionary containing experiment results including:
            - Teacher, baseline and student model accuracies and times
            - Analysis statistics (mean, std, etc.)
            - Experiment parameters and metadata
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
                                                       BASELINE_MODEL_PATH,
                                                       STUDENT_MODEL_PATH])
        
        if basic_files_exist and model_files_exist:
            print(f"Experiment {experiment_id} already exists, replotting and skipping")
            # load the results
            results = pd.read_csv(os.path.join(folderpath, experiment_id, RESULTS_CSV_PATH))
            output = load_json(os.path.join(folderpath, experiment_id, OUTPUT_JSON_PATH))
            # plot results
            if plot_if_exists:
                plot_results(output, os.path.join(folderpath, experiment_id))
            return output, results
        else:
            print(f"Experiment {experiment_id} already exists, but some files are missing, continuing") 

    make_dir(os.path.join(folderpath, experiment_id), overwrite=True)
    teacher_model_path = os.path.join(folderpath, experiment_id, TEACHER_CHECKPOINT_PATH)

    # create models
    baseline_tm = MultiClassTsetlinMachine(
        params.student.C, params.student.T, params.student.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)
    baseline_teacher_tm = MultiClassTsetlinMachine(
        params.teacher.C, params.teacher.T, params.teacher.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)
    student_tm = MultiClassTsetlinMachine(
        params.student.C, params.student.T, params.student.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)

    # create a results dataframe
    results = pd.DataFrame(columns=DISTRIBUTION_RESULTS_COLUMNS, index=range(params.combined_epochs))
    try:
        teacher_done, baseline_done, student_done = [False]*3
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
        teacher_done = True
        print(f'Baseline teacher training time: {end-start:.2f} s')

        # copy first teacher_epochs results to student results
        results.loc[:params.teacher.epochs, ACC_TEST_STUDENT] = results.loc[:params.teacher.epochs, ACC_TEST_TEACHER]
        results.loc[:params.teacher.epochs, ACC_TRAIN_STUDENT] = results.loc[:params.teacher.epochs, ACC_TRAIN_TEACHER]
        results.loc[:params.teacher.epochs, TIME_TRAIN_STUDENT] = results.loc[:params.teacher.epochs, TIME_TRAIN_TEACHER]
        results.loc[:params.teacher.epochs, TIME_TEST_STUDENT] = results.loc[:params.teacher.epochs, TIME_TEST_TEACHER]

        # train baseline student
        print(f"Creating a baseline student with {params.student.C} clauses and training on original data")
        start = time()
        bs_pbar = tqdm(range(params.combined_epochs), desc="Baseline", leave=False, dynamic_ncols=True)
        for i in bs_pbar:
            train_result, test_result, train_time, test_time = train_step(baseline_tm, X_train, Y_train, X_test, Y_test, i)
            results.loc[i, ACC_TRAIN_BASELINE], results.loc[i, TIME_TRAIN_BASELINE] = train_result, train_time
            results.loc[i, ACC_TEST_BASELINE], results.loc[i, TIME_TEST_BASELINE] = test_result, test_time
            bs_pbar.set_description(f"Baseline: {results[ACC_TEST_BASELINE].mean():.2f} +/- {results[ACC_TEST_BASELINE].std():.2f}%")

        bs_pbar.close()
        end = time()
        student_done = True
        print(f'Baseline student training time: {end-start:.2f} s')

        print(f"Loading teacher model from {teacher_model_path}, trained for {params.teacher.epochs} epochs")
        teacher_tm = load_pkl(teacher_model_path)
        if not save_all:
            rm_file(teacher_model_path) # remove the teacher model file. we don't need it anymore

        # GET soft labels
        print(f"Initializing student with {params.student.C} clauses from teacher and z={params.z}")
        student_tm.init_from_teacher(teacher_tm, X_train, Y_train, clauses_per_class=params.student.C, z=params.z)
        print(f"Generating soft labels from teacher")
        soft_labels = teacher_tm.get_soft_labels(X_train)
        print(f"Soft labels generated")

        start = time()
        print(f"Training student model for {params.student.epochs} epochs")
        dt_pbar = tqdm(range(params.teacher.epochs, params.combined_epochs), desc="Student", leave=False, dynamic_ncols=True)
        for i in dt_pbar:
            train_result, test_result, train_time, test_time = train_step(student_tm, X_train, Y_train, X_test, Y_test, i, soft_labels, params.temperature, params.alpha)
            results.loc[i, ACC_TRAIN_STUDENT], results.loc[i, TIME_TRAIN_STUDENT] = train_result, train_time
            results.loc[i, ACC_TEST_STUDENT], results.loc[i, TIME_TEST_STUDENT] = test_result, test_time
            dt_pbar.set_description(f"Student: {results[ACC_TEST_STUDENT].mean():.2f} +/- {results[ACC_TEST_STUDENT].std():.2f}%")

        dt_pbar.close()
        end = time()
        student_done = True
        print(f'Teacher-student training time: {end-start:.2f} s')

        if save_all:
            save_pkl(baseline_teacher_tm, os.path.join(folderpath, experiment_id, TEACHER_BASELINE_MODEL_PATH))
            save_pkl(baseline_tm, os.path.join(folderpath, experiment_id, BASELINE_MODEL_PATH))
            save_pkl(student_tm, os.path.join(folderpath, experiment_id, STUDENT_MODEL_PATH))

        total_time = time() - exp_start

        # THIS IS DONE BECAUSE the teacher model will skew inference time when it doesn't actually affect reality
        post_teacher_results = results.iloc[params.teacher.epochs:]

        output = {
            "analysis": {
                # average accuracy on the test set
                "avg_acc_test_teacher": results[ACC_TEST_TEACHER].mean(),
                "avg_acc_test_baseline": results[ACC_TEST_BASELINE].mean(),
                "avg_acc_test_student": results[ACC_TEST_STUDENT].mean(),

                # standard deviation of accuracy on the test set
                "std_acc_test_teacher": results[ACC_TEST_TEACHER].std(),
                "std_acc_test_baseline": results[ACC_TEST_BASELINE].std(),
                "std_acc_test_student": results[ACC_TEST_STUDENT].std(),

                # average accuracy on the training set
                "avg_acc_train_teacher": results[ACC_TRAIN_TEACHER].mean(),
                "avg_acc_train_baseline": results[ACC_TRAIN_BASELINE].mean(),
                "avg_acc_train_student": results[ACC_TRAIN_STUDENT].mean(),

                # standard deviation of accuracy on the training set
                "std_acc_train_teacher": results[ACC_TRAIN_TEACHER].std(),
                "std_acc_train_baseline": results[ACC_TRAIN_BASELINE].std(),
                "std_acc_train_student": results[ACC_TRAIN_STUDENT].std(),

                # final accuracy on the test set
                "final_acc_test_student": results[ACC_TEST_STUDENT].iloc[-1],
                "final_acc_test_teacher": results[ACC_TEST_TEACHER].iloc[-1],
                "final_acc_test_baseline": results[ACC_TEST_BASELINE].iloc[-1],

                # final accuracy on the training set
                "final_acc_train_student": results[ACC_TRAIN_STUDENT].iloc[-1],
                "final_acc_train_teacher": results[ACC_TRAIN_TEACHER].iloc[-1],
                "final_acc_train_baseline": results[ACC_TRAIN_BASELINE].iloc[-1],

                # sum of all training epoch times
                "sum_time_train_teacher": results[TIME_TRAIN_TEACHER].sum(),
                "sum_time_train_baseline": results[TIME_TRAIN_BASELINE].sum(),
                "sum_time_train_student": results[TIME_TRAIN_STUDENT].sum(),

                # sum of all test set evaluation times
                "sum_time_test_teacher": results[TIME_TEST_TEACHER].sum(),
                "sum_time_test_baseline": results[TIME_TEST_BASELINE].sum(),
                "sum_time_test_student": results[TIME_TEST_STUDENT].sum(),

                # average time for each training epoch
                "avg_time_train_teacher": results[TIME_TRAIN_TEACHER].mean(),
                "avg_time_train_baseline": results[TIME_TRAIN_BASELINE].mean(),
                "avg_time_train_student": post_teacher_results[TIME_TRAIN_STUDENT].mean(),

                # average time for each test set evaluation
                "avg_time_test_teacher": results[TIME_TEST_TEACHER].mean(),
                "avg_time_test_baseline": results[TIME_TEST_BASELINE].mean(),
                "avg_time_test_student": post_teacher_results[TIME_TEST_STUDENT].mean(),

                # inference time for each epoch
                "inference_time_teacher": post_teacher_results[TIME_TEST_TEACHER].mean(),
                "inference_time_baseline": post_teacher_results[TIME_TEST_BASELINE].mean(),
                "inference_time_student": post_teacher_results[TIME_TEST_STUDENT].mean(),

                "total_time": total_time,
            },
            "data": {
                "X_train": X_train.shape,
                "Y_train": Y_train.shape,
                "X_test": X_test.shape,
                "Y_test": Y_test.shape,
                "num_classes": len(np.unique(Y_train)),
            },
            "type": "distribution",
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
                visualize_activation_maps(baseline_teacher_tm, baseline_tm, student_tm, 
                                        X_test[samples], Y_test[samples], dataset.image_shape, os.path.join(fpath, 
                                        experiment_name+"_"+ACTIVATION_MAPS_PNG_PATH))
            except Exception as e:
                print(f"Error making activation maps: {e}")
                print("Make sure the dataset has a valid image shape")

        # plot results
        plot_results(output, fpath)

        return output, results
    except Exception as e:
        print("Experiment interrupted")

        # save location
    

        return None, None
    
def aggregate_distribution_distillation_experiment(
    num_experiments: int,
    dataset: Dataset,
    experiment_name: str,
    params: dict = DISTRIB_DISTILLED_DEFAULTS,
    folderpath: str = DEFAULT_FOLDERPATH,
    save_all: bool = False,
    overwrite: bool = False,
    make_activation_maps: bool = True,
) -> dict:
    """
    Aggregate distribution distillation experiments.
    """
    # check that the folderpath exists, if not, create it
    folderpath = os.path.join(folderpath, experiment_name)
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    
    # run each experiment
    outputs = []
    output_dataframe = pd.DataFrame(columns=["acc_train_teacher", "acc_train_baseline", "acc_train_student",
                                             "time_train_teacher", "time_train_baseline", "time_train_student",
                                             "acc_test_teacher", "acc_test_baseline", "acc_test_student",
                                             "time_test_teacher", "time_test_baseline", "time_test_student",
                                             "total_time"])

    # average the json results
    aggregated_output = {
        "experiment_name": experiment_name,
        "num_experiments": num_experiments,
        "params": params,

        "data": {
            "X_train": dataset.X_train.shape,
            "Y_train": dataset.Y_train.shape,
            "X_test": dataset.X_test.shape,
            "Y_test": dataset.Y_test.shape,
            "num_classes": len(np.unique(dataset.Y_train)),
        }
    }
    print(f"Running '{experiment_name}' {num_experiments} times and saving results to {folderpath}")
    for i in range(1, num_experiments+1):
        print(f"Running experiment {i} of {num_experiments}")
        params["_agg_num"] = i
        output, _ = distribution_distillation_experiment(dataset, experiment_name, params, folderpath, save_all, overwrite, make_activation_maps, plot_if_exists=False)
        analysis = output["analysis"]
        output_dataframe.loc[i] = [analysis["avg_acc_train_teacher"], analysis["avg_acc_train_baseline"], analysis["avg_acc_train_student"],
                                  analysis["avg_time_train_teacher"], analysis["avg_time_train_baseline"], analysis["avg_time_train_student"],
                                  analysis["avg_acc_test_teacher"], analysis["avg_acc_test_baseline"], analysis["avg_acc_test_student"],
                                  analysis["avg_time_test_teacher"], analysis["avg_time_test_baseline"], analysis["avg_time_test_student"],
                                  analysis["total_time"]]
        outputs.append(output)


        # normalized time
        aggregated_output["num_experiments"] = i
        aggregated_output["analysis"] = {
            "avg_acc_test_teacher": output_dataframe["acc_test_teacher"].mean(),
            "avg_acc_test_baseline": output_dataframe["acc_test_baseline"].mean(),
            "avg_acc_test_student": output_dataframe["acc_test_student"].mean(),
            "std_acc_test_teacher": output_dataframe["acc_test_teacher"].std(),
            "std_acc_test_baseline": output_dataframe["acc_test_baseline"].std(),
            "std_acc_test_student": output_dataframe["acc_test_student"].std(),

            "avg_acc_train_teacher": output_dataframe["acc_train_teacher"].mean(),
            "avg_acc_train_baseline": output_dataframe["acc_train_baseline"].mean(),
            "avg_acc_train_student": output_dataframe["acc_train_student"].mean(),
            "std_acc_train_teacher": output_dataframe["acc_train_teacher"].std(),
            "std_acc_train_baseline": output_dataframe["acc_train_baseline"].std(),
            "std_acc_train_student": output_dataframe["acc_train_student"].std(),

            "avg_time_train_teacher": output_dataframe["time_train_teacher"].mean(),
            "avg_time_train_baseline": output_dataframe["time_train_baseline"].mean(),
            "avg_time_train_student": output_dataframe["time_train_student"].mean(),
            "std_time_train_teacher": output_dataframe["time_train_teacher"].std(),
            "std_time_train_baseline": output_dataframe["time_train_baseline"].std(),
            "std_time_train_student": output_dataframe["time_train_student"].std(),

            "avg_time_test_teacher": output_dataframe["time_test_teacher"].mean(),
            "avg_time_test_baseline": output_dataframe["time_test_baseline"].mean(),
            "avg_time_test_student": output_dataframe["time_test_student"].mean(),
            "std_time_test_teacher": output_dataframe["time_test_teacher"].std(),
            "std_time_test_baseline": output_dataframe["time_test_baseline"].std(),
            "std_time_test_student": output_dataframe["time_test_student"].std(),

            # normalized time. normalize by teacher time
            "avg_time_train_teacher_normalized": output_dataframe["time_train_teacher"].mean() / output_dataframe["time_train_teacher"].mean(),
            "avg_time_train_baseline_normalized": output_dataframe["time_train_baseline"].mean() / output_dataframe["time_train_teacher"].mean(),
            "avg_time_train_student_normalized": output_dataframe["time_train_student"].mean() / output_dataframe["time_train_teacher"].mean(),
            "std_time_train_teacher_normalized": output_dataframe["time_train_teacher"].transform(lambda x: x / output_dataframe["time_train_teacher"].mean()).std(),
            "std_time_train_baseline_normalized": output_dataframe["time_train_baseline"].transform(lambda x: x / output_dataframe["time_train_teacher"].mean()).std(),
            "std_time_train_student_normalized": output_dataframe["time_train_student"].transform(lambda x: x / output_dataframe["time_train_teacher"].mean()).std(),

            "avg_time_test_teacher_normalized": output_dataframe["time_test_teacher"].mean() / output_dataframe["time_test_teacher"].mean(),
            "avg_time_test_baseline_normalized": output_dataframe["time_test_baseline"].mean() / output_dataframe["time_test_teacher"].mean(),
            "avg_time_test_student_normalized": output_dataframe["time_test_student"].mean() / output_dataframe["time_test_teacher"].mean(),
            "std_time_test_teacher_normalized": output_dataframe["time_test_teacher"].transform(lambda x: x / output_dataframe["time_test_teacher"].mean()).std(),
            "std_time_test_baseline_normalized": output_dataframe["time_test_baseline"].transform(lambda x: x / output_dataframe["time_test_teacher"].mean()).std(),
            "std_time_test_student_normalized": output_dataframe["time_test_student"].transform(lambda x: x / output_dataframe["time_test_teacher"].mean()).std(),

            "avg_total_time": output_dataframe["total_time"].mean(),
            "std_total_time": output_dataframe["total_time"].std(),
        }
        save_json(aggregated_output, os.path.join(folderpath, AGGREGATED_OUTPUT_JSON_PATH))
    
    del params["_agg_num"]

    # save aggregated output
    save_json(aggregated_output, os.path.join(folderpath, AGGREGATED_OUTPUT_JSON_PATH))
    output_dataframe.to_csv(os.path.join(folderpath, AGGREGATED_RESULTS_CSV_PATH))

    return aggregated_output
    


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
            This doesn't usually make a difference to the student model's performance.

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
            This doesn't usually make a difference to the student model's performance.

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
) -> dict:
    """
    Run a clause-based distillation experiment comparing teacher, baseline, and student models.

    Training flow:
        1. Teacher model trained on original data for combined_epochs (checkpoint saved after teacher_epochs)
        2. Baseline model trained on original data for combined_epochs (no KD)
        3. Student model trained on transformed teacher outputs for student_epochs (clause-based KD)
        4. Student_ds model trained on downsampled transformed outputs (if downsample > 0)

    Args:
        dataset (Dataset): The dataset to use for the experiment
        experiment_name (str): Name of the experiment
        params (dict, optional): Parameters for the experiment. Defaults to CLAUSE_DISTILLED_DEFAULTS.
        folderpath (str, optional): Path to save experiment results. Defaults to DEFAULT_FOLDERPATH.
        save_all (bool, optional): Whether to save all models. Defaults to False. If True, saves
            all models to the experiment directory with paths teacher_baseline.pkl, baseline.pkl, student.pkl
        overwrite (bool, optional): Whether to overwrite existing results. Defaults to False.

    Returns:
        dict: Dictionary containing experiment results including:
            - Teacher, baseline and student model accuracies
            - Training and testing times
            - Number of clauses dropped during downsampling
            - Total experiment time
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
                                                       BASELINE_MODEL_PATH,
                                                       STUDENT_MODEL_PATH])
        
        if basic_files_exist and model_files_exist:
            print(f"Experiment {experiment_id} already exists, replotting and skipping")
            # load the results
            results = pd.read_csv(os.path.join(folderpath, experiment_id, RESULTS_CSV_PATH))
            output = load_json(os.path.join(folderpath, experiment_id, OUTPUT_JSON_PATH))
            # plot results
            plot_results(output, os.path.join(folderpath, experiment_id), output["params"]["downsample"])
            return output, results
        else:
            print(f"Experiment {experiment_id} already exists, but some files are missing, continuing") 

    make_dir(os.path.join(folderpath, experiment_id), overwrite=True)
    teacher_model_path = os.path.join(folderpath, experiment_id, TEACHER_CHECKPOINT_PATH)

    # create models
    baseline_tm = MultiClassTsetlinMachine(
        params.student.C, params.student.T, params.student.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)
    baseline_teacher_tm = MultiClassTsetlinMachine(
        params.teacher.C, params.teacher.T, params.teacher.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)
    student_tm = MultiClassTsetlinMachine(
        params.student.C, params.student.T, params.student.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)
    student_ds_tm = MultiClassTsetlinMachine(
        params.student.C, params.student.T, params.student.s, number_of_state_bits=params.number_of_state_bits, weighted_clauses=params.weighted_clauses)

    # create a results dataframe
    results = pd.DataFrame(columns=CLAUSE_RESULTS_COLUMNS, index=range(params.combined_epochs))
    
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

    # copy first teacher_epochs results to student results and student_ds results
    results.loc[:params.teacher.epochs, ACC_TEST_STUDENT] = results.loc[:params.teacher.epochs, ACC_TEST_TEACHER]
    results.loc[:params.teacher.epochs, ACC_TRAIN_STUDENT] = results.loc[:params.teacher.epochs, ACC_TRAIN_TEACHER]
    results.loc[:params.teacher.epochs, TIME_TRAIN_STUDENT] = results.loc[:params.teacher.epochs, TIME_TRAIN_TEACHER]
    results.loc[:params.teacher.epochs, TIME_TEST_STUDENT] = results.loc[:params.teacher.epochs, TIME_TEST_TEACHER]
    results.loc[:params.teacher.epochs, ACC_TEST_STUDENT_DS] = results.loc[:params.teacher.epochs, ACC_TEST_TEACHER]
    results.loc[:params.teacher.epochs, ACC_TRAIN_STUDENT_DS] = results.loc[:params.teacher.epochs, ACC_TRAIN_TEACHER]
    results.loc[:params.teacher.epochs, TIME_TEST_STUDENT_DS] = results.loc[:params.teacher.epochs, TIME_TEST_TEACHER]
    results.loc[:params.teacher.epochs, TIME_TRAIN_STUDENT_DS] = results.loc[:params.teacher.epochs, TIME_TRAIN_TEACHER]

    # train baseline
    print(f"Creating a baseline with {params.student.C} clauses and training on original data")
    start = time()
    bs_pbar = tqdm(range(params.combined_epochs), desc="Baseline", leave=False, dynamic_ncols=True)
    for i in bs_pbar:
        train_result, test_result, train_time, test_time = train_step(baseline_tm, X_train, Y_train, X_test, Y_test, i)
        results.loc[i, ACC_TRAIN_BASELINE], results.loc[i, TIME_TRAIN_BASELINE] = train_result, train_time
        results.loc[i, ACC_TEST_BASELINE], results.loc[i, TIME_TEST_BASELINE] = test_result, test_time
        bs_pbar.set_description(f"Baseline: {results[ACC_TEST_BASELINE].mean():.2f} +/- {results[ACC_TEST_BASELINE].std():.2f}%")

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

    start = time()
    print(f"Training student model for {params.student.epochs} epochs")
    dt_pbar = tqdm(range(params.teacher.epochs, params.combined_epochs), desc="Student", leave=False, dynamic_ncols=True)
    for i in dt_pbar:
        train_result, test_result, train_time, test_time = train_step(student_tm, X_train_transformed, Y_train, X_test_transformed, Y_test, i)
        results.loc[i, ACC_TRAIN_STUDENT], results.loc[i, TIME_TRAIN_STUDENT] = train_result, train_time
        results.loc[i, ACC_TEST_STUDENT], results.loc[i, TIME_TEST_STUDENT] = test_result, test_time
        dt_pbar.set_description(f"Student: {results[ACC_TEST_STUDENT].mean():.2f} +/- {results[ACC_TEST_STUDENT].std():.2f}%")

    dt_pbar.close()
    end = time()

    print(f'Student (no downsampling) training time: {end-start:.2f} s')

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

    # train student downsampled
    print(f"Training student DS model for {params.student.epochs} epochs")
    dt_pbar = tqdm(range(params.teacher.epochs, params.combined_epochs), desc="Student DS", leave=False, dynamic_ncols=True)
    for i in dt_pbar:
        train_result, test_result, train_time, test_time = train_step(student_ds_tm, X_train_downsampled, Y_train, X_test_downsampled, Y_test, i)
        results.loc[i, ACC_TRAIN_STUDENT_DS], results.loc[i, TIME_TRAIN_STUDENT_DS] = train_result, train_time
        results.loc[i, ACC_TEST_STUDENT_DS], results.loc[i, TIME_TEST_STUDENT_DS] = test_result, test_time
        dt_pbar.set_description(f"Student DS: {results[ACC_TEST_STUDENT_DS].mean():.2f} +/- {results[ACC_TEST_STUDENT_DS].std():.2f}%")

    dt_pbar.close()
    end = time()

    print(f'Student (downsampled) training time: {end-start:.2f} s')

    if save_all:
        save_pkl(baseline_teacher_tm, os.path.join(folderpath, experiment_id, TEACHER_BASELINE_MODEL_PATH))
        save_pkl(baseline_tm, os.path.join(folderpath, experiment_id, BASELINE_MODEL_PATH))
        save_pkl(student_tm, os.path.join(folderpath, experiment_id, STUDENT_MODEL_PATH))
        save_pkl(student_ds_tm, os.path.join(folderpath, experiment_id, STUDENT_DS_MODEL_PATH))
    total_time = time() - exp_start

    # THIS IS DONE BECAUSE the teacher model will skew inference time when it doesn't actually affect reality
    post_teacher_results = results.iloc[params.teacher.epochs:]

    output = {
        "analysis": {
            # average accuracy on the test set
            "avg_acc_test_teacher": results[ACC_TEST_TEACHER].mean(),
            "avg_acc_test_baseline": results[ACC_TEST_BASELINE].mean(),
            "avg_acc_test_student": results[ACC_TEST_STUDENT].mean(),
            "avg_acc_test_student_ds": results[ACC_TEST_STUDENT_DS].mean(),

            # standard deviation of accuracy on the test set
            "std_acc_test_teacher": results[ACC_TEST_TEACHER].std(),
            "std_acc_test_baseline": results[ACC_TEST_BASELINE].std(),
            "std_acc_test_student": results[ACC_TEST_STUDENT].std(),
            "std_acc_test_student_ds": results[ACC_TEST_STUDENT_DS].std(),

            # average accuracy on the training set
            "avg_acc_train_teacher": results[ACC_TRAIN_TEACHER].mean(),
            "avg_acc_train_baseline": results[ACC_TRAIN_BASELINE].mean(),
            "avg_acc_train_student": results[ACC_TRAIN_STUDENT].mean(),
            "avg_acc_train_student_ds": results[ACC_TRAIN_STUDENT_DS].mean(),

            # standard deviation of accuracy on the training set
            "std_acc_train_teacher": results[ACC_TRAIN_TEACHER].std(),
            "std_acc_train_baseline": results[ACC_TRAIN_BASELINE].std(),
            "std_acc_train_student": results[ACC_TRAIN_STUDENT].std(),
            "std_acc_train_student_ds": results[ACC_TRAIN_STUDENT_DS].std(),
            # final accuracy on the test set
            "final_acc_test_student": results[ACC_TEST_STUDENT].iloc[-1],
            "final_acc_test_teacher": results[ACC_TEST_TEACHER].iloc[-1],
            "final_acc_test_baseline": results[ACC_TEST_BASELINE].iloc[-1],
            "final_acc_test_student_ds": results[ACC_TEST_STUDENT_DS].iloc[-1],
            # final accuracy on the training set
            "final_acc_train_student": results[ACC_TRAIN_STUDENT].iloc[-1],
            "final_acc_train_teacher": results[ACC_TRAIN_TEACHER].iloc[-1],
            "final_acc_train_baseline": results[ACC_TRAIN_BASELINE].iloc[-1],
            "final_acc_train_student_ds": results[ACC_TRAIN_STUDENT_DS].iloc[-1],

            # sum of all training epoch times
            "sum_time_train_teacher": results[TIME_TRAIN_TEACHER].sum(),
            "sum_time_train_baseline": results[TIME_TRAIN_BASELINE].sum(),
            "sum_time_train_student": results[TIME_TRAIN_STUDENT].sum(),
            "sum_time_train_student_ds": results[TIME_TRAIN_STUDENT_DS].sum(),

            # sum of all test set evaluation times
            "sum_time_test_teacher": results[TIME_TEST_TEACHER].sum(),
            "sum_time_test_baseline": results[TIME_TEST_BASELINE].sum(),
            "sum_time_test_student": results[TIME_TEST_STUDENT].sum(),
            "sum_time_test_student_ds": results[TIME_TEST_STUDENT_DS].sum(),

            # average time for each training epoch
            "avg_time_train_teacher": results[TIME_TRAIN_TEACHER].mean(),
            "avg_time_train_baseline": results[TIME_TRAIN_BASELINE].mean(),
            "avg_time_train_student": results[TIME_TRAIN_STUDENT].mean(),
            "avg_time_train_student_ds": results[TIME_TRAIN_STUDENT_DS].mean(),

            # average time for each test set evaluation
            "avg_time_test_teacher": results[TIME_TEST_TEACHER].mean(),
            "avg_time_test_baseline": results[TIME_TEST_BASELINE].mean(),
            "avg_time_test_student": results[TIME_TEST_STUDENT].mean(),
            "avg_time_test_student_ds": results[TIME_TEST_STUDENT_DS].mean(),

            # inference time for each epoch
            "inference_time_teacher": post_teacher_results[TIME_TEST_TEACHER].mean(),
            "inference_time_baseline": post_teacher_results[TIME_TEST_BASELINE].mean(),
            "inference_time_student": post_teacher_results[TIME_TEST_STUDENT].mean(),
            "inference_time_student_ds": post_teacher_results[TIME_TEST_STUDENT_DS].mean(),
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
            "test_transform_time": test_transform_time,
            "train_transform_time": train_transform_time,
        },
        "params": params.to_dict(),
        "type": "clause",
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
    plot_results(output, fpath, downsample=params.downsample)

    return output, results
