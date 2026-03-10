# Use this file to fix plots and calculations after the experiments are done
import csv
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from util import load_pkl, load_json, save_json
from __init__ import  OUTPUT_JSON_PATH, PLOT_FIGSIZE, PLOT_DPI, ACC_TEST_DISTILLED, ACC_TRAIN_DISTILLED, AGGREGATED_OUTPUT_JSON_PATH
import os 
def iterate_over_file_in_folder(folder="experiments", file_extension=".json"):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    yield data, file_path

def make_paper_1_tables(exps: list[tuple[str, str]]):
    """
    exps: list of experiment directory paths
    """

    # hyperparameter table
    hyperparam_table = pd.DataFrame(columns=["Dataset", "$|C_T|$", "$|C_S|$", "$T_T$", "$T_S$", "$s_T$", "$s_S$", "$\\delta^*$", "$E_T$", "$E_S$"], index=[])

    # dataset size table
    dataset_size_table = pd.DataFrame(columns=["Dataset", "$|X_{train}|$", "$|X_{test}|$", "$|L|$", "$\\zeta$", "$\\delta^*\\%$"], index=[])

    # combined train table with time
    train_table = pd.DataFrame(columns=["Dataset", "$Acc'_T$", "$\\mathcal{T}'_T$", "$Acc'_S$", "$\\mathcal{T}'_S$", "$Acc'_D$", "$\\mathcal{T}'_D$"], index=[])
    test_table = pd.DataFrame(columns=["Dataset", "$Acc_T$", "$\\mathcal{T}_T$", "$Acc_S$", "$\\mathcal{T}_S$", "$Acc_D$", "$\\mathcal{T}_D$"], index=[])

    for exp in exps:
        print(exp)
        exp_output = load_json(os.path.join(exp, OUTPUT_JSON_PATH))

        # get row name
        rowname = exp_output["experiment_name"]

        # get hyperparameters
        new_row = {
            "Dataset": rowname,
            "$|C_T|$": f'{exp_output["params"]["teacher"]["C"]}',
            "$|C_S|$": f'{exp_output["params"]["student"]["C"]}',
            "$T_T$": f'{exp_output["params"]["teacher"]["T"]}',
            "$T_S$": f'{exp_output["params"]["student"]["T"]}',
            "$s_T$": f'{exp_output["params"]["teacher"]["s"]}',
            "$s_S$": f'{exp_output["params"]["student"]["s"]}',
            "$\\delta^*$": f'{exp_output["params"]["downsample"]}',
            "$E_T$": f'{exp_output["params"]["teacher"]["epochs"]}',
            "$E_S$": f'{exp_output["params"]["student"]["epochs"]}'
        }
        hyperparam_table = hyperparam_table._append(new_row, ignore_index=True) 

        # get dataset size
        new_row = {
            "Dataset": rowname,
            "$|X_{train}|$": f'{exp_output["data"]["X_train"][0]}',
            "$|X_{test}|$": f'{exp_output["data"]["X_test"][0]}',
            "$|L|$": f'{exp_output["data"]["X_train"][1]}',
            "$\\zeta$": f'{exp_output["data"]["num_classes"]}',
            "$\\delta^*\\%$": f'{exp_output["downsample_info"]["reduction_percentage"]:.2f}'
        }
        dataset_size_table = dataset_size_table._append(new_row, ignore_index=True)

        # get combined train table with time
        new_row = {
            "Dataset": rowname,
            "$Acc'_T$": f'{exp_output["analysis"]["avg_acc_train_teacher"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_train_teacher"]:.2f}',
            "$Acc'_S$": f'{exp_output["analysis"]["avg_acc_train_student"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_train_student"]:.2f}',
            "$Acc'_D$": f'{exp_output["analysis"]["avg_acc_train_distilled"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_train_distilled"]:.2f}',
            "$Acc'^{down}_D$": f'{exp_output["analysis"]["avg_acc_train_distilled_ds"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_train_distilled_ds"]:.2f}',
            "$\\mathcal{T}'_T$": f'{exp_output["analysis"]["avg_time_train_teacher"]:.2f}',
            "$\\mathcal{T}'_S$": f'{exp_output["analysis"]["avg_time_train_student"]:.2f}',
            "$\\mathcal{T}'_D$": f'{exp_output["analysis"]["avg_time_train_distilled"]:.2f}',
            "$\\mathcal{T}'_{PCD}$": f'{exp_output["analysis"]["avg_time_train_distilled_ds"]:.2f}'
        }
        train_table = train_table._append(new_row, ignore_index=True)
        
        new_row = {
            "Dataset": rowname,
            "$Acc_T$": f'{exp_output["analysis"]["avg_acc_test_teacher"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_test_teacher"]:.2f}',
            "$Acc_S$": f'{exp_output["analysis"]["avg_acc_test_student"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_test_student"]:.2f}',
            "$Acc_D$": f'{exp_output["analysis"]["avg_acc_test_distilled"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_test_distilled"]:.2f}',
            "$Acc'^{down}_D$": f'{exp_output["analysis"]["avg_acc_test_distilled_ds"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_test_distilled_ds"]:.2f}',
            "$\\mathcal{T}_T$": f'{exp_output["analysis"]["avg_time_test_teacher"]:.2f}',
            "$\\mathcal{T}_S$": f'{exp_output["analysis"]["avg_time_test_student"]:.2f}',
            "$\\mathcal{T}_D$": f'{exp_output["analysis"]["avg_time_test_distilled"]:.2f}',
            "$\\mathcal{T}'_{PCD}$": f'{exp_output["analysis"]["avg_time_test_distilled_ds"]:.2f}'
        }
        test_table = test_table._append(new_row, ignore_index=True)

    # Define table configurations
    table_configs = [
        {
            "name": "hyperparam_table",
            "file_name": "hyperparam_table",
            "caption": "Experiment Hyperparameters (CKD)",
            "label": "tab:hyperparams-ckd",
            "column_format": "l"*len(hyperparam_table.columns)
        },
        {
            "name": "dataset_size_table",
            "file_name": "dataset_size_table",
            "caption": "Dataset Size",
            "label": "tab:dataset-size-ckd",
            "column_format": "l"*len(dataset_size_table.columns)
        },
        {
            "name": "train_table",
            "file_name": "train_table",
            "caption": "Training Results (CKD)",
            "label": "tab:train-table-ckd",
            "column_format": "l"+("c"*(len(train_table.columns)-1))
        },
        {
            "name": "test_table",
            "file_name": "test_table",
            "caption": "Testing Results (CKD)",
            "label": "tab:test-table-ckd",
            "column_format": "l"+("c"*(len(test_table.columns)-1))
        }
    ]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join("assets", "paper_1")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tables in CSV and LaTeX formats
    for config in table_configs:
        table = locals()[config["name"]]
        
        # Save CSV
        # csv_path = os.path.join(output_dir, f"{config['file_name']}.csv")
        # table.to_csv(csv_path, index=False)
        
        # Save LaTeX
        latex_table = table.to_latex(
            index=False, 
            escape=False, 
            column_format=config["column_format"],
            caption=config["caption"], 
            label=config["label"],
        )
        latex_path = os.path.join(output_dir, f"{config['file_name']}.tex")
        with open(latex_path, "w") as f:
            f.write(latex_table)

def make_experiment_tables(exps: list[tuple[str, str]]):
    """
    exps: list of experiment directory paths
    """

    # hyperparameter table
    hyperparam_table = pd.DataFrame(columns=["Dataset", "$|C_T|$", "$|C_S|$", "$T_T$", "$T_S$", "$s_T$", "$s_S$", "$\\tau$", "$\\alpha$", "$z$", "$E_T$", "$E_S$"], index=[])

    # dataset size table
    dataset_size_table = pd.DataFrame(columns=["Dataset", "$|X_{train}|$", "$|X_{test}|$", "$|L|$", "$\\zeta$"], index=[])

    # combined train table with time
    train_table = pd.DataFrame(columns=["Dataset", "$Acc'_T$", "$\\mathcal{T}'_T$", "$Acc'_S$", "$\\mathcal{T}'_S$", "$Acc'_D$", "$\\mathcal{T}'_D$"], index=[])
    test_table = pd.DataFrame(columns=["Dataset", "$Acc_T$", "$\\mathcal{T}_T$", "$Acc_S$", "$\\mathcal{T}_S$", "$Acc_D$", "$\\mathcal{T}_D$"], index=[])

    for exp in exps:
        exp_output = load_json(os.path.join(exp, OUTPUT_JSON_PATH))

        # get row name
        rowname = exp_output["experiment_name"]

        # get hyperparameters
        new_row = {
            "Dataset": rowname,
            "$|C_T|$": f'{exp_output["params"]["teacher"]["C"]}',
            "$|C_S|$": f'{exp_output["params"]["student"]["C"]}',
            "$T_T$": f'{exp_output["params"]["teacher"]["T"]}',
            "$T_S$": f'{exp_output["params"]["student"]["T"]}',
            "$s_T$": f'{exp_output["params"]["teacher"]["s"]}',
            "$s_S$": f'{exp_output["params"]["student"]["s"]}',
            "$\\tau$": f'{exp_output["params"]["temperature"]}',
            "$\\alpha$": f'{exp_output["params"]["alpha"]}',
            "$z$": f'{exp_output["params"]["z"]}',
            "$E_T$": f'{exp_output["params"]["teacher"]["epochs"]}',
            "$E_S$": f'{exp_output["params"]["student"]["epochs"]}'
        }
        hyperparam_table = hyperparam_table._append(new_row, ignore_index=True) 

        # get dataset size
        new_row = {
            "Dataset": rowname,
            "$|X_{train}|$": f'{exp_output["data"]["X_train"][0]}',
            "$|X_{test}|$": f'{exp_output["data"]["X_test"][0]}',
            "$|L|$": f'{exp_output["data"]["X_train"][1]}',
            "$\\zeta$": f'{exp_output["data"]["num_classes"]}'
        }
        dataset_size_table = dataset_size_table._append(new_row, ignore_index=True)

        # get combined train table with time
        new_row = {
            "Dataset": rowname,
            "$Acc'_T$": f'{exp_output["analysis"]["avg_acc_train_teacher"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_train_teacher"]:.2f}',
            "$Acc'_S$": f'{exp_output["analysis"]["avg_acc_train_student"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_train_student"]:.2f}',
            "$Acc'_D$": f'{exp_output["analysis"]["avg_acc_train_distilled"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_train_distilled"]:.2f}',
            "$\\mathcal{T}'_T$": f'{exp_output["analysis"]["avg_time_train_teacher"]:.2f}',
            "$\\mathcal{T}'_S$": f'{exp_output["analysis"]["avg_time_train_student"]:.2f}',
            "$\\mathcal{T}'_D$": f'{exp_output["analysis"]["avg_time_train_distilled"]:.2f}'
        }
        train_table = train_table._append(new_row, ignore_index=True)
        
        new_row = {
            "Dataset": rowname,
            "$Acc_T$": f'{exp_output["analysis"]["avg_acc_test_teacher"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_test_teacher"]:.2f}',
            "$Acc_S$": f'{exp_output["analysis"]["avg_acc_test_student"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_test_student"]:.2f}',
            "$Acc_D$": f'{exp_output["analysis"]["avg_acc_test_distilled"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_test_distilled"]:.2f}',
            "$\\mathcal{T}_T$": f'{exp_output["analysis"]["avg_time_test_teacher"]:.2f}',
            "$\\mathcal{T}_S$": f'{exp_output["analysis"]["avg_time_test_student"]:.2f}',
            "$\\mathcal{T}_D$": f'{exp_output["analysis"]["avg_time_test_distilled"]:.2f}'
        }
        test_table = test_table._append(new_row, ignore_index=True)

    # Define table configurations
    table_configs = [
        {
            "name": "hyperparam_table",
            "file_name": "hyperparam_table",
            "caption": "Experiment Hyperparameters (DKD)",
            "label": "tab:hyperparams-dkd",
            "column_format": "l"*len(hyperparam_table.columns)
        },
        {
            "name": "dataset_size_table",
            "file_name": "dataset_size_table",
            "caption": "Dataset Size (DKD)",
            "label": "tab:dataset-size-dkd",
            "column_format": "l"*len(dataset_size_table.columns)
        },
        {
            "name": "train_table",
            "file_name": "train_table",
            "caption": "Training Results (DKD)",
            "label": "tab:train-table-dkd",
            "column_format": "l"+("c"*(len(train_table.columns)-1))
        },
        {
            "name": "test_table",
            "file_name": "test_table",
            "caption": "Testing Results (DKD)",
            "label": "tab:test-table-dkd",
            "column_format": "l"+("c"*(len(test_table.columns)-1))
        }
    ]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join("assets", "experiment")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tables in CSV and LaTeX formats
    for config in table_configs:
        table = locals()[config["name"]]
        
        # Save CSV
        # csv_path = os.path.join(output_dir, f"{config['file_name']}.csv")
        # table.to_csv(csv_path, index=False)
        
        # Save LaTeX
        latex_table = table.to_latex(
            index=False, 
            escape=False, 
            column_format=config["column_format"],
            caption=config["caption"], 
            label=config["label"],
        )
        latex_path = os.path.join(output_dir, f"{config['file_name']}.tex")
        with open(latex_path, "w") as f:
            f.write(latex_table)
        
def _to_latex_hline(df, column_format, caption, label):
    """Like df.to_latex() but produces \\hline-style rules and table* environment."""
    raw = df.to_latex(index=False, escape=False, column_format=column_format,
                      caption=caption, label=label)
    raw = raw.replace(r'\begin{table}', r'\begin{table*}')
    raw = raw.replace(r'\end{table}', r'\end{table*}')
    lines = raw.splitlines()
    result = []
    past_midrule = False
    for line in lines:
        stripped = line.strip()
        if stripped == r'\toprule':
            result.append(r'\hline')
        elif stripped == r'\midrule':
            result.append(r'\hline')
            past_midrule = True
        elif stripped == r'\bottomrule':
            pass  # last data row's appended \hline already closes the table
        elif past_midrule and stripped.endswith(r'\\'):
            result.append(line + r'\hline')
        else:
            result.append(line)
    return '\n'.join(result)

def make_experiment_tables_aggregate(exps: list[tuple[str, str]]):
    """
    top_dir: top directory of aggregate results
    """

    # hyperparameter table
    hyperparam_table = pd.DataFrame(columns=["Dataset", "$|C_T|$", "$|C_S|$", "$T_T$", "$T_S$", "$s_T$", "$s_S$", "$\\tau$", "$\\alpha$", "$z$", "$E_T$", "$E_S$", "$K$"], index=[])

    # dataset size table
    dataset_size_table = pd.DataFrame(columns=["Dataset", "$|X_{train}|$", "$|X_{test}|$", "$|L|$", "$\\zeta$"], index=[])

    # combined train table with time
    train_table = pd.DataFrame(columns=["Dataset", "$Acc'_T$", "$\\mathcal{T}'_T$", "$Acc'_B$", "$\\mathcal{T}'_B$", "$Acc'_S$", "$\\mathcal{T}'_S$"], index=[])
    test_table = pd.DataFrame(columns=["Dataset", "$Acc_T$", "$\\mathcal{T}_T$", "$Acc_B$", "$\\mathcal{T}_B$", "$Acc_S$", "$\\mathcal{T}_S$"], index=[])

    for exp in exps:
        # load aggregated output
        if not os.path.exists(os.path.join(exp, AGGREGATED_OUTPUT_JSON_PATH)):
            print(f"No aggregated output found for {exp}")
            continue
        exp_output = load_json(os.path.join(exp, AGGREGATED_OUTPUT_JSON_PATH))

        # get row name
        rowname = exp_output["experiment_name"]

        # get hyperparameters
        new_row = {
            "Dataset": rowname,
            "$|C_T|$": f'{exp_output["params"]["teacher"]["C"]}',
            "$|C_S|$": f'{exp_output["params"]["student"]["C"]}',
            "$T_T$": f'{exp_output["params"]["teacher"]["T"]}',
            "$T_S$": f'{exp_output["params"]["student"]["T"]}',
            "$s_T$": f'{exp_output["params"]["teacher"]["s"]}',
            "$s_S$": f'{exp_output["params"]["student"]["s"]}',
            "$\\tau$": f'{exp_output["params"]["temperature"]}',
            "$\\alpha$": f'{exp_output["params"]["alpha"]}',
            "$z$": f'{exp_output["params"]["z"]}',
            "$E_T$": f'{exp_output["params"]["teacher"]["epochs"]}',
            "$E_S$": f'{exp_output["params"]["student"]["epochs"]}',
            "$K$": f'{exp_output["num_experiments"]}'
        }
        hyperparam_table = hyperparam_table._append(new_row, ignore_index=True)

        # get dataset size
        new_row = {
            "Dataset": rowname,
            "$|X_{train}|$": f'{exp_output["data"]["X_train"][0]}',
            "$|X_{test}|$": f'{exp_output["data"]["X_test"][0]}',
            "$|L|$": f'{exp_output["data"]["X_train"][1]}',
            "$\\zeta$": f'{exp_output["data"]["num_classes"]}'
        }
        dataset_size_table = dataset_size_table._append(new_row, ignore_index=True)

        # get combined train table with time
        new_row = {
            "Dataset": rowname,
            "$Acc'_T$": f'{exp_output["analysis"]["avg_acc_train_teacher"]:.2f} \\newline $\pm$ {exp_output["analysis"]["std_acc_train_teacher"]:.2f}',
            "$Acc'_B$": f'{exp_output["analysis"]["avg_acc_train_student"]:.2f} \\newline $\pm$ {exp_output["analysis"]["std_acc_train_student"]:.2f}',
            "$Acc'_S$": f'{exp_output["analysis"]["avg_acc_train_distilled"]:.2f} \\newline $\pm$ {exp_output["analysis"]["std_acc_train_distilled"]:.2f}',
            "$\\mathcal{T}'_T$": f'{exp_output["analysis"]["avg_time_train_teacher"]:.2f} \\newline $\pm$ {exp_output["analysis"]["std_time_train_teacher"]:.2f}',
            "$\\mathcal{T}'_B$": f'{exp_output["analysis"]["avg_time_train_student"]:.2f} \\newline $\pm$ {exp_output["analysis"]["std_time_train_student"]:.2f}',
            "$\\mathcal{T}'_S$": f'{exp_output["analysis"]["avg_time_train_distilled"]:.2f} \\newline $\pm$ {exp_output["analysis"]["std_time_train_distilled"]:.2f}'
        }
        train_table = train_table._append(new_row, ignore_index=True)

        new_row = {
            "Dataset": rowname,
            "$Acc_T$": f'{exp_output["analysis"]["avg_acc_test_teacher"]:.2f} \\newline $\pm$ {exp_output["analysis"]["std_acc_test_teacher"]:.2f}',
            "$Acc_B$": f'{exp_output["analysis"]["avg_acc_test_student"]:.2f} \\newline $\pm$ {exp_output["analysis"]["std_acc_test_student"]:.2f}',
            "$Acc_S$": f'{exp_output["analysis"]["avg_acc_test_distilled"]:.2f} \\newline $\pm$ {exp_output["analysis"]["std_acc_test_distilled"]:.2f}',
            "$\\mathcal{T}_T$": f'{exp_output["analysis"]["avg_time_test_teacher"]:.2f} \\newline $\pm$ {exp_output["analysis"]["std_time_test_teacher"]:.2f}',
            "$\\mathcal{T}_B$": f'{exp_output["analysis"]["avg_time_test_student"]:.2f} \\newline $\pm$ {exp_output["analysis"]["std_time_test_student"]:.2f}',
            "$\\mathcal{T}_S$": f'{exp_output["analysis"]["avg_time_test_distilled"]:.2f} \\newline $\pm$ {exp_output["analysis"]["std_time_test_distilled"]:.2f}'
        }
        test_table = test_table._append(new_row, ignore_index=True)

    # Define table configurations
    table_configs = [
        {
            "name": "hyperparam_table",
            "file_name": "hyperparam_table",
            "caption": "Experiment Hyperparameters",
            "label": "tab:hyperparams-dkd",
            "column_format": "|l|" + "c|" * (len(hyperparam_table.columns) - 1)
        },
        {
            "name": "dataset_size_table",
            "file_name": "dataset_size_table",
            "caption": "Dataset Size",
            "label": "tab:dataset-size-dkd",
            "column_format": "|l|" + "c|" * (len(dataset_size_table.columns) - 1)
        },
        {
            "name": "train_table",
            "file_name": "train_table",
            "caption": "Training Results",
            "label": "tab:train-table-dkd",
            "column_format": "|l|" + "c|" * (len(train_table.columns) - 1)
        },
        {
            "name": "test_table",
            "file_name": "test_table",
            "caption": "Testing Results",
            "label": "tab:test-table-dkd",
            "column_format": "|l|" + "c|" * (len(test_table.columns) - 1)
        }
    ]

    # Create output directory if it doesn't exist
    output_dir = os.path.join("assets", "experiment")
    os.makedirs(output_dir, exist_ok=True)

    # Save tables in CSV and LaTeX formats
    for config in table_configs:
        table = locals()[config["name"]]

        # Save CSV
        # csv_path = os.path.join(output_dir, f"{config['file_name']}.csv")
        # table.to_csv(csv_path, index=False)

        # Save LaTeX
        latex_table = _to_latex_hline(table, config["column_format"], config["caption"], config["label"])
        latex_path = os.path.join(output_dir, f"{config['file_name']}.tex")
        with open(latex_path, "w") as f:
            f.write(latex_table)


def make_formatted_tables(exps: list[str]):
    """
    Produce combined_test_table.tex and combined_train_table.tex with datasets as
    columns and metrics as rows. Includes mean ± std for Teacher, Baseline, Student
    (with t-test significance stars on Student), and Δ (S − B). A double \\hline
    separates the accuracy block from the time block.
    """
    def _stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    # Per phase: (acc_teacher_key, acc_baseline_key, acc_student_key, time_teacher_key, ...)
    phase_keys = {
        "train": (
            "avg_acc_train_teacher", "std_acc_train_teacher",
            "avg_acc_train_student", "std_acc_train_student",
            "avg_acc_train_distilled", "std_acc_train_distilled",
            "avg_time_train_teacher", "std_time_train_teacher",
            "avg_time_train_student", "std_time_train_student",
            "avg_time_train_distilled", "std_time_train_distilled",
        ),
        "test": (
            "avg_acc_test_teacher", "std_acc_test_teacher",
            "avg_acc_test_student", "std_acc_test_student",
            "avg_acc_test_distilled", "std_acc_test_distilled",
            "avg_time_test_teacher", "std_time_test_teacher",
            "avg_time_test_student", "std_time_test_student",
            "avg_time_test_distilled", "std_time_test_distilled",
        ),
    }

    # Collect per-dataset: aggregated stats + per-run lists for t-test and delta
    dataset_order = []
    by_dataset = {}  # dataset_name -> {agg, run_data}
    for exp in exps:
        agg_path = os.path.join(exp, AGGREGATED_OUTPUT_JSON_PATH)
        if not os.path.exists(agg_path):
            print(f"No aggregated output found for {exp}, skipping")
            continue
        agg = load_json(agg_path)
        dataset_name = agg["experiment_name"].split("_")[0]
        dataset_order.append(dataset_name)

        run_data = {
            "train": {"student_acc": [], "distilled_acc": [], "student_time": [], "distilled_time": []},
            "test": {"student_acc": [], "distilled_acc": [], "student_time": [], "distilled_time": []},
        }
        for entry in sorted(os.scandir(exp), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            run_json = os.path.join(entry.path, OUTPUT_JSON_PATH)
            if not os.path.exists(run_json):
                continue
            run_output = load_json(run_json)
            a = run_output["analysis"]
            run_data["train"]["student_acc"].append(a["avg_acc_train_student"])
            run_data["train"]["distilled_acc"].append(a["avg_acc_train_distilled"])
            run_data["train"]["student_time"].append(a["avg_time_train_student"])
            run_data["train"]["distilled_time"].append(a["avg_time_train_distilled"])
            run_data["test"]["student_acc"].append(a["avg_acc_test_student"])
            run_data["test"]["distilled_acc"].append(a["avg_acc_test_distilled"])
            run_data["test"]["student_time"].append(a["avg_time_test_student"])
            run_data["test"]["distilled_time"].append(a["avg_time_test_distilled"])

        by_dataset[dataset_name] = {"agg": agg, "run_data": run_data}

    output_dir = os.path.join("assets", "experiment")
    os.makedirs(output_dir, exist_ok=True)

    for phase in ("train", "test"):
        keys = phase_keys[phase]
        (acc_t_avg, acc_t_std, acc_b_avg, acc_b_std, acc_s_avg, acc_s_std,
         time_t_avg, time_t_std, time_b_avg, time_b_std, time_s_avg, time_s_std) = keys

        rows = []
        for ds in dataset_order:
            data = by_dataset[ds]
            agg = data["agg"]["analysis"]
            rd = data["run_data"][phase]

            # Accuracy block
            acc_teacher = f"{agg[acc_t_avg]:.2f} $\\pm$ {agg[acc_t_std]:.2f}"
            acc_baseline = f"{agg[acc_b_avg]:.2f} $\\pm$ {agg[acc_b_std]:.2f}"
            _, p_acc = stats.ttest_rel(rd["distilled_acc"], rd["student_acc"], alternative="greater")
            stars_acc = _stars(p_acc)
            acc_student = f"{agg[acc_s_avg]:.2f} $\\pm$ {agg[acc_s_std]:.2f}"
            delta_acc = f'{((sum(rd["distilled_acc"]) - sum(rd["student_acc"])) / len(rd["student_acc"])):+.2f}{stars_acc}'

            # Time block
            time_teacher = f"{agg[time_t_avg]:.2f} $\\pm$ {agg[time_t_std]:.2f}"
            time_baseline = f"{agg[time_b_avg]:.2f} $\\pm$ {agg[time_b_std]:.2f}"
            _, p_time = stats.ttest_rel(rd["distilled_time"], rd["student_time"], alternative="two-sided")
            stars_time = _stars(p_time)
            time_student = f"{agg[time_s_avg]:.2f} $\\pm$ {agg[time_s_std]:.2f}"
            delta_time = f'{((sum(rd["distilled_time"]) - sum(rd["student_time"])) / len(rd["student_time"])):+.2f}{stars_time}'

            rows.append({
                "acc_teacher": acc_teacher,
                "acc_baseline": acc_baseline,
                "acc_student": acc_student,
                "delta_acc": delta_acc,
                "time_teacher": time_teacher,
                "time_baseline": time_baseline,
                "time_student": time_student,
                "delta_time": delta_time,
            })

        # Build LaTeX: columns = Metric | MNIST | KMNIST | EMNIST | IMDB
        col_fmt = "|l|" + "c|" * len(dataset_order)
        header = "Metric & " + " & ".join(dataset_order) + " \\\\"
        line = "\\hline"

        body = []
        body.append("$Acc_T$ & " + " & ".join(r["acc_teacher"] for r in rows) + " \\\\")
        body.append(line)
        body.append("$Acc_B$ & " + " & ".join(r["acc_baseline"] for r in rows) + " \\\\")
        body.append(line)
        body.append("$Acc_S$ (DKD) & " + " & ".join(r["acc_student"] for r in rows) + " \\\\")
        body.append(line)
        body.append("$\\Delta$ ($Acc_S-Acc_B$) & " + " & ".join(r["delta_acc"] for r in rows) + " \\\\")
        body.append("\\hline")
        body.append("\\hline")
        body.append("$\\mathcal{T}_T$ & " + " & ".join(r["time_teacher"] for r in rows) + " \\\\")
        body.append(line)
        body.append("$\\mathcal{T}_B$ & " + " & ".join(r["time_baseline"] for r in rows) + " \\\\")
        body.append(line)
        body.append("$\\mathcal{T}_S$ (DKD)& " + " & ".join(r["time_student"] for r in rows) + " \\\\")
        body.append(line)
        body.append("$\\Delta$ ($\\mathcal{T}_S-\\mathcal{T}_B$) & " + " & ".join(r["delta_time"] for r in rows) + " \\\\")

        caption = "Training Results" if phase == "train" else "Testing Results"
        label = "tab:combined-train-dkd" if phase == "train" else "tab:combined-test-dkd"
        latex = (
            "\\begin{table*}\n"
            f"\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{col_fmt}}}\n"
            f"{line}\n"
            f"{header}\n"
            f"{line}\n"
            + "\n".join(body) + "\n"
            f"{line}\n"
            "\\end{tabular}\n"
            "\\end{table*}\n"
        )
        filename = "combined_train_table.tex" if phase == "train" else "combined_test_table.tex"
        out_path = os.path.join(output_dir, filename)
        with open(out_path, "w") as f:
            f.write(latex)
        print(f"Saved {filename} to {out_path}")


def make_combined_graphs(exps: list[tuple[str, str]], output_dir: str):
    """
    Create combined bar graphs for multiple experiments showing accuracy and time comparisons.
    
    Args:
        exps: list of experiment directory paths
    """
    # Set up matplotlib configuration
    plt.rcParams['font.family'] = 'CMU Serif'
    plt.rcParams['font.serif'] = 'CMU Serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.size'] = 14
    
    # Colors for different models
    colors = {
        "distilled": "tab:blue",
        "Distilled w/ PCD": "tab:purple",
        "teacher": "tab:orange",
        "student": "tab:green"
    }
    
    # Process data for each experiment
    experiment_data = []
    for exp in exps:
        downsampled = False
        print(exp)
        exp_output = load_json(os.path.join(exp, OUTPUT_JSON_PATH))
        
        # Extract dataset name (e.g., "MNIST" from "MNIST_tC800_sC100...")
        dataset_name = exp_output["experiment_name"].split('_')[0]
        
        # Calculate means for accuracy and time
        data = {
            "name": dataset_name,
            "train": {
                "teacher": {
                    "acc": exp_output["analysis"]["avg_acc_train_teacher"],
                    "time": exp_output["analysis"]["avg_time_train_teacher"]
                },
                "student": {
                    "acc": exp_output["analysis"]["avg_acc_train_student"],
                    "time": exp_output["analysis"]["avg_time_train_student"]
                },
                "distilled": {
                    "acc": exp_output["analysis"]["avg_acc_train_distilled"],
                    "time": exp_output["analysis"]["avg_time_train_distilled"]
                }
            },
            "test": {
                "teacher": {
                    "acc": exp_output["analysis"]["avg_acc_test_teacher"],
                    "time": exp_output["analysis"]["avg_time_test_teacher"]
                },
                "student": {
                    "acc": exp_output["analysis"]["avg_acc_test_student"],
                    "time": exp_output["analysis"]["avg_time_test_student"]
                },
                "distilled": {
                    "acc": exp_output["analysis"]["avg_acc_test_distilled"],
                    "time": exp_output["analysis"]["avg_time_test_distilled"]
                }
            }
        }
        if "downsample_info" in exp_output:
            data["train"]["Distilled w/ PCD"] = {
                "acc": exp_output["analysis"]["avg_acc_train_distilled_ds"],
                "time": exp_output["analysis"]["avg_time_train_distilled_ds"]
            }
            data["test"]["Distilled w/ PCD"] = {
                "acc": exp_output["analysis"]["avg_acc_test_distilled_ds"],
                "time": exp_output["analysis"]["avg_time_test_distilled_ds"]
            }
            downsampled = True
        experiment_data.append(data)
    
    # Create plots for both train and test, and for both accuracy and time
    for phase in ["train", "test"]:
        for metric in ["acc", "time"]:
            plt.figure(figsize=(8,4.5), dpi=PLOT_DPI)
            
            # Calculate bar positions
            n_experiments = len(experiment_data)
            n_models = 3  # teacher, student, distilled
            bar_width = 0.25
            spacing = 0.5  # Space between experiment groups
            bar_spacing = 0.01  # 1px spacing between bars (in data coordinates)
            
            # Calculate positions for each bar
            positions = []
            for i in range(n_experiments):
                base_pos = i * (n_models * bar_width + spacing)
                positions.append({
                    "teacher": base_pos,
                    "student": base_pos + bar_width + bar_spacing,  # Add spacing after first bar
                    "distilled": base_pos + 2 * (bar_width + bar_spacing)  # Add spacing after second bar
                })
                if downsampled:
                    positions[-1]["Distilled w/ PCD"] = base_pos + 3 * (bar_width + bar_spacing)  # Add spacing after third bar
            
            # Plot bars

            models = ["teacher", "student", "distilled", "Distilled w/ PCD"]
            if not downsampled:
                models.remove("Distilled w/ PCD")
            for model in models:
                if metric == "time":
                    # Normalize time values relative to teacher for each experiment
                    means = []
                    for data in experiment_data:
                        teacher_time = data[phase]["teacher"]["time"]
                        model_time = data[phase][model]["time"]
                        means.append(model_time / teacher_time)  # Relative to teacher
                else:
                    means = [data[phase][model][metric] for data in experiment_data]
                pos = [pos[model] for pos in positions]
                
                bars = plt.bar(pos, means, bar_width, 
                             label=model.capitalize() if model != "Distilled w/ PCD" else "Distilled w/ PCD",
                             color=colors[model],
                             zorder=10)  # Set zorder to 10 to put bars above grid
                
            
            # Customize plot
            plt.grid(True, linestyle='dotted', which='major', axis='y', zorder=0)
            plt.grid(True, linestyle='dotted', which='minor', axis='y', alpha=0.2, zorder=0)
            plt.minorticks_on()
            
            plt.xlabel('Dataset')
            if metric == "acc":
                plt.ylabel(f'Average {phase.capitalize()} Accuracy (%)')
            else:
                plt.ylabel(f'Average {phase.capitalize()} Time (normalized)')
            
            # Set x-axis ticks and labels
            plt.xticks([pos["student"] for pos in positions], 
                      [data["name"] for data in experiment_data],
                      rotation=0)
            
            # Adjust y-axis range to focus on the relevant region
            if metric == "time":
                # For time, set y-axis to show relative speedup/slowdown
                #plt.ylim(0, 1.2)  # Show from 0x to 2x teacher's time
                # get current y-axis limits
                y_min, y_max = plt.ylim()
                # set y-axis limits to show from 0x to 2x teacher's time
                plt.ylim(0, 1.2 * y_max)
            else:
                # For accuracy, use the original range calculation
                y_min = min(min(data[phase][model][metric] for model in models) 
                           for data in experiment_data)
                y_max = max(max(data[phase][model][metric] for model in models) 
                           for data in experiment_data)
                y_range = y_max - y_min
                plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
            
            # Add a horizontal line at y=1 for time plots to show teacher baseline
            if metric == "time":
                plt.axhline(y=1, color='black', linestyle='--', alpha=0.3, zorder=5)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Add legend
            # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3).set_zorder(100)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=2).set_zorder(100)

            # Save plot
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"combined_{phase}_{metric}.png"), bbox_inches="tight")
            plt.close()

def make_combined_graphs_aggregate(exps: list[tuple[str, str]], output_dir: str):
    """
    Create combined bar graphs for multiple experiments showing accuracy and time comparisons.
    
    Args:
        exps: list of experiment directory paths
    """
    # Set up matplotlib configuration
    plt.rcParams['font.family'] = 'CMU Serif'
    plt.rcParams['font.serif'] = 'CMU Serif'
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['font.size'] = 14
    
    # Colors for different models
    colors = {
        "distilled": "tab:blue",
        "Distilled w/ PCD": "tab:purple",
        "teacher": "tab:orange",
        "student": "tab:green"
    }
    
    # Process data for each experiment
    experiment_data = []
    for exp in exps:
        print(exp)
        if not os.path.exists(os.path.join(exp, AGGREGATED_OUTPUT_JSON_PATH)):
            print(f"No aggregated output found for {exp}")
            continue
        exp_output = load_json(os.path.join(exp, AGGREGATED_OUTPUT_JSON_PATH))

        # Extract dataset name (e.g., "MNIST" from "MNIST_tC800_sC100...")
        dataset_name = exp_output["experiment_name"].split('_')[0]

        # Calculate means for accuracy and time
        data = {
            "name": dataset_name,
            "K": exp_output["num_experiments"],
            "train": {
                "teacher": {
                    "avg_acc": exp_output["analysis"]["avg_acc_train_teacher"],
                    "std_acc": exp_output["analysis"]["std_acc_train_teacher"],
                    "avg_time": exp_output["analysis"]["avg_time_train_teacher_normalized"],
                    "std_time": exp_output["analysis"]["std_time_train_teacher_normalized"],
                },
                "student": {
                    "avg_acc": exp_output["analysis"]["avg_acc_train_student"],
                    "std_acc": exp_output["analysis"]["std_acc_train_student"],
                    "avg_time": exp_output["analysis"]["avg_time_train_student_normalized"],
                    "std_time": exp_output["analysis"]["std_time_train_student_normalized"],
                },
                "distilled": {
                    "avg_acc": exp_output["analysis"]["avg_acc_train_distilled"],
                    "std_acc": exp_output["analysis"]["std_acc_train_distilled"],
                    "avg_time": exp_output["analysis"]["avg_time_train_distilled_normalized"],
                    "std_time": exp_output["analysis"]["std_time_train_distilled_normalized"],
                }
            },
            "test": {
                "teacher": {
                    "avg_acc": exp_output["analysis"]["avg_acc_test_teacher"],
                    "std_acc": exp_output["analysis"]["std_acc_test_teacher"],
                    "avg_time": exp_output["analysis"]["avg_time_test_teacher_normalized"],
                    "std_time": exp_output["analysis"]["std_time_test_teacher_normalized"],
                },
                "student": {
                    "avg_acc": exp_output["analysis"]["avg_acc_test_student"],
                    "std_acc": exp_output["analysis"]["std_acc_test_student"],
                    "avg_time": exp_output["analysis"]["avg_time_test_student_normalized"],
                    "std_time": exp_output["analysis"]["std_time_test_student_normalized"],
                },
                "distilled": {
                    "avg_acc": exp_output["analysis"]["avg_acc_test_distilled"],
                    "std_acc": exp_output["analysis"]["std_acc_test_distilled"],
                    "avg_time": exp_output["analysis"]["avg_time_test_distilled_normalized"],
                    "std_time": exp_output["analysis"]["std_time_test_distilled_normalized"],
                }
            }
        }
        experiment_data.append(data)

    # Create plots for both train and test, and for both accuracy and time
    for phase in ["train", "test"]:
        for metric in ["acc", "time"]:
            plt.figure(figsize=(8,4.5), dpi=PLOT_DPI)
            
            # Calculate bar positions
            n_experiments = len(experiment_data)
            n_models = 3  # teacher, student, distilled
            bar_width = 0.25
            spacing = 0.5  # Space between experiment groups
            bar_spacing = 0.01  # 1px spacing between bars (in data coordinates)
            
            # Calculate positions for each bar
            positions = []
            for i in range(n_experiments):
                base_pos = i * (n_models * bar_width + spacing)
                positions.append({
                    "teacher": base_pos,
                    "student": base_pos + bar_width + bar_spacing,  # Add spacing after first bar
                    "distilled": base_pos + 2 * (bar_width + bar_spacing)  # Add spacing after second bar
                })
            
            # Plot bars

            display_names = {"teacher": "Teacher", "student": "Baseline", "distilled": "Student"}
            models = ["teacher", "student", "distilled"]
            for model in models:
                means = [data[phase][model]["avg_"+metric] for data in experiment_data]
                stds = [data[phase][model]["std_"+metric] for data in experiment_data]
                pos = [pos[model] for pos in positions]

                bars = plt.bar(pos, means, bar_width,
                             label=display_names[model],
                             color=colors[model],
                             zorder=10)  # Set zorder to 10 to put bars above grid
                
                # Add error bars
                error_bars = plt.errorbar(pos, means, yerr=stds, fmt='none', ecolor='black', capsize=5, zorder=10)

            # Customize plot
            plt.grid(True, linestyle='dotted', which='major', axis='y', zorder=0)
            plt.grid(True, linestyle='dotted', which='minor', axis='y', alpha=0.2, zorder=0)
            plt.minorticks_on()

            plt.xlabel('Dataset')
            if metric == "acc":
                plt.ylabel(f'Average {phase.capitalize()} Accuracy (%)')
            else:
                plt.ylabel(f'Average {phase.capitalize()} Time (normalized)')

            # Set x-axis ticks and labels
            plt.xticks([pos["student"] for pos in positions],
                      [f"{data['name']}\n$K={data['K']}$" for data in experiment_data],
                      rotation=0)
            
            # Adjust y-axis range to focus on the relevant region
            if metric == "time":
                # For time, set y-axis to show relative speedup/slowdown
                #plt.ylim(0, 1.2)  # Show from 0x to 2x teacher's time
                # get current y-axis limits
                y_min, y_max = plt.ylim()
                plt.ylim(0, 1.1 * y_max)
            else:
                # For accuracy, use the original range calculation
                y_min = min(min(data[phase][model]["avg_"+metric] for model in models) 
                           for data in experiment_data)
                y_max = max(max(data[phase][model]["avg_"+metric] for model in models) 
                           for data in experiment_data)
                y_range = y_max - y_min
                plt.ylim(y_min - 0.1 * y_range, y_max + 0.2 * y_range)
            
            # Add a horizontal line at y=1 for time plots to show teacher baseline
            if metric == "time":
                plt.axhline(y=1, color='black', linestyle='--', alpha=0.3, zorder=5)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Add legend
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3).set_zorder(100)
            

            # Save plot
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f"combined_{phase}_{metric}.png"), bbox_inches="tight")
            plt.close()

def j(*args):
    return os.path.join(*args)
    
if __name__ == "__main__":
    # Paper 2 experiments
    paper2_exps = [
        j("combined_results", "distribution", "MNIST_tC1000_sC100_tT10_sT10_ts4.0_ss4.0_te120_se240_temp3.0_a0.5_z0.3"),
        j("combined_results", "distribution", "KMNIST_tC2000_sC200_tT100_sT100_ts8.2_ss8.2_te120_se240_temp4.0_a0.5_z0.3"),
        j("combined_results", "distribution", "EMNIST_tC1000_sC100_tT100_sT100_ts4.0_ss4.0_te120_se240_temp4.0_a0.5_z0.2"),
        j("combined_results", "distribution", "IMDB_tC8000_sC4000_tT6000_sT6000_ts7.0_ss7.0_te30_se60_temp3.0_a0.5_z0.2"),
    ]

    # Generate tables
    # make_paper_1_tables(paper1_exps)
    paper2_aggregate_exps = [
        j("results", "aggregate_distribution", "MNIST"),
        j("results", "aggregate_distribution", "KMNIST"),
        j("results", "aggregate_distribution", "EMNIST"),
        j("results", "aggregate_distribution", "IMDB"),
    ]       
    make_experiment_tables_aggregate(paper2_aggregate_exps)
    make_formatted_tables(paper2_aggregate_exps)

    make_combined_graphs_aggregate(paper2_aggregate_exps, j("assets", "experiment"))

    # Generate combined graphs
    #print("tables done")
    # make_combined_graphs(paper1_exps, "assets/paper_1")
    # print("paper 1 graphs done")
    # make_combined_graphs(paper2_exps, "experiment")
    # print("paper 2 graphs done")

"""
    for folder in os.listdir("combined_results/distribution"):
        output = load_json(os.path.join("combined_results", "distribution", folder, OUTPUT_JSON_PATH))
        print(output["experiment_name"])
        results = pd.DataFrame(output["results"])
        output["analysis"]["avg_acc_test_distilled"] = results[ACC_TEST_DISTILLED].mean()
        output["analysis"]["std_acc_test_distilled"] = results[ACC_TEST_DISTILLED].std()
        output["analysis"]["avg_acc_train_distilled"] = results[ACC_TRAIN_DISTILLED].mean()
        output["analysis"]["std_acc_train_distilled"] = results[ACC_TRAIN_DISTILLED].std()
        save_json(output, os.path.join("combined_results", "distribution", folder, OUTPUT_JSON_PATH))
        plot_results(output, os.path.join("combined_results", "distribution", folder))
"""