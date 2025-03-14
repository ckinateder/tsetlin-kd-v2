# Use this file to fix plots and calculations after the experiments are done
import csv
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from util import load_pkl, load_json, save_json
from __init__ import  OUTPUT_JSON_PATH, PLOT_FIGSIZE, PLOT_DPI, ACC_TEST_DISTILLED, ACC_TRAIN_DISTILLED
import os 
from __init__ import  OUTPUT_JSON_PATH, PLOT_FIGSIZE, PLOT_DPI
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
            "$\\mathcal{T}'^{down}_D$": f'{exp_output["analysis"]["avg_time_train_distilled_ds"]:.2f}'
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
            "$\\mathcal{T}'^{down}_D$": f'{exp_output["analysis"]["avg_time_test_distilled_ds"]:.2f}'
        }
        test_table = test_table._append(new_row, ignore_index=True)

    # Define table configurations
    table_configs = [
        {
            "name": "hyperparam_table",
            "file_name": "hyperparam_table",
            "caption": "Hyperparameters",
            "label": "tab:hyperparams",
            "column_format": "l"*len(hyperparam_table.columns)
        },
        {
            "name": "dataset_size_table",
            "file_name": "dataset_size_table",
            "caption": "Dataset Size",
            "label": "tab:dataset-size",
            "column_format": "l"*len(dataset_size_table.columns)
        },
        {
            "name": "train_table",
            "file_name": "train_table",
            "caption": "Train Table",
            "label": "tab:train-table",
            "column_format": "l"+("c"*(len(train_table.columns)-1))
        },
        {
            "name": "test_table",
            "file_name": "test_table",
            "caption": "Test Table",
            "label": "tab:test-table",
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
        
        
        


def make_paper_2_tables(exps: list[tuple[str, str]]):
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
            "caption": "Hyperparameters",
            "label": "tab:hyperparams",
            "column_format": "l"*len(hyperparam_table.columns)
        },
        {
            "name": "dataset_size_table",
            "file_name": "dataset_size_table",
            "caption": "Dataset Size",
            "label": "tab:dataset-size",
            "column_format": "l"*len(dataset_size_table.columns)
        },
        {
            "name": "train_table",
            "file_name": "train_table",
            "caption": "Train Table",
            "label": "tab:train-table",
            "column_format": "l"+("c"*(len(train_table.columns)-1))
        },
        {
            "name": "test_table",
            "file_name": "test_table",
            "caption": "Test Table",
            "label": "tab:test-table",
            "column_format": "l"+("c"*(len(test_table.columns)-1))
        }
    ]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join("assets", "paper_2")
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
        
def j(*args):
    return os.path.join(*args)
if __name__ == "__main__":

    make_paper_1_tables([
        j("combined_results", "clause", "MNIST_tC800_sC100_tT10_sT10_ts7.0_ss7.0_te120_se240_ds0.15"),
        j("combined_results", "clause", "KMNIST_tC400_sC100_tT100_sT100_ts5_ss5_te120_se240_ds0.22"),
        j("combined_results", "clause", "IMDB_tC10000_sC2000_tT6000_sT6000_ts5.0_ss5.0_te30_se90_ds0.15")
    ])

    make_paper_2_tables([
        j("combined_results", "distribution", "EMNIST_tC1000_sC100_tT100_sT100_ts4.0_ss4.0_te60_se120_temp4.0_a0.5_z0.2"),
        j("combined_results", "distribution", "MNIST_tC1000_sC100_tT10_sT10_ts4.0_ss4.0_te120_se240_temp3.0_a0.5_z0.3"),
        j("combined_results", "distribution", "KMNIST_tC2000_sC200_tT100_sT100_ts8.2_ss8.2_te120_se240_temp4.0_a0.5_z0.3"),
        j("combined_results", "distribution", "IMDB_tC8000_sC4000_tT6000_sT6000_ts7.0_ss7.0_te30_se60_temp3.0_a0.5_z0.2"),
    ])

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