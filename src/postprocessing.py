# Use this file to fix plots and calculations after the experiments are done
import csv
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from util import load_pkl, load_json, save_json
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
    # test accuracy table
    test_acc_table = pd.DataFrame(columns=["Dataset", "$Acc_T$", "$Acc_S$", "$Acc_{CKD}$", "$Acc_{CKD-PCD}$"], index=[])

    # information table
    info_table = pd.DataFrame(columns=["Dataset", "$I_T$", "$I_S$", "$I_{CKD}$", "$I_{CKD-PCD}$"], index=[])

    # training time table
    # cols = ["dataset", "teacher", "student", "CKD", "CKD-PCD"]
    training_time_table = pd.DataFrame(columns=["Dataset", "$\mathcal{T}_T$", "$\mathcal{T}_S$", "$\mathcal{T}_{CKD}$", "$\mathcal{T}_{CKD-PCD}$"], index=[])
    
    # best r value for PCD table
    pcd_r_table = pd.DataFrame(columns=["Dataset", "$r$"], index=[])
    
    for exp in exps:
        print(exp)
        ckd_exp, ckd_pcd_exp = exp
        ckd_exp_output = load_json(os.path.join(ckd_exp, OUTPUT_JSON_PATH))
        ckd_pcd_exp_output = load_json(os.path.join(ckd_pcd_exp, OUTPUT_JSON_PATH))

        # get row name
        rowname = ckd_exp_output["experiment_name"].split("-")[0]

        # get test accuracy
        print(test_acc_table)
        new_row = {
            "Dataset": rowname,
            "$Acc_T$": f'{ckd_exp_output["analysis"]["avg_acc_test_teacher"]:.2f} $\pm$ {ckd_exp_output["analysis"]["std_acc_test_teacher"]:.2f}',
            "$Acc_S$": f'{ckd_exp_output["analysis"]["avg_acc_test_student"]:.2f} $\pm$ {ckd_exp_output["analysis"]["std_acc_test_student"]:.2f}',
            "$Acc_{CKD}$": f'{ckd_exp_output["analysis"]["avg_acc_test_distilled"]:.2f} $\pm$ {ckd_exp_output["analysis"]["std_acc_test_distilled"]:.2f}',
            "$Acc_{CKD-PCD}$": f'{ckd_pcd_exp_output["analysis"]["avg_acc_test_distilled"]:.2f} $\pm$ {ckd_pcd_exp_output["analysis"]["std_acc_test_distilled"]:.2f}'
        }
        test_acc_table = test_acc_table._append(new_row, ignore_index=True)

        # get training time
        new_row = {
            "Dataset": rowname,
            "$\mathcal{T}_T$": f'{ckd_exp_output["analysis"]["avg_time_train_teacher"]:.2f}',
            "$\mathcal{T}_S$": f'{ckd_exp_output["analysis"]["avg_time_train_student"]:.2f}',
            "$\mathcal{T}_{CKD}$": f'{ckd_exp_output["analysis"]["avg_time_train_distilled"]:.2f}',
            "$\mathcal{T}_{CKD-PCD}$": f'{ckd_pcd_exp_output["analysis"]["avg_time_train_distilled"]:.2f}'
        }
        training_time_table = training_time_table._append(new_row, ignore_index=True)

        # get information in scientific notation, 3 decimal places
        new_row = {
            "Dataset": rowname,
            "$I_T$": f'{ckd_exp_output["mutual_information"]["I_teacher"]:.3e}',
            "$I_S$": f'{ckd_exp_output["mutual_information"]["I_student"]:.3e}',
            "$I_{CKD}$": f'{ckd_exp_output["mutual_information"]["I_distilled"]:.3e}',
            "$I_{CKD-PCD}$": f'{ckd_pcd_exp_output["mutual_information"]["I_distilled"]:.3e}'
        }
        info_table = info_table._append(new_row, ignore_index=True)

        # get best r value for PCD
        new_row = {
            "Dataset": rowname,
            "r": ckd_pcd_exp_output["params"]["downsample"]
        }
        pcd_r_table = pcd_r_table._append(new_row, ignore_index=True)

    # save tables
    test_acc_table.to_csv(os.path.join("assets", "paper_1", "test_acc_table.csv"), index=False)
    training_time_table.to_csv(os.path.join("assets", "paper_1", "training_time_table.csv"), index=False)
    pcd_r_table.to_csv(os.path.join("assets", "paper_1", "pcd_r_table.csv"), index=False)
    info_table.to_csv(os.path.join("assets", "paper_1", "info_table.csv"), index=False)

    # save tables in latex format
    # Export to LaTeX with specific formatting
    column_format = "lllll"
    latex_table = test_acc_table.to_latex(index=False, escape=False, column_format=column_format, caption="Average Test Accuracy (\\%)", label="tab:test-acc")

    with open(os.path.join("assets", "paper_1", "test_acc_table.tex"), "w") as f:
        f.write(latex_table)

    latex_table = training_time_table.to_latex(index=False, escape=False, column_format=column_format, caption="Average Training Time (s)", label="tab:training-time")
    with open(os.path.join("assets", "paper_1", "training_time_table.tex"), "w") as f:
        f.write(latex_table)

    latex_table = info_table.to_latex(index=False, escape=False, column_format=column_format, caption="Information (nats)", label="tab:info")
    with open(os.path.join("assets", "paper_1", "info_table.tex"), "w") as f:
        f.write(latex_table)

    latex_table = pcd_r_table.to_latex(index=False, escape=False, column_format="ll", caption="Best $r$ value for $PCD$", label="tab:best-pcd")
    with open(os.path.join("assets", "paper_1", "pcd_r_table.tex"), "w") as f:
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

    """
    make_paper_1_tables([
       (j("combined_results", "ckd", "IMDB-Downsample-Take-2", "ds_tnc10000_snc2000_T6000_s4.0_te30_se90_downsample0"),
        j("combined_results", "ckd", "IMDB-Downsample-Take-2", "ds_tnc10000_snc2000_T6000_s4.0_te30_se90_downsample0.25")),
        (j("combined_results", "ckd", "KMNIST-Downsample-Take-3", "ds_tnc400_snc100_T100_s5_te50_se100_downsample0"),
        j("combined_results", "ckd", "KMNIST-Downsample-Take-3", "ds_tnc400_snc100_T100_s5_te50_se100_downsample0.22")),
        (j("combined_results", "ckd", "MNIST-Downsample-Take-5", "ds_tnc800_snc100_T10_s7.0_te50_se100_downsample0"),
        j("combined_results", "ckd", "MNIST-Downsample-Take-5", "ds_tnc800_snc100_T10_s7.0_te50_se100_downsample0.15")),
        (j("combined_results", "ckd", "MNIST3D-Downsample-Take-2", "ds_tnc1500_snc250_T100_s3.0_te20_se70_downsample0"),
        j("combined_results", "ckd", "MNIST3D-Downsample-Take-2", "ds_tnc1500_snc250_T100_s3.0_te20_se70_downsample0.15")),
    ])
    """
    print(os.listdir("combined_results"))
    
    make_paper_2_tables([
        j("distribution", "combined_results", "EMNIST_tC1000_sC100_tT100_sT100_ts4.0_ss4.0_te60_se120_temp4.0_a0.5_z0.2"),
        j("distribution", "combined_results", "MNIST_tC1000_sC100_tT10_sT10_ts4.0_ss4.0_te60_se120_temp3.0_a0.5_z0.3"),
        j("distribution", "combined_results", "KMNIST_tC2000_sC200_tT100_sT100_ts8.2_ss8.2_te60_se120_temp4.0_a0.5_z0.3"),
        j("distribution", "combined_results", "IMDB_tC8000_sC4000_tT6000_sT6000_ts7.0_ss7.0_te30_se60_temp3.0_a0.5_z0.2"),
    ])
