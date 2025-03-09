# Use this file to fix plots and calculations after the experiments are done
import csv
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from util import load_pkl, load_json, save_json
import os 
from stats import calculate_information
from __init__ import  OUTPUT_JSON_PATH, PLOT_FIGSIZE, PLOT_DPI
def iterate_over_file_in_folder(folder="experiments", file_extension=".json"):
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    yield data, file_path

def combine_into_cols():
    # make a single csv with all the results. prepend the experiment name to each result column
    output = pd.DataFrame()
    for experiment, file_path in iterate_over_file_in_folder():
        results = experiment["results"]
        params = experiment["params"]
        experiment_id = experiment["id"]
        experiment_name = experiment["experiment_name"]
        # add columns to the output dataframe
        intermediate = pd.DataFrame(results)
        intermediate.columns = [f"{experiment_name}_{col}" for col in intermediate.columns]
        output = pd.concat([output, intermediate], axis=1)

    # sort columns alphabetically
    output = output.reindex(sorted(output.columns), axis=1)

    output.to_csv(os.path.join("experiments", "one_table.csv"), index=False)

        

def process_mi():
    # make a csv wiht params at the top and experiment id as the row
    mis = pd.DataFrame(columns=["experiment_name", "id", "teacher_num_clauses", "student_num_clauses", "T", "s", "teacher_epochs", "student_epochs", "weighted_clauses","number_of_state_bits", "mutual_info_sd" ,"mutual_info_td"])
    for experiment, file_path in iterate_over_file_in_folder():
        print(file_path)
        params = experiment["params"]
        experiment_id = experiment["id"]

        mis = mis._append({
            "experiment_name": experiment["experiment_name"],
            "id": experiment_id,
            "teacher_num_clauses": params["teacher_num_clauses"],
            "student_num_clauses": params["student_num_clauses"],
            "T": params["T"],
            "s": params["s"],
            "teacher_epochs": params["teacher_epochs"],
            "student_epochs": params["student_epochs"],
            "weighted_clauses": params["weighted_clauses"],
            "number_of_state_bits": params["number_of_state_bits"],
            "mutual_info_sd": experiment["mutual_information"]["sklearn_student"],
            "mutual_info_td": experiment["mutual_information"]["sklearn_teacher"]
        }, ignore_index=True)
    # sort alph
    mis = mis.sort_values(by="experiment_name")
    mis.to_csv(os.path.join("experiments", "mutual_info.csv"), index=False)
    return mis
                    
def add_analyses():
    """
    
    # compute averages for accuracy
    avg_acc_test_teacher = results["acc_test_teacher"].mean()
    std_acc_test_teacher = results["acc_test_teacher"].std()
    avg_acc_test_student = results["acc_test_student"].mean()
    std_acc_test_student = results["acc_test_student"].std()
    avg_acc_test_distilled = results["acc_test_distilled"].mean()
    std_acc_test_distilled = results["acc_test_distilled"].std()

    # compute sum of training times
    sum_time_train_teacher = results["time_train_teacher"].sum()
    sum_time_train_student = results["time_train_student"].sum()
    sum_time_train_distilled = results["time_train_distilled"].sum()

    """
    import numpy as np
    for experiment, file_path in iterate_over_file_in_folder():
        results = experiment["results"]
        results_pd = pd.DataFrame(results)
        experiment["analysis"] = { 
            "avg_acc_test_teacher": results_pd["acc_test_teacher"].mean(),
            "std_acc_test_teacher": results_pd["acc_test_teacher"].std(),
            "avg_acc_test_student": results_pd["acc_test_student"].mean(),
            "std_acc_test_student": results_pd["acc_test_student"].std(),
            "avg_acc_test_distilled": results_pd["acc_test_distilled"].mean(),
            "std_acc_test_distilled": results_pd["acc_test_distilled"].std(),
            "final_acc_test_distilled": results_pd["acc_test_distilled"].iloc[-1],
            "sum_time_train_teacher": results_pd["time_train_teacher"].sum(),
            "sum_time_train_student": results_pd["time_train_student"].sum(),
            "sum_time_train_distilled": results_pd["time_train_distilled"].sum()
        }
        with open(file_path, 'w') as f:
            json.dump(experiment, f, indent=4)

def make_charts():
    for experiment, file_path in iterate_over_file_in_folder():
        results = experiment["results"]
        results_pd = pd.DataFrame(results)
        params = experiment["params"]
        del params["number_of_state_bits"]
        del params["weighted_clauses"]

        # make a chart of the accuracies
        plt.figure(figsize=(8,6))
        plt.plot(results_pd["acc_test_distilled"], label="Distilled")
        plt.plot(results_pd["acc_test_teacher"], label="Teacher", alpha=0.5)
        plt.plot(results_pd["acc_test_student"], label="Student", alpha=0.5)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(experiment["experiment_name"])
        plt.xticks(range(0, len(results_pd), 5))
        plt.legend(loc="upper left")
        plt.grid(linestyle='dotted')
        # add text of parameters
        params_text = "\n".join([f"{k}: {v}" for k, v in params.items()])
        plt.gcf().text(0.68, 0.14, params_text, fontsize=8, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=1))
        plt.savefig(file_path.replace(".json", ".png"))
        plt.close()

def make_accuracy_table():
    """
    experiment, avg_acc_test_teacher, std_acc_test_teacher, avg_acc_test_student, std_acc_test_student, avg_acc_test_distilled, std_acc_test_distilled, final_acc_test_distilled, mutual_info_sd, mutual_info_td
    """
    table = pd.DataFrame(columns=["experiment", "avg_acc_test_teacher", "std_acc_test_teacher", "avg_acc_test_student", "std_acc_test_student", "avg_acc_test_distilled", "std_acc_test_distilled", "final_acc_test_distilled", "mutual_info_sd", "mutual_info_td"])
    for experiment, file_path in iterate_over_file_in_folder():
        results = experiment["results"]
        results_pd = pd.DataFrame(results)
        params = experiment["params"]
        new_row = pd.DataFrame([{
            "experiment": experiment["experiment_name"],
            "avg_acc_test_teacher": round(results_pd["acc_test_teacher"].mean(), 3),
            "std_acc_test_teacher": round(results_pd["acc_test_teacher"].std(), 3),
            "avg_acc_test_student": round(results_pd["acc_test_student"].mean(), 3),
            "std_acc_test_student": round(results_pd["acc_test_student"].std(), 3),
            "avg_acc_test_distilled": round(results_pd["acc_test_distilled"].mean(), 3),
            "std_acc_test_distilled": round(results_pd["acc_test_distilled"].std(), 3),
            "final_acc_test_distilled": round(results_pd["acc_test_distilled"].iloc[-1], 3),
            "mutual_info_sd": round(experiment["mutual_information"]["sklearn_student"], 3),
            "mutual_info_td": round(experiment["mutual_information"]["sklearn_teacher"], 3)
        }])
        table = pd.concat([table, new_row], ignore_index=True)
        
    table = table.sort_values(by="experiment")
    table.to_csv(os.path.join("experiments", "accuracy_table.csv"), index=False)

def make_condensed_accuracy_table():
    """
    experiment, acc_teacher +/- std_teacher, acc_student +/- std_student, acc_distilled +/- std_distilled, mutual_info_sd, mutual_info_td
    """
    table = pd.DataFrame(columns=["experiment", "acc_teacher", "acc_student", "acc_distilled", "mutual_info_sd", "mutual_info_td"])
    for experiment, file_path in iterate_over_file_in_folder():
        results = experiment["results"]
        results_pd = pd.DataFrame(results)
        params = experiment["params"]
        
        acc_teacher = f"{round(results_pd['acc_test_teacher'].mean(), 3)} +/- {round(results_pd['acc_test_teacher'].std(), 3)}"
        acc_student = f"{round(results_pd['acc_test_student'].mean(), 3)} +/- {round(results_pd['acc_test_student'].std(), 3)}"
        acc_distilled = f"{round(results_pd['acc_test_distilled'].mean(), 3)} +/- {round(results_pd['acc_test_distilled'].std(), 3)}"
        mutual_info_sd = f"{round(experiment['mutual_information']['sklearn_student'], 3)}"
        mutual_info_td = f"{round(experiment['mutual_information']['sklearn_teacher'], 3)}"

        new_row = pd.DataFrame([{
            "experiment": experiment["experiment_name"],
            "acc_teacher": acc_teacher,
            "acc_student": acc_student,
            "acc_distilled": acc_distilled,
            "mutual_info_sd": mutual_info_sd,
            "mutual_info_td": mutual_info_td
        }])
        table = pd.concat([table, new_row], ignore_index=True)
        
    table = table.sort_values(by="experiment")
    table.to_csv(os.path.join("experiments", "accuracy_table.csv"), index=False)
        
def fix_mi_calculations(top_folderpath):
    for exp_path in os.listdir(top_folderpath):
        if not os.path.isdir(os.path.join(top_folderpath, exp_path)):
            continue

        # load output
        print(f"Fixing MI for {exp_path}")
        output = load_json(os.path.join(top_folderpath, exp_path, OUTPUT_JSON_PATH))
        # compute the mutual information
        I_student = calculate_information(output["helpful_for_calculations"]["L_student"], output["helpful_for_calculations"]["C_student"])
        I_teacher = calculate_information(output["helpful_for_calculations"]["L_teacher"], output["helpful_for_calculations"]["C_teacher"])
        I_distilled = calculate_information(output["helpful_for_calculations"]["L_distilled"], output["helpful_for_calculations"]["C_distilled"])

        #print(f"I_student: {I_student}")
        #print(f"I_teacher: {I_teacher}")
        #print(f"I_distilled: {I_distilled}")

        # update output
        output["mutual_information"] = {
            "I_student": I_student,
            "I_teacher": I_teacher,
            "I_distilled": I_distilled
        }
        save_json(output, os.path.join(top_folderpath, exp_path, OUTPUT_JSON_PATH))
        print(f"Saved to {os.path.join(top_folderpath, exp_path, OUTPUT_JSON_PATH)}")

def fix_mi_plots(top_folderpath):
    mis = {}
    for exp_path in os.listdir(top_folderpath):
        if not os.path.isdir(os.path.join(top_folderpath, exp_path)):
            continue

        output = load_json(os.path.join(top_folderpath, exp_path, OUTPUT_JSON_PATH))
        downsample = output["params"]["downsample"]
        mis[downsample] = output["mutual_information"]

    downsamples = np.array(list(mis.keys()))
    downsamples.sort()
    horiz_alpha = 0.8
    marker_size = 3
    x_ticks = np.arange(0, downsamples.max()+0.05, 0.05)
    all_i_true_distilled = np.array([mis[downsample]["I_distilled"] for downsample in downsamples])
    baseline_i_teacher = mis[0]["I_teacher"]
    baseline_i_student = mis[0]["I_student"]
    
    plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    plt.axhline(y=baseline_i_teacher, linestyle=':', color="orange", alpha=horiz_alpha, label="Teacher")
    plt.axhline(y=baseline_i_student, linestyle=':', color="green", alpha=horiz_alpha, label="Student")
    plt.plot(downsamples, all_i_true_distilled, marker='o', markersize=marker_size, label="Distilled")
    plt.xlabel("Downsample Rate")
    plt.ylabel("Information (nats)")
    plt.legend(loc="upper right")
    plt.xticks(x_ticks)
    plt.grid(linestyle='dotted')
    plt.savefig(os.path.join(top_folderpath, "downsample_results_information.png"))
    plt.close()

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
    # test accuracy table
    test_acc_table = pd.DataFrame(columns=["Dataset", "$Acc_T$", "$Acc_S$", "$Acc_D$"], index=[])

    # train accuracy table
    train_acc_table = pd.DataFrame(columns=["Dataset", "$Acc_T$", "$Acc_S$", "$Acc_D$"], index=[])

    # training time table
    training_time_table = pd.DataFrame(columns=["Dataset", "$\mathcal{T}_T$", "$\mathcal{T}_S$", "$\mathcal{T}_D$"], index=[])

    # test time table
    test_time_table = pd.DataFrame(columns=["Dataset", "$t_T$", "$t_S$", "$t_D$"], index=[])
    
    # hyperparameter table
    hyperparam_table = pd.DataFrame(columns=["Dataset", "$|C_T|$", "$|C_S|$", "$T_T$", "$T_S$", "$s_T$", "$s_S$", "$\\tau$", "$\\alpha$", "$z$", "$E_T$", "$E_S$"], index=[])

    # dataset size table
    dataset_size_table = pd.DataFrame(columns=["Dataset", "$|X_{train}|$", "$|X_{test}|$", "$|L|$", "$\\zeta$"], index=[])

    # combined train table with time
    train_table = pd.DataFrame(columns=["Dataset", "$Acc_T$", "$Acc_S$", "$Acc_D$", "$\mathcal{T}_T$", "$\mathcal{T}_S$", "$\mathcal{T}_D$"], index=[])
    test_table = pd.DataFrame(columns=["Dataset", "$Acc_T$", "$Acc_S$", "$Acc_D$", "$t_T$", "$t_S$", "$t_D$"], index=[])
    for exp in exps:
        print(exp)
        exp_output = load_json(os.path.join(exp, OUTPUT_JSON_PATH))

        # get row name
        rowname = exp_output["experiment_name"]

        # get test accuracy
        new_row = {
            "Dataset": rowname,
            "$Acc_T$": f'{exp_output["analysis"]["avg_acc_test_teacher"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_test_teacher"]:.2f}',
            "$Acc_S$": f'{exp_output["analysis"]["avg_acc_test_student"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_test_student"]:.2f}',
            "$Acc_D$": f'{exp_output["analysis"]["avg_acc_test_distilled"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_test_distilled"]:.2f}'
        }
        test_acc_table = test_acc_table._append(new_row, ignore_index=True)

        # get train accuracy
        new_row = {
            "Dataset": rowname,
            "$Acc_T$": f'{exp_output["analysis"]["avg_acc_train_teacher"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_train_teacher"]:.2f}',
            "$Acc_S$": f'{exp_output["analysis"]["avg_acc_train_student"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_train_student"]:.2f}',
            "$Acc_D$": f'{exp_output["analysis"]["avg_acc_train_distilled"]:.2f} $\pm$ {exp_output["analysis"]["std_acc_train_distilled"]:.2f}'
        }
        train_acc_table = train_acc_table._append(new_row, ignore_index=True)

        # get training time
        new_row = {
            "Dataset": rowname,
            "$\mathcal{T}_T$": f'{exp_output["analysis"]["avg_time_train_teacher"]:.2f}',
            "$\mathcal{T}_S$": f'{exp_output["analysis"]["avg_time_train_student"]:.2f}',
            "$\mathcal{T}_D$": f'{exp_output["analysis"]["avg_time_train_distilled"]:.2f}'
        }
        training_time_table = training_time_table._append(new_row, ignore_index=True)   

        # get test time      
        new_row = {
            "Dataset": rowname,
            "$t_T$": f'{exp_output["analysis"]["avg_time_test_teacher"]:.2f}',
            "$t_S$": f'{exp_output["analysis"]["avg_time_test_student"]:.2f}',
            "$t_D$": f'{exp_output["analysis"]["avg_time_test_distilled"]:.2f}'
        }
        test_time_table = test_time_table._append(new_row, ignore_index=True)

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
            "$Acc_T$": f'{exp_output["analysis"]["avg_acc_train_teacher"]:.2f}',
            "$Acc_S$": f'{exp_output["analysis"]["avg_acc_train_student"]:.2f}',
            "$Acc_D$": f'{exp_output["analysis"]["avg_acc_train_distilled"]:.2f}',
            "$\mathcal{T}_T$": f'{exp_output["analysis"]["avg_time_train_teacher"]:.2f}',
            "$\mathcal{T}_S$": f'{exp_output["analysis"]["avg_time_train_student"]:.2f}',
            "$\mathcal{T}_D$": f'{exp_output["analysis"]["avg_time_train_distilled"]:.2f}'
        }
        train_table = train_table._append(new_row, ignore_index=True)
        
        new_row = {
            "Dataset": rowname,
            "$Acc_T$": f'{exp_output["analysis"]["avg_acc_test_teacher"]:.2f}',
            "$Acc_S$": f'{exp_output["analysis"]["avg_acc_test_student"]:.2f}',
            "$Acc_D$": f'{exp_output["analysis"]["avg_acc_test_distilled"]:.2f}',
            "$t_T$": f'{exp_output["analysis"]["avg_time_test_teacher"]:.2f}',
            "$t_S$": f'{exp_output["analysis"]["avg_time_test_student"]:.2f}',
            "$t_D$": f'{exp_output["analysis"]["avg_time_test_distilled"]:.2f}'
        }
        test_table = test_table._append(new_row, ignore_index=True)

    # Define table configurations
    table_configs = [
        {
            "name": "test_acc_table",
            "file_name": "test_acc_table",
            "caption": "Average Test Accuracy (\\%)",
            "label": "tab:test-acc"
        },
        {
            "name": "training_time_table",
            "file_name": "training_time_table",
            "caption": "Average Training Time (s)",
            "label": "tab:training-time"
        },
        {
            "name": "test_time_table",
            "file_name": "test_time_table",
            "caption": "Average Test Time (s)",
            "label": "tab:test-time"
        },
        {
            "name": "train_acc_table",
            "file_name": "train_acc_table",
            "caption": "Average Train Accuracy (\\%)",
            "label": "tab:train-acc"
        },
        {
            "name": "hyperparam_table",
            "file_name": "hyperparam_table",
            "caption": "Hyperparameters",
            "label": "tab:hyperparams"
        },
        {
            "name": "dataset_size_table",
            "file_name": "dataset_size_table",
            "caption": "Dataset Size",
            "label": "tab:dataset-size"
        },
        {
            "name": "train_table",
            "file_name": "train_table",
            "caption": "Train Table",
            "label": "tab:train-table"
        },
        {
            "name": "test_table",
            "file_name": "test_table",
            "caption": "Test Table",
            "label": "tab:test-table"
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
            column_format="l"*len(table.columns),
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
        j("combined_results", "EMNIST_tC1000_sC100_tT100_sT100_ts4.0_ss4.0_te60_se120_temp4.0_a0.5_z0.2"),
        j("combined_results", "IMDB_tC8000_sC4000_tT6000_sT6000_ts7.0_ss7.0_te30_se60_temp3.0_a0.5_z0.2"),
        j("combined_results", "MNIST_tC1000_sC100_tT10_sT10_ts4.0_ss4.0_te60_se120_temp3.0_a0.5_z0.2"),
        j("combined_results", "KMNIST_tC2000_sC200_tT100_sT100_ts8.2_ss8.2_te60_se120_temp4.0_a0.5_z0.2"),
    ])
