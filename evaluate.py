# we want to load the GDM models we trained and evaluate them on the test networks.
# we also want to load the rl models we trained and the optimal dismantling they found test them independently.
import numpy as np

from run_rl_v3 import evaluate as evaluate_rl
from run_gdm import compute_metrics
import os
import json
from data import GDMTestData, GDMTrainingData
import argparse
from tqdm import tqdm
from models.gdm.GDM_GAT import GAT_Model, GDM_GATArgs
from models.pyg_models import TGModelWrapper
import torch
from global_settings import device, lcc_threshold_fn


def evaluate_gdm(model, graph):
    model.eval()
    with torch.no_grad():
        data_batch = graph.clone()
        pred = model(data_batch).squeeze()
        nodes_removed, with_reinsert, _ = compute_metrics(data_batch, pred, True)
    return nodes_removed, with_reinsert


def get_gdm_models_paths(saved_model_path):
    gdm_model_paths = []
    for f in os.listdir(saved_model_path):
        if os.path.isfile(os.path.join(saved_model_path, f)) and f.startswith("gdm"):
            # is gdm model
            filename, ext = os.path.splitext(os.path.basename(f))
            assert ext == ".pt"
            gdm_model_paths.append((os.path.join(saved_model_path, f), filename))
    return gdm_model_paths


def main_gdm(args):
    test_dataset = GDMTestData(size=args.test_size, test_dataset=args.test_set)
    train_dataset = GDMTrainingData()

    saved_model_path = "./saved_models/"
    log_dir = "./results/"
    model_paths = get_gdm_models_paths(saved_model_path)
    print([model_paths[i][1] for i in range(len(model_paths))])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    pbar = tqdm(total=len(model_paths)*len(test_dataset))
    for j, (f, model_name) in enumerate(model_paths):
        model_args = GDM_GATArgs(conv_layers=[40, 30], heads=[10, 10], fc_layers=[100, 100])
        model = TGModelWrapper(GAT_Model(model_args), 1, num_classes=train_dataset.num_classes).to(device)
        model.load_state_dict(torch.load(f))
        model.eval()
        for i, graph in enumerate(test_dataset):
            graph_name = os.path.splitext(os.path.basename(test_dataset.files[i]))[0]
            pbar.set_description(f"Model {j+1}/{len(model_paths)}: {model_name} | Graph {i+1}/{len(test_dataset)}: {graph_name} ({graph.num_nodes})")
            if os.path.exists(os.path.join(log_dir, "results.json")):
                with open(os.path.join(log_dir, "results.json"), "r") as result_f:
                    results = json.load(result_f)
            else:
                results = {}
            if graph_name not in list(results.keys()):
                results[graph_name] = {}
            if model_name not in list(results[graph_name].keys()):
                results[graph_name][model_name] = {}
                # we need to evaluate that model on that graph
                removal_count, removal_with_reinsert_count = evaluate_gdm(model, graph.clone().to(device))
                results[graph_name][model_name]["removal_count"] = removal_count
                results[graph_name][model_name]["removal_with_reinsert_count"] = removal_with_reinsert_count
                with open(os.path.join(log_dir, "results.json"), "w+") as result_f:
                    json.dump(results, result_f, indent=4)
            pbar.update(1)

def get_rl_paths(saved_model_path, test_dataset):
    rl_model_paths = {}
    rl_removal_paths = {}
    for f in os.listdir(saved_model_path):
        if os.path.isdir(os.path.join(saved_model_path, f)):
            found = False
            for graph_file in test_dataset.files[:len(test_dataset)]:
                if f in graph_file:
                    found = True
                    break
            if found:
                for d_f in os.listdir(os.path.join(saved_model_path, f)):
                    filename, ext = os.path.splitext(os.path.basename(d_f))
                    if ext == ".pt":
                        # is a model file
                        dict_to_update = rl_model_paths
                    elif ext == ".npy":
                        # is a removal list
                        dict_to_update = rl_removal_paths
                    else:
                        continue
                    if f not in dict_to_update.keys():
                        dict_to_update[f] = []
                    dict_to_update[f].append((os.path.join(saved_model_path, f, d_f), filename))
    return rl_model_paths, rl_removal_paths


def main_rl(args):
    test_dataset = GDMTestData(size=args.test_size, test_dataset=args.test_set)
    train_dataset = GDMTrainingData()
    saved_model_path = "./saved_models/"
    log_dir = "./results/"
    rl_model_paths, rl_removal_paths = get_rl_paths(saved_model_path, test_dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    pbar = tqdm(total=len(rl_model_paths.values()))
    for i, graph in enumerate(test_dataset):
        graph_name = os.path.splitext(os.path.basename(test_dataset.files[i]))[0]
        if graph_name in list(rl_model_paths.keys()):
            # do the models
            for j, (model_path, model_name) in enumerate(rl_model_paths[graph_name]):
                model_args = GDM_GATArgs(conv_layers=[40, 30], heads=[10, 10], fc_layers=[100, 100], use_sigmoid=False,
                                         for_rl=True)
                model = TGModelWrapper(GAT_Model(model_args), 1, num_classes=train_dataset.num_classes).to(device)
                model.load_state_dict(torch.load(model_path))
                model.eval()

                pbar.set_description(f"Graph {i+1}/{len(test_dataset)}: {graph_name} ({graph.num_nodes}) | Model {j+1}/{len(rl_model_paths[graph_name])}: {model_name}")
                if os.path.exists(os.path.join(log_dir, "results.json")):
                    with open(os.path.join(log_dir, "results.json"), "r") as result_f:
                        results = json.load(result_f)
                else:
                    results = {}
                if graph_name not in list(results.keys()):
                    results[graph_name] = {}
                if model_name not in list(results[graph_name].keys()):
                    results[graph_name][model_name] = {}
                    # we need to evaluate that model on that graph
                    removals, removals_with_reinsert = evaluate_rl(model, graph.clone().to(device), lcc_threshold_fn)
                    removal_count = len(removals)
                    removal_with_reinsert_count = len(removals_with_reinsert)
                    results[graph_name][model_name]["removal_count"] = removal_count
                    results[graph_name][model_name]["removal_with_reinsert_count"] = removal_with_reinsert_count
                    with open(os.path.join(log_dir, "results.json"), "w+") as result_f:
                        json.dump(results, result_f, indent=4)
                pbar.update(1)
            # do the saved removals
            file_paths = [tup[0] for tup in rl_removal_paths[graph_name]]
            file_names = [tup[1] for tup in rl_removal_paths[graph_name]]
            keys = {"best_eval": ["best_eval_removal_nodes", "best_eval_removal_with_reinsert_nodes"],
                    "best_train": ["best_removal_nodes", "best_removal_with_reinsert_nodes"]}
            for key in keys.keys():
                if os.path.exists(os.path.join(log_dir, "results.json")):
                    with open(os.path.join(log_dir, "results.json"), "r") as result_f:
                        results = json.load(result_f)
                else:
                    results = {}
                if key not in list(results[graph_name].keys()) and keys[key][0] in file_names:
                    results[graph_name][key] = {}
                    index = file_names.index(keys[key][0])
                    file_path = file_paths[index]
                    removals = np.load(file_path).tolist()
                    removal_count = len(removals)
                    results[graph_name][key]["removal_count"] = removal_count
                    if keys[key][1] in file_names:
                        index = file_names.index(keys[key][1])
                        file_path = file_paths[index]
                        removals_with_reinsert = np.load(file_path).tolist()
                        removal_with_reinsert_count = len(removals_with_reinsert)
                        results[graph_name][key]["removal_with_reinsert_count"] = removal_with_reinsert_count
                    with open(os.path.join(log_dir, "results.json"), "w+") as result_f:
                        json.dump(results, result_f, indent=4)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_set", type=str, default=GDMTestData.available_test_sets[0],
                        choices=[GDMTestData.available_test_sets])
    parser.add_argument("-s", "--test_size", type=int, default=-1)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    main_gdm(args)
    main_rl(args)