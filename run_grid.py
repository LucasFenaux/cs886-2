from data import GDMTrainingData, GDMTestData
from torch_geometric.loader import DataLoader
from models.models import get_model, GDM_GATArgs
from global_settings import device, threshold
import json
import numpy as np
from scipy import stats
import csv
import os


def gdm_grid():
    from run_gdm import train, get_parser
    args = get_parser().parse_args()

    # we re-run the grid that they run for their experiment as specified in their github repository
    conv_layers = [[30, 20], [40, 30, 20, 10], [40, 30]]
    heads = [[5, 5], [1, 1, 1, 1], [10, 10]]
    fc_layers = [100, 100]
    args.epochs = 50
    args.lr = 0.003
    args.wd = 1e-5
    args.test_size = 1
    args.reinsert = True
    train_dataset = GDMTrainingData()
    test_dataset = GDMTestData(size=args.test_size, test_dataset=args.test_set)  # only want the corruption network for now
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    results = {}
    for i in range(len(conv_layers)):
    # if args.model == 'GDM_GAT':
        model_args = GDM_GATArgs(conv_layers=conv_layers[i], heads=heads[i], fc_layers=fc_layers)
    # else:
    #     model_args = None

        model = get_model(args.model, in_channel=train_dataset.num_features, out_channel=args.hidden_dim,
                          num_classes=train_dataset.num_classes, num_layers=args.num_layers, gdm_args=model_args).to(device)

        eval_histograms, eval_nodes_removed_history = train(args, train_loader, eval_loader, model)
        results[i] = {"conv_layers": conv_layers[i], "heads": heads[i], "fc_layers": fc_layers,
                      "eval_nodes_removed_history": eval_nodes_removed_history[-1].global_avg,
                      "eval_histograms": eval_histograms[-1],
                      }

    with open("./grid_results.json", "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)


def next_node_grid():
    from run_next_node_dm import main, get_parser
    args = get_parser().parse_args()
    args.epochs = 20
    args.lr = 0.01
    args.test_size = 1
    args.weight_decay = 1e-5
    args.reinsert = True
    args.evaluate_every = args.epochs * 2  # higher so that we don't waste compute
    if not os.path.exists("./results/"):
        os.makedirs("./results/")
    with open("./results/run_args.json", "w") as f:
        json.dump(vars(args), f)
    num_runs_per = 3
    confidence = 0.95
    alpha = 1 - confidence
    #parameters to vary
    losses = ["MSE", ]#"CE"]
    labels = ["probabilities", "normalized_probabilities", "keystone", "normalized_keystone"]
    recompute_every = ["step", "no_keystone"]
    use_sigmoid = [True, False]
    paths_to_evaluate_list = [1000,]

    raw_rows = [["Loss", "Labels", "Recompute_every", "use_sigmoid", "paths_to_evaluate_list", "No_reinsert", "Reinsert"], ]
    stat_rows = [["Loss", "Labels", "Recompute_every", "use_sigmoid", "paths_to_evaluate_list", "No_reinsert_avg",
                  "No_reinsert_std", "margin_of_error", "Reinsert_avg", "Reinsert_std", "margin_of_error"], ]
    for loss in losses:
        for label in labels:
            for recompute in recompute_every:
                for sigmoid in use_sigmoid:
                    for paths_to_evaluate in paths_to_evaluate_list:
                        if (label == "probabilities" or label == "normalized_probabilities") and recompute == "no_keystone":
                            continue
                        no_reinserts = []
                        reinserts = []
                        for i in range(num_runs_per):
                            args.loss = loss
                            args.labels = label
                            args.recompute_target_every = recompute
                            args.use_sigmoid = sigmoid
                            args.paths_to_evaluate = paths_to_evaluate
                            print(f"Running {loss} + {label} + {recompute} + {sigmoid} + {paths_to_evaluate} + {i}")
                            no_reinsert, reinsert = main(args)
                            no_reinserts.append(no_reinsert)
                            reinserts.append(reinsert)
                            raw_rows.append([loss, label, recompute, sigmoid, paths_to_evaluate, no_reinsert, reinsert])
                        # print(no_reinserts)
                        # print(reinserts)
                        no_reinserts = np.array(no_reinserts)
                        reinserts = np.array(reinserts)
                        n = num_runs_per

                        no_reinsert_avg = np.mean(no_reinserts)
                        reinsert_avg = np.mean(reinserts)

                        no_reinsert_std = np.std(no_reinserts)
                        reinsert_std = np.std(reinserts)

                        t_value = stats.t.ppf(1 - alpha / 2, df=n - 1)
                        no_reinsert_margin_of_error = t_value * no_reinsert_std / np.sqrt(n)
                        reinsert_margin_of_error = t_value * reinsert_std / np.sqrt(n)

                        stat_rows.append([loss, label, recompute, sigmoid, paths_to_evaluate, no_reinsert_avg,
                                          no_reinsert_std, no_reinsert_margin_of_error, reinsert_avg, reinsert_std,
                                          reinsert_margin_of_error])

    with open("./results/raw_next_node_grid.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(raw_rows)
    with open("./results/stat_next_node_grid.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(stat_rows)


if __name__ == '__main__':
    # gdm_grid()
    next_node_grid()