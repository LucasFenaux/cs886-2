
from data import GDMTrainingData, GDMTestData
from torch_geometric.loader import DataLoader
from models.models import get_model, GDM_GATArgs
from global_settings import device, threshold
import json



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




if __name__ == '__main__':
    gdm_grid()