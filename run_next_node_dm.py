import argparse
from data import GDMTrainingData, GDMTestData
from torch_geometric.loader import DataLoader
from models.models import get_model, GDM_GATArgs
from global_settings import device, threshold, lcc_threshold_fn
from tqdm import tqdm
import torch
from util import get_largest_connected_component, remove_node_from_pyg_graph
from smoothed_value import SmoothedValue
from reinsertion import reinsertion
from generate_labels import generate_keystone_labels
from run_gdm import main as run_gdm_main



def evaluate(model, loader, reinsert, lcc_threshold_fn):
    model.eval()
    total_nodes_removed_pre_reinsert = SmoothedValue()
    if reinsert:
        total_nodes_removed_post_reinsert = SmoothedValue()

    with torch.no_grad():
        for i, data_batch in enumerate(loader):
            # TODO: make somehow work with a batch size higher than 1
            for j, data in enumerate(data_batch.to_data_list()):
                new_data = data.clone()
                start_lcc_size = get_largest_connected_component(new_data).num_nodes
                current_lcc_size = start_lcc_size
                lcc_threshold = lcc_threshold_fn(start_lcc_size)
                nodes_removed = 0
                node_ids = list(range(new_data.num_nodes))
                removals = []
                while current_lcc_size > lcc_threshold:
                    pred = model(new_data).squeeze()
                    # now we got to remove the node we predicted out
                    node_idx = pred.argmax(dim=0)
                    new_data = remove_node_from_pyg_graph(new_data, node_idx).to(device)
                    node_id = node_ids.pop(node_idx)
                    nodes_removed += 1
                    removals.append(node_id)
                    current_lcc_size = get_largest_connected_component(new_data).num_nodes

                total_nodes_removed_pre_reinsert.update(nodes_removed, n=1)
                if reinsert:
                    new_data, reinserted_nodes, sub_history = reinsertion(data, new_data, lcc_threshold, removals, start_lcc_size)
                    removals = [node_id for node_id in removals if node_id not in reinserted_nodes]
                    nodes_removed_count = len(removals)
                    total_nodes_removed_post_reinsert.update(nodes_removed_count, n=1)
                else:
                    total_nodes_removed_post_reinsert = None
    return total_nodes_removed_pre_reinsert, total_nodes_removed_post_reinsert


def train(args, train_loader, eval_loader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/10)
    if args.loss_fn == "MSE":
        loss_fn = torch.nn.MSELoss()
    elif args.loss_fn == "CE":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    # loss_fn = torch.nn.BCEWithLogitsLoss()


    for epoch in range(args.epochs):
        pbar = tqdm(train_loader)
        model.train()
        total_loss = SmoothedValue()
        total_nodes_removed = SmoothedValue()
        if args.reinsert:
            total_nodes_removed_after_reinsertion = SmoothedValue()

        for i, data_batch in enumerate(pbar):
            # TODO: make somehow work with a batch size higher than 1
            for j, data in enumerate(data_batch.to_data_list()):
                local_loss = SmoothedValue()
                new_data = data.clone()
                start_lcc_size = get_largest_connected_component(data).num_nodes
                current_lcc_size = start_lcc_size
                lcc_threshold = lcc_threshold_fn(start_lcc_size)
                return_probs = (args.labels == "probabilities") or (args.labels == "normalized_probabilities")

                nodes_removed = 0
                node_ids = list(range(data.num_nodes))
                removals = []
                while current_lcc_size > lcc_threshold:
                    new_y = generate_keystone_labels(new_data, lcc_threshold, paths_to_evaluate=args.paths_to_evaluate,
                                                     return_probs=return_probs)
                    new_data.y = new_y
                    if args.recompute_target_every == "no_keystone" and (args.labels == "keystone" or args.labels == "normalized_keystone"):
                        num_keystones = new_data.y.sum().item()
                        while num_keystones > 0:
                            optimizer.zero_grad()
                            pred = model(new_data).squeeze()
                            if args.labels == "keystones":
                                labels = new_data.y
                            else:
                                labels = new_data.y / new_data.y.sum().item()
                            loss = loss_fn(pred, labels)
                            local_loss.update(loss.item(), n=1)
                            optimizer.step()
                            node_idx = pred.argmax(dim=0)
                            new_data = remove_node_from_pyg_graph(new_data, node_idx).to(device)
                            num_keystones = new_data.y.sum().item()
                            node_id = node_ids.pop(node_idx)
                            nodes_removed += 1
                            removals.append(node_id)
                            current_lcc_size = get_largest_connected_component(new_data).num_nodes
                            if current_lcc_size <= lcc_threshold:
                                break
                    else:
                        optimizer.zero_grad()
                        pred = model(new_data).squeeze()
                        if "normalized" in args.labels:
                            labels = new_data.y / new_data.y.sum().item()
                        else:
                            labels = new_data.y
                        loss = loss_fn(pred, labels)
                        local_loss.update(loss.item(), n=1)
                        optimizer.step()
                        node_idx = pred.argmax(dim=0)
                        new_data = remove_node_from_pyg_graph(new_data, node_idx).to(device)
                        node_id = node_ids.pop(node_idx)
                        nodes_removed += 1
                        removals.append(node_id)
                        current_lcc_size = get_largest_connected_component(new_data).num_nodes

                total_nodes_removed.update(nodes_removed, n=1)
                total_loss.update(local_loss.global_avg, n=1)
                if args.reinsert:
                    new_data, reinserted_nodes, sub_history = reinsertion(data, new_data, lcc_threshold,
                                                                          removals, start_lcc_size)
                    post_reinsertion_removals = [node_id for node_id in removals if node_id not in reinserted_nodes]
                    nodes_removed_after_reinsertion = len(post_reinsertion_removals)
                    total_nodes_removed_after_reinsertion.update(nodes_removed_after_reinsertion, n=1)
                    pbar.set_description(f"Epoch: {epoch} | Loss: {total_loss:.4f} | Nodes Removed: {total_nodes_removed}, with reinsert: {total_nodes_removed_after_reinsertion}")
                else:
                    pbar.set_description(f"Epoch: {epoch} | Loss: {total_loss:.4f} | Nodes Removed: {total_nodes_removed}")
        scheduler.step()
        if epoch % args.evaluate_every == 0:
            total_nodes_removed_pre_reinsert, total_nodes_removed_post_reinsert = evaluate(model, eval_loader,
                                                                                           args.reinsert, lcc_threshold_fn)
            if args.reinsert:
                print(f"Eval Nodes Removed: {total_nodes_removed_pre_reinsert.global_avg:.2f} | with reinsert: {total_nodes_removed_post_reinsert.global_avg:.2f}")
            else:
                print(f"Eval Nodes Removed: {total_nodes_removed_pre_reinsert.global_avg:.2f}")
            print(
                f"Epoch global metrics: Loss  {total_loss.global_avg:.4f} | Nodes removed: {total_nodes_removed.global_avg:.2f} | with reinsert: {total_nodes_removed_after_reinsertion.global_avg:.2f}")
        else:
            print(f"Epoch global metrics: Loss  {total_loss.global_avg:.4f} | Nodes removed: {total_nodes_removed.global_avg:.2f} | with reinsert: {total_nodes_removed_after_reinsertion.global_avg:.2f}")
    total_nodes_removed_pre_reinsert, total_nodes_removed_post_reinsert = evaluate(model, eval_loader,
                                                                                   args.reinsert, lcc_threshold_fn)
    print(f"Final results | Nodes removed: {total_nodes_removed_pre_reinsert.global_avg:.2f} | with reinsert: {total_nodes_removed_post_reinsert.global_avg:.2f}")
    return total_nodes_removed_pre_reinsert.global_avg, total_nodes_removed_post_reinsert.global_avg

def pretrain(args, epochs: int = 1):
    print("GDM pre-training")
    epochs_cache = args.epochs
    lr = args.lr
    wd = args.wd
    test_size = args.test_size
    evaluate_every = args.evaluate_every
    args.epochs = epochs
    args.lr = 0.003
    args.wd = 1e-5
    args.test_size = 1
    args.evaluate_every = args.epochs * 2
    model = run_gdm_main(args)
    args.epochs = epochs_cache
    args.lr = lr
    args.wd = wd
    args.test_size = test_size
    args.evaluate_every = evaluate_every
    print("Done with GDM pre-training")
    return model, args


def main(args):
    train_dataset = GDMTrainingData()
    test_dataset = GDMTestData(size=args.test_size, test_dataset=args.test_set)  # only want the corruption network for now
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
    model_args = GDM_GATArgs(conv_layers=[40, 30], heads=[10, 10], fc_layers=[100, 100], use_sigmoid=args.use_sigmoid)

    model = get_model(args.model, in_channel=train_dataset.num_features, out_channel=args.hidden_dim,
                      num_classes=train_dataset.num_classes, num_layers=args.num_layers, gdm_args=model_args).to(device)
    if args.pretrain:
        model, args = pretrain(args)
        evaluate(model, eval_loader, args.reinsert, lcc_threshold_fn)

    no_reinsert, reinsert = train(args, train_loader, eval_loader, model)
    return no_reinsert, reinsert


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="GDM_GAT", choices=["GDM_GAT", "GCN", "GAT"])
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--evaluate_every", type=int, default=10, help="Evaluate model on validation set")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("-t", "--test_set", type=str, default=GDMTestData.available_test_sets[0],
                        choices=[GDMTestData.available_test_sets])
    parser.add_argument("-s", "--test_size", type=int, default=-1)
    parser.add_argument("--test_batch_size", type=int, default=1, help="Test batch size")
    parser.add_argument("--wd", "--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("-r", "--reinsert", action="store_true", help="Perform reinsertion")
    parser.add_argument("--loss_fn", type=str, default="MSE", choices=["MSE", "CE"])
    parser.add_argument("--labels", type=str, default="probabilities", choices=["probabilities",
                                                                                "normalized_probabilities",
                                                                                "keystone", "normalized_keystone"])
    parser.add_argument("--recompute_target_every", type=str, default="step", choices=["step", "no_keystone"])
    parser.add_argument("--use_sigmoid", type=bool, default=True, help="Use sigmoid")
    # parser.add_argument("--save_results", type=str, default=None, help="Save results to file name provided if not None")
    # useless ones but oh well
    parser.add_argument("-l", "--num_layers", type=int, default=3, help="Number of layers")
    parser.add_argument("-d", "--hidden_dim", type=int, default=50, help="Hidden dimension")
    parser.add_argument("-p", "--paths_to_evaluate", type=int, default=1000)
    parser.add_argument("--pretrain", action="store_true")
    # parser.add_argument("--save_results", action="store_true", help="Save results")
    # parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    main(parser.parse_args())