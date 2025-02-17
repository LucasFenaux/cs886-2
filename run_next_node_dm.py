import argparse
from data import GDMTrainingData, GDMTestData
from torch_geometric.loader import DataLoader
from models.models import get_model, GDM_GATArgs
from global_settings import device, threshold
from tqdm import tqdm
import torch
from util import get_largest_connected_component, remove_node_from_pyg_graph
from smoothed_value import SmoothedValue
import torch.nn.functional as F
from reinsertion import reinsertion
from generate_labels import generate_keystone_labels


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


def train_key_every_step(args, train_loader, eval_loader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    lcc_threshold_fn = lambda x: max(int(threshold * x), 1)

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
                new_y = generate_keystone_labels(new_data, lcc_threshold)
                new_data = data.clone()
                new_data.y = new_y
                # make sure it's only 0's and 1's
                assert sum((new_data.y == 0).to(torch.int32)) + sum((new_data.y == 1).to(torch.int32)) == len(new_data.y)


                nodes_removed = 0
                node_ids = list(range(data.num_nodes))
                removals = []
                # while current_lcc_size > lcc_threshold:
                #     num_keystones = new_data.y.sum().item()

                while current_lcc_size > lcc_threshold:
                    optimizer.zero_grad()
                    pred = model(new_data).squeeze()
                    loss_fn = torch.nn.CrossEntropyLoss()
                    labels = new_data.y/new_data.y.sum().item()
                    loss = loss_fn(pred, labels)

                    # loss = loss_fn(pred, new_data.y.to(torch.float))
                    local_loss.update(loss.item(), n=1)
                    optimizer.step()

                    # now we got to remove the node we predicted out
                    node_idx = pred.argmax(dim=0)
                    is_keystone = new_data.y[node_idx].item() == 1
                    new_data = remove_node_from_pyg_graph(new_data, node_idx).to(device)
                    num_keystones = new_data.y.sum().item()
                    node_id = node_ids.pop(node_idx)
                    nodes_removed += 1
                    removals.append(node_id)
                    current_lcc_size = get_largest_connected_component(new_data).num_nodes

                    if current_lcc_size <= lcc_threshold:
                        break
                    new_y = generate_keystone_labels(new_data, lcc_threshold)
                    new_data.y = new_y

                    # if current_lcc_size > lcc_threshold:
                    #     # compute the next set of keystones
                    #     new_y = generate_keystone_labels(new_data, lcc_threshold_fn)
                    #     new_data.y = new_y
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

        if epoch % args.evaluate_every == 0 or epoch == args.epochs - 1:
            total_nodes_removed_pre_reinsert, total_nodes_removed_post_reinsert = evaluate(model, eval_loader,
                                                                                           args.reinsert, lcc_threshold_fn)

            # eval_nodes_removed, histograms = evaluate(model, eval_loader, args.reinsert)
            # eval_nodes_removed_history.append(eval_nodes_removed)
            # eval_histograms.append(histograms)
            if args.reinsert:
                print(f"Eval Nodes Removed: {total_nodes_removed_pre_reinsert.global_avg:.2f} | with reinsert: {total_nodes_removed_post_reinsert.global_avg:.2f}")
            else:
                print(f"Eval Nodes Removed: {total_nodes_removed_pre_reinsert.global_avg:.2f}")
            print(
                f"Epoch global metrics: Loss  {total_loss.global_avg:.4f} | Nodes removed: {total_nodes_removed.global_avg:.2f} | with reinsert: {total_nodes_removed_after_reinsertion.global_avg:.2f}")
        else:
            print(f"Epoch global metrics: Loss  {total_loss.global_avg:.4f} | Nodes removed: {total_nodes_removed.global_avg:.2f} | with reinsert: {total_nodes_removed_after_reinsertion.global_avg:.2f}")




def train(args, train_loader, eval_loader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    # loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    lcc_threshold_fn = lambda x: max(int(threshold * x), 1)

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
                new_y = generate_keystone_labels(new_data, lcc_threshold)
                new_data = data.clone()
                new_data.y = new_y
                # make sure it's only 0's and 1's
                assert sum((new_data.y == 0).to(torch.int32)) + sum((new_data.y == 1).to(torch.int32)) == len(new_data.y)

                nodes_removed = 0
                node_ids = list(range(data.num_nodes))
                removals = []
                while current_lcc_size > lcc_threshold:
                    num_keystones = new_data.y.sum().item()

                    while num_keystones > 0:
                        optimizer.zero_grad()
                        pred = model(new_data).squeeze()
                        loss_fn = torch.nn.CrossEntropyLoss()
                        labels = new_data.y/new_data.y.sum().item()
                        loss = loss_fn(pred, labels)
                        # indices = (new_data.y == 1).nonzero(as_tuple=True)[0]
                        # labels = F.one_hot(indices, num_classes=new_data.y.numel()).to(torch.float).to(device)
                        # loss = 0.
                        # for k in range(labels.size(0)):
                        #     loss += loss_fn(pred, labels[k])

                        # loss = loss_fn(pred, new_data.y.to(torch.float))
                        local_loss.update(loss.item(), n=1)
                        optimizer.step()

                        # now we got to remove the node we predicted out
                        node_idx = pred.argmax(dim=0)
                        is_keystone = new_data.y[node_idx].item() == 1
                        new_data = remove_node_from_pyg_graph(new_data, node_idx).to(device)
                        num_keystones = new_data.y.sum().item()
                        node_id = node_ids.pop(node_idx)
                        nodes_removed += 1
                        removals.append(node_id)
                        current_lcc_size = get_largest_connected_component(new_data).num_nodes

                        # pbar.set_description(f"Epoch: {epoch} | T Loss: {total_loss.global_avg:.4f} |  "
                        #                      f"E Loss: {local_loss:.4f} | Num K: {num_keystones}")

                        # if epoch % args.evaluate_every == 0 or epoch == args.epochs - 1:
                        #     # only compute if want every epoch or only first and last
                        #     nodes_removed, _ = compute_metrics(data, pred, args.reinsert)
                        #     total_nodes_removed.update(nodes_removed, n=1)
                        #     pbar.set_description(
                        #         f"Epoch: {epoch} | Loss: {total_loss:.4f} | Nodes removed: {total_nodes_removed:.2f}")
                        # else:
                        #     pbar.set_description(f"Epoch: {epoch} | Loss: {total_loss:.4f}")

                        if current_lcc_size <= lcc_threshold:
                            break

                    if current_lcc_size > lcc_threshold:
                        # compute the next set of keystones
                        new_y = generate_keystone_labels(new_data, lcc_threshold)
                        new_data.y = new_y
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

        if epoch % args.evaluate_every == 0 or epoch == args.epochs - 1:
            total_nodes_removed_pre_reinsert, total_nodes_removed_post_reinsert = evaluate(model, eval_loader,
                                                                                           args.reinsert, lcc_threshold_fn)

            # eval_nodes_removed, histograms = evaluate(model, eval_loader, args.reinsert)
            # eval_nodes_removed_history.append(eval_nodes_removed)
            # eval_histograms.append(histograms)
            if args.reinsert:
                print(f"Eval Nodes Removed: {total_nodes_removed_pre_reinsert.global_avg:.2f} | with reinsert: {total_nodes_removed_post_reinsert.global_avg:.2f}")
            else:
                print(f"Eval Nodes Removed: {total_nodes_removed_pre_reinsert.global_avg:.2f}")
            print(
                f"Epoch global metrics: Loss  {total_loss.global_avg:.4f} | Nodes removed: {total_nodes_removed.global_avg:.2f} | with reinsert: {total_nodes_removed_after_reinsertion.global_avg:.2f}")
        else:
            print(f"Epoch global metrics: Loss  {total_loss.global_avg:.4f} | Nodes removed: {total_nodes_removed.global_avg:.2f} | with reinsert: {total_nodes_removed_after_reinsertion.global_avg:.2f}")


def main(args):
    train_dataset = GDMTrainingData()
    test_dataset = GDMTestData(size=args.test_size, test_dataset=args.test_set)  # only want the corruption network for now
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    model_args = GDM_GATArgs(conv_layers=[40, 30], heads=[10, 10], fc_layers=[100, 100])

    model = get_model(args.model, in_channel=train_dataset.num_features, out_channel=args.hidden_dim,
                      num_classes=train_dataset.num_classes, num_layers=args.num_layers, gdm_args=model_args).to(device)

    train(args, train_loader, eval_loader, model)
    # train_key_every_step(args, train_loader, eval_loader, model)



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="GDM_GAT", choices=["GDM_GAT"])
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--evaluate-every", type=int, default=-10, help="Evaluate model on validation set")
    parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate")
    parser.add_argument("-t", "--test_set", type=str, default=GDMTestData.available_test_sets[0],
                        choices=[GDMTestData.available_test_sets])
    parser.add_argument("-s", "--test_size", type=int, default=-1)
    parser.add_argument("--test_batch_size", type=int, default=1, help="Test batch size")
    parser.add_argument("--wd", "--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("-r", "--reinsert", action="store_true", help="Perform reinsertion")

    # useless ones but oh well
    parser.add_argument("-l", "--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("-d", "--hidden_dim", type=int, default=8, help="Hidden dimension")
    # parser.add_argument("--save_results", action="store_true", help="Save results")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)