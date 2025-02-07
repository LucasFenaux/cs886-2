import argparse
from data import GDMTrainingData, GDMTestData
from torch_geometric.loader import DataLoader
from model import get_model
from global_settings import device, threshold
from tqdm import tqdm
import torch
from util import get_largest_connected_component, remove_node_from_pyg_graph
from smoothed_value import SmoothedValue
from time import sleep
# def evaluate(model, loader):
#     # evaluate
#     model.eval()
#     with torch.no_grad():


def evaluate(model, loader):
    model.eval()
    total_nodes_removed = SmoothedValue()
    histograms = []
    with torch.no_grad():
        for data_batch in loader:
            pred = model(data_batch).squeeze()

            if len(data_batch) == 1:
                nodes_removed, histogram = compute_metrics(data_batch, pred)
                total_nodes_removed.update(nodes_removed, n=1)
                histograms.append(histogram)
            else:
                # we go one graph in the batch at a time
                data_list = data_batch.to_data_list()
                # we gotta partition pred
                index = 0
                for data in data_list:
                    part_pred = pred[index:index + data.num_nodes]
                    nodes_removed, histogram = compute_metrics(data, part_pred)
                    index += data.num_nodes
                    total_nodes_removed.update(nodes_removed, n=1)  # one element at a time
                    histograms.append(histogram)
    return total_nodes_removed, histograms


def compute_metrics(data, pred):
    data = data.to(device)
    start_lcc_size = get_largest_connected_component(data).num_nodes
    current_lcc_size = start_lcc_size
    nodes_removed_count = 0
    history = [(0, 1.0)]
    new_data = data.clone()
    new_pred = pred.clone()
    while current_lcc_size > threshold*start_lcc_size:
        node_idx = new_pred.argmax(dim=0)
        new_data = remove_node_from_pyg_graph(new_data, node_idx).to(device)
        nodes_removed_count += 1
        new_pred = torch.cat([new_pred[ :node_idx], new_pred[ node_idx + 1:]], dim=0).to(device)
        current_lcc_size = get_largest_connected_component(new_data).num_nodes
        history.append((nodes_removed_count, current_lcc_size/start_lcc_size))

    return nodes_removed_count, history


def train(args, train_loader, eval_loader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    eval_histograms = []

    for epoch in range(args.epochs):
        pbar = tqdm(train_loader)
        model.train()
        total_loss = SmoothedValue()
        total_nodes_removed = SmoothedValue()

        for i, data_batch in enumerate(pbar):
            optimizer.zero_grad()
            pred = model(data_batch).squeeze()
            loss = loss_fn(pred, data_batch.y)
            total_loss.update(loss.item(), n=len(data_batch))
            loss.backward()
            optimizer.step()
            if epoch % args.evaluate_every == 0 or epoch == args.epochs - 1:
                # only compute if want every epoch or only first and last
                if len(data_batch) == 1:
                    nodes_removed, _ = compute_metrics(data_batch, pred)
                    total_nodes_removed.update(nodes_removed, n=1)
                else:
                    # we go one graph in the batch at a time
                    data_list = data_batch.to_data_list()
                    # we gotta partition pred
                    index = 0
                    for data in data_list:
                        part_pred = pred[index:index+data.num_nodes]
                        nodes_removed, _ = compute_metrics(data, part_pred)
                        index += data.num_nodes
                        total_nodes_removed.update(nodes_removed, n=1)  # one element at a time

                pbar.set_description(f"Epoch: {epoch} | Loss: {total_loss:.4f} | Nodes removed: {total_nodes_removed:.2f}")
            else:
                pbar.set_description(f"Epoch: {epoch} | Loss: {total_loss:.4f}")

            if epoch == 0 or epoch == args.epochs - 1:
                # hella expensive so we only do it once at the start and the end
                total_nodes_removed, histograms = evaluate(model, eval_loader)
                eval_histograms.append(histograms)
                print(f"Eval Nodes Removed: {total_nodes_removed.global_avg:.2f}")
        pbar.close()
        if epoch % args.evaluate_every == 0 or epoch == args.epochs - 1:
            print(f"Epoch global metrics: Loss  {total_loss.global_avg:.4f} | Nodes removed: {total_nodes_removed.global_avg:.2f}")
        else:
            print(f"Epoch global metrics: Loss  {total_loss.global_avg:.4f}")
        sleep(0.1)




def main(args):
    train_dataset = GDMTrainingData()
    test_dataset = GDMTestData(size=args.test_size, test_dataset=args.test_set)  # only want the corruption network for now
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)

    model = get_model(args.model, in_channel=train_dataset.num_features, out_channel=args.hidden_dim,
                      num_classes=train_dataset.num_classes, num_layers=args.num_layers).to(device)

    train(args, train_loader, eval_loader, model)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default="GCN", choices=["GCN", "GAT"])
    parser.add_argument("-l", "--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("-d", "--hidden_dim", type=int, default=8, help="Hidden dimension")
    parser.add_argument("-b", "--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--evaluate-every", type=int, default=-10, help="Evaluate model on validation set")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-t", "--test_set", type=str, default=GDMTestData.available_test_sets[0],
                        choices=[GDMTestData.available_test_sets])
    parser.add_argument("-s", "--test_size", type=int, default=-1)
    parser.add_argument("--test_batch_size", type=int, default=1, help="Test batch size")
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    main(parse_args())