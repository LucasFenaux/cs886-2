import torch
from tqdm import tqdm
from data import generate_data
from util import plot_generated_data, get_largest_connected_component
from models.model import get_model


def main():
    num_nodes = 1000
    epochs = 100
    num_layers = 2
    model_name = 'GAT'
    plot = False
    num_classes = 2
    out_channels = 8
    # means = [[-0.1], [0.1]]
    # stds = [[1], [1]]
    base_mean = 0.5
    base_std = 1
    num_node_features = 1  # cost of removal
    means = [[-base_mean]*num_node_features, [base_mean]*num_node_features]
    stds = [[base_std]*num_node_features, [base_std]*num_node_features]
    p, q = 0.02, 0.01

    train_data = generate_data(num_nodes, means, stds, p, q)
    # train_data = graph_tool_to_pyg(load_graph_file("../review/dataset/test_synth/Erdos_Renyi_n100000_m400000_UD_0.gt"))
    print(f"Training largest connected component: {get_largest_connected_component(train_data).num_nodes}")

    if plot:
        plot_generated_data(train_data, "train")

    test_data = generate_data(num_nodes, means, stds, p, q)
    # test_data = graph_tool_to_pyg(load_graph_file("../review/dataset/test_synth/Erdos_Renyi_n100000_m400000_UD_1.gt"))

    print(f"Test largest connected component: {get_largest_connected_component(train_data).num_nodes}")

    # model = GCN(in_channels=num_node_features, out_channels=8).to(device)
    model = get_model(model_name, num_node_features, out_channel=out_channels, num_classes=num_classes,
                      num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    print(model(train_data).size())
    pbar = tqdm(range(epochs), desc="Epoch")

    # Training loop
    model.train()
    final_loss, final_acc = 0, 0
    for epoch in pbar:
        optimizer.zero_grad()
        out = model(train_data).squeeze()
        loss = criterion(out, train_data.y)
        loss.backward()
        optimizer.step()
        pred = (out > 0.5)
        accuracy = (pred == (train_data.y.to(torch.bool))).to(torch.float).mean().item()
        final_loss, final_acc = loss, accuracy
        pbar.set_description(f'Epoch {epoch + 1}, Loss: {loss.item()}, Accuracy: {accuracy}')
    print(f"Final | Loss: {final_loss.item()}, Accuracy: {final_acc}")

    if plot:
        plot_generated_data(test_data, "test")

    # testing
    model.eval()
    with torch.no_grad():
        out = model(test_data).squeeze()
        loss = criterion(out, test_data.y)
        accuracy = ((out > 0.5) == (test_data.y.to(torch.bool))).to(torch.float).mean().item()
        print(f"Test | Loss: {loss.item()}, Accuracy: {accuracy}")



if __name__ == '__main__':
    main()