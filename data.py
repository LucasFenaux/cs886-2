import torch
import random
from torch_geometric.data import Data, Dataset
from global_settings import device, data_root
import os
from util import prepare_gdm_graph
from graph_tool import load_graph


def generate_data(num_nodes, means, stds, p, q):
    y = (torch.rand(num_nodes) > 0.5).to(torch.int64)
    X = []
    # build features using normal distribution
    for node_id in range(num_nodes):
        label = y[node_id].item()
        feature = torch.normal(mean=torch.Tensor(means[label]), std=torch.Tensor(stds[label])).to(torch.float32)
        X.append(feature)

    X = torch.stack(X).to(device)
    edge_indices = [[], []]
    # building edge indices
    for i in range(num_nodes-1):
        for j in range(i+1, num_nodes):
            if y[i] == y[j]:
                # we use p
                if random.random() < p:
                    # undirected edge
                    edge_indices[0].append(i)
                    edge_indices[1].append(j)
                    edge_indices[0].append(j)
                    edge_indices[1].append(i)
            else:
                # we use q
                if random.random() < q:
                    # undirected edge
                    edge_indices[0].append(i)
                    edge_indices[1].append(j)
                    edge_indices[0].append(j)
                    edge_indices[1].append(i)
    # edge_indices[0] = torch.tensor(edge_indices[0], dtype=torch.long, device=device)
    # edge_indices[1] = torch.tensor(edge_indices[1], dtype=torch.long, device=device)
    edge_indices = torch.tensor(edge_indices, dtype=torch.long, device=device)
    y = y.to(torch.float32).to(device)
    data = Data(x=X, edge_index=edge_indices, y=y)
    print(data)
    return data


class GDMTrainingData(Dataset):
    num_features = 4
    num_classes = 1  # we are doing regression

    def __init__(self, root: str =os.path.join(data_root, "synth_train_NEW/dataset/"), fmt: str = ".graphml"):
        super(GDMTrainingData, self).__init__()
        self.root = root
        self.fmt = fmt
        self.files = self.find_files()

    def find_files(self):
        file_list = []
        for file in os.listdir(self.root):
            if file.endswith(self.fmt):
                file_list.append(os.path.join(self.root, file))
        return file_list

    def __getitem__(self, item):
        file = self.files[item]
        graph = load_graph(file)
        data = prepare_gdm_graph(graph)
        return data

    def __len__(self):
        return len(self.files)


class GDMTestData(Dataset):
    num_features = 4
    num_classes = 1
    available_test_sets = ["test", "test_ew", "test_large", "test_synth"]

    def __init__(self, size: int = -1, root: str = data_root, test_dataset = "test", fmt: str = ".graphml"):
        super(GDMTestData, self).__init__()
        self.root = root
        self.fmt = fmt
        assert test_dataset in self.available_test_sets
        self.test_dataset = test_dataset
        self.size = size
        self.files = self.find_files()
        if size > len(self.files):
            raise ValueError(f"size is too big, only {len(self.files)} are available")
        else:
            print(self.files[:size])

    def find_files(self):
        actual_dir = os.path.join(self.root, self.test_dataset, "dataset")
        file_list = []
        for file in os.listdir(actual_dir):
            if file.endswith(self.fmt):
                file_list.append(os.path.join(actual_dir, file))
        return sorted(file_list)  # to ensure consistency

    def __getitem__(self, item):
        if self.size == 1 and self.test_dataset == "test":
            # we choose the corruption graph to compare with the paper graph
            file =  os.path.join(self.root, self.test_dataset, "dataset", "corruption.graphml")
            assert file in self.files
        else:
            file = self.files[item]

        graph = load_graph(file)
        data = prepare_gdm_graph(graph)
        return data

    def __len__(self):
        if self.size > 0:
            return self.size
        else:
            return len(self.files)