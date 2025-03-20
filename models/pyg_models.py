from torch_geometric.nn import GCNConv
from torch.nn import Linear
import torch.nn.functional as F
import torch


class TGModelWrapper(torch.nn.Module):
    def __init__(self, model, out_channels, num_classes):
        super().__init__()
        self.model = model
        if num_classes == 2:
            # binary classification
            self.lin = Linear(out_channels, 1)
        else:
            self.lin = Linear(out_channels, num_classes)
        if out_channels == num_classes:
            self.lin = torch.nn.Identity()

    def encode(self, data):
        return self.model.encode(data.x, data.edge_index)

    def get_action(self, x):
            return torch.argmax(self.forward(x), dim=-1).item()

    def forward(self, data):
        x = self.model(data.x, data.edge_index)
        return self.lin(x)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers: int = 2, num_classes: int = 2):
        super(GCN, self).__init__()
        assert num_layers >= 2
        convs = [GCNConv(in_channels, 16)]
        for i in range(num_layers - 2):
            convs.append(GCNConv(16, 16))
        convs.append(GCNConv(16, out_channels))
        self.convs = torch.nn.ModuleList(convs)
        self.lin1 = Linear(out_channels, out_channels)
        if num_classes == 2:
            # binary classification
            self.lin2 = Linear(out_channels, 1)
        else:
            self.lin2 = Linear(out_channels, num_classes)

    def encode(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        return x

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = x.view(x.size(0))
        return x

    def get_action(self, x):
        return torch.argmax(self.forward(x), dim=-1).item()