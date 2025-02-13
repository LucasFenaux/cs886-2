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
        if num_classes == 2:
            # binary classification
            self.lin = Linear(out_channels, 1)
        else:
            self.lin = Linear(out_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.lin(x)
        return x