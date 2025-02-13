from global_settings import device
from torch_geometric.nn.models import GAT
from models.gdm.GDM_GAT import GAT_Model, GDM_GATArgs
from models.pyg_models import GCN, TGModelWrapper



def get_model(model_name, in_channel, out_channel, num_classes, num_layers: int = 3, gdm_args: GDM_GATArgs = None):
    if model_name == 'GCN':
        model = GCN(in_channel, out_channel, num_layers, num_classes)
    elif model_name == 'GAT':
        model = TGModelWrapper(GAT(in_channel, out_channel, num_layers=num_layers), out_channel, num_classes)
    elif model_name == 'GDM_GAT':
        if gdm_args is None:
            gdm_args = GDM_GATArgs()
        model = TGModelWrapper(GAT_Model(gdm_args), 1, num_classes)
    else:
        raise NotImplementedError

    return model.to(device)