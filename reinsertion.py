import torch
from torch_geometric.data import Data
from util import get_largest_connected_component
from tqdm import tqdm
from global_settings import device


def reinsert_node_to_pyg_graph(current_data: Data, current_node_ids: list[int], original_data: Data, node_id: int) -> Data:
    """
    Reinsert a node (specified by its original id) from the original_data
    into the current_data graph.

    Parameters:
      current_data: PyG Data object representing the current (dismantled) graph.
      original_data: PyG Data object for the full original graph.
      node_id: the original node id to reinsert.

    Returns:
      current_data: Updated Data object with the node (and its incident edges)
                    reinserted if possible.
    """



    # Get the new node's feature from original_data (shape [num_features])
    new_feat = original_data.x[node_id].unsqueeze(0)  # shape [1, num_features]
    current_node_ids.append(node_id)
    new_node_idx = current_data.num_nodes

    new_x = torch.cat([current_data.x, new_feat], dim=0)
    if hasattr(original_data, 'y') and original_data.y is not None:
        new_y = torch.cat([current_data.y,  original_data.y[node_id].unsqueeze(0)], dim=0)
    else:
        new_y = None

    # now we add back the edges
    orig_edges = original_data.edge_index.tolist()
    new_edges = []

    for u, v in zip(*orig_edges):
        if u == node_id and (v in current_node_ids):
            v_idx = current_node_ids.index(v)
            new_edges.append([new_node_idx, v_idx])
        elif v == node_id and (u in current_node_ids):
            u_idx = current_node_ids.index(u)
            new_edges.append([u_idx, new_node_idx])

    if len(new_edges) > 0:
        new_edges_tensor = torch.tensor(new_edges, dtype=torch.int32).to(device).t().contiguous()
        new_edge_index = torch.cat([current_data.edge_index, new_edges_tensor], dim=1)
    else:
        new_edge_index = current_data.edge_index

    return Data(x=new_x, y=new_y, edge_index=new_edge_index)


def reinsertion(original_data, dismantled_data, lcc_size_threshold, removals: list[int], start_lcc_size: int = None):
    dismantled_lcc_size = get_largest_connected_component(dismantled_data).num_nodes
    if dismantled_lcc_size > lcc_size_threshold:
        raise ValueError(f"the dismantled graph provided to reinsert nodes into has a LCC that is too large: "
                         f"{dismantled_lcc_size} compared to the lcc size threshold of {lcc_size_threshold}.")
    current_node_ids = [node_id for node_id in list(range(original_data.num_nodes)) if node_id not in removals]
    if start_lcc_size is None:
        start_lcc_size = get_largest_connected_component(original_data).num_nodes
    # we go through the removals in reverse order
    reinserted_nodes = []
    new_data = dismantled_data.clone()
    # pbar = tqdm(reversed(removals))
    history = []
    # pbar.set_description(f"Reinsertion")
    # for removed_node in pbar:
    for removed_node in reversed(removals):
        candidate_data = reinsert_node_to_pyg_graph(new_data, current_node_ids, original_data, removed_node)
        candidate_lcc_size = get_largest_connected_component(candidate_data).num_nodes

        if candidate_lcc_size <= lcc_size_threshold:
            new_data = candidate_data
            reinserted_nodes.append(removed_node)
            history.append((len(removals) - len(reinserted_nodes), candidate_lcc_size/start_lcc_size))
        else:
            current_node_ids.pop(-1)  # gotta remove the node id we added as it's not actually in the graph

    # print(f"Original num of removed nodes: {len(removals)} | Reinserted {len(reinserted_nodes)} nodes | "
    #       f"New num of removed nodes: {len(removals) - len(reinserted_nodes)}")
    return new_data, reinserted_nodes, history



