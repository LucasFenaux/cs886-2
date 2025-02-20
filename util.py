import torch_geometric
from torch_geometric.transforms import LargestConnectedComponents
import matplotlib.pyplot as plt
import torch
import numpy as np
from torch_geometric.data import Data
from graph_tool.all import Graph, load_graph
from global_settings import device, features_they_use


def plot_generated_data(data, label: str = ""):
    """
    Plots the generated 2D points.

    Assumes:
      - data.x is a tensor of shape (num_nodes, 2) containing the 2D feature values.
      - data.y is a tensor of shape (num_nodes,) with binary labels (0 or 1).
    Each class is plotted in a different color.
    """
    # Move tensors to CPU and convert to NumPy arrays
    points = data.x.cpu().numpy()  # shape: (num_nodes, 2)
    labels = data.y.cpu().numpy().flatten().astype(int)

    plt.figure(figsize=(8, 6))

    # Define colors for classes 0 and 1
    colors = ['red', 'blue']

    # Plot points for each class
    for cls, color in zip([0, 1], colors):
        idx = (labels == cls)
        plt.scatter(points[idx, 0], points[idx, 1],
                    color=color, label=f"Class {cls}", alpha=0.7, edgecolor='k')

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(label + " | Scatter Plot of Generated 2D Points")
    plt.legend()
    plt.grid(True)
    plt.show()


def get_largest_connected_component(data):
    """
    Uses the LargestConnectedComponents transform to extract the largest
    connected subgraph from the input data.
    """
    transform = LargestConnectedComponents()
    largest_component_data = transform(data)  # Returns data corresponding to the largest connected component.
    return largest_component_data


def pyg_to_graph_tool(data: Data, directed: bool = False) -> Graph:
    """
    Converts a PyTorch Geometric graph to a graph_tool Graph.

    Args:
        data (Data): A torch_geometric.data.Data object.
        directed (bool): Whether the resulting graph_tool Graph should be directed.

    Returns:
        g (Graph): A graph_tool Graph object with the same vertices and edges as data.
    """
    # Create a new graph-tool graph
    g = Graph(directed=directed)

    # Determine the number of nodes.
    # The Data object should have `num_nodes` or use x's first dimension.
    if hasattr(data, 'num_nodes'):
        num_nodes = data.num_nodes
    else:
        num_nodes = data.x.size(0) if data.x is not None else 0

    # Add vertices to the graph
    g.add_vertex(num_nodes)

    # Ensure edge_index is on CPU and convert to numpy for iteration.
    edge_index = data.edge_index.cpu().numpy()  # shape: [2, num_edges]

    # Add edges. Note: if your PyG graph is undirected and has duplicate entries,
    # graph-tool might add parallel edges. Adjust as needed.
    for src, dst in zip(edge_index[0], edge_index[1]):
        # Adding the edge using vertex indices.
        g.add_edge(int(src), int(dst))

    # Optionally, transfer node features as a vertex property map.
    if hasattr(data, 'x') and data.x is not None:
        # Determine data type (assume floats for now)
        # Create a property map for vertices with vector<double>
        prop = g.new_vertex_property("vector<double>")
        # Convert data.x to numpy (assumed shape: [num_nodes, feature_dim])
        x_np = data.x.cpu().numpy()
        for v in range(num_nodes):
            # Store the feature vector as a list of floats
            prop[g.vertex(v)] = x_np[v].tolist()
        # Attach the property map with a name, e.g., "x"
        g.vertex_properties["x"] = prop

    # Optionally, transfer node labels (if available)
    if hasattr(data, 'y') and data.y is not None:
        # Assume y is a 1D tensor with one label per node
        y_prop = g.new_vertex_property("int")
        y_np = data.y.cpu().numpy().flatten()
        for v in range(num_nodes):
            y_prop[g.vertex(v)] = int(y_np[v])
        g.vertex_properties["y"] = y_prop

    return g


def graph_tool_to_pyg(gt_graph: Graph, add_reverse_edges: bool = True) -> Data:
    """
    Convert a graph-tool Graph to a PyTorch Geometric Data object.

    If no vertex property "x" is found, each node is assigned a feature equal to its order (starting at 1).

    Args:
        gt_graph (Graph): A graph-tool Graph object.
        add_reverse_edges (bool): If True and the graph is undirected,
            add the reverse edge for each edge to ensure symmetry.

    Returns:
        Data: A torch_geometric.data.Data object with attributes:
              - edge_index: LongTensor of shape [2, num_edges]
              - x (optional): FloatTensor of node features.
              - y (optional): LongTensor of node labels.
    """
    # Determine the number of nodes
    num_nodes = int(gt_graph.num_vertices())

    # Extract edges from the graph-tool graph.
    edge_list = []
    for e in gt_graph.edges():
        u = int(e.source())
        v = int(e.target())
        edge_list.append((u, v))
        if add_reverse_edges and not gt_graph.is_directed():
            edge_list.append((v, u))

    if len(edge_list) > 0:
        edge_index_np = np.array(edge_list).T  # shape: [2, num_edges]
        edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Transfer vertex property "x" if present, otherwise assign default features.
    x = None
    if 'x' in gt_graph.vertex_properties:
        x_prop = gt_graph.vertex_properties['x']
        x_list = []
        for v in gt_graph.vertices():
            value = x_prop[v]
            try:
                feature = list(value)
            except TypeError:
                feature = [float(value)]
            x_list.append(feature)
        x = torch.tensor(x_list, dtype=torch.float)
    else:
        # No node features found; assign each node a feature equal to its order (starting at 1)
        x = torch.arange(1, num_nodes + 1, dtype=torch.float).unsqueeze(-1)

    # Transfer vertex property "y" if available.
    y = None
    if 'y' in gt_graph.vertex_properties:
        y_prop = gt_graph.vertex_properties['y']
        y_list = [int(y_prop[v]) for v in gt_graph.vertices()]
        y = torch.tensor(y_list, dtype=torch.float)
    else:
        y = (torch.rand(num_nodes, dtype=torch.float) > 0.5).to(torch.float)

    # Create and return the PyG Data object.
    data = Data(x=x.to(device), edge_index=edge_index.to(device), y=y.to(device))
    return data


def save_graph(graph, filename):
    if isinstance(graph, Graph) and filename.endswith('.gt'):
        graph.save(filename)
    elif isinstance(graph, torch_geometric.data.Data) and filename.endswith('.pt'):
        torch.save(graph, filename)
    else:
        raise TypeError("Graph must be of type torch_geometric.data.Data or a graph_tool.Graph")

def load_graph_file(filename):
    if isinstance(filename, str):
        if filename.endswith('.pt'):
            graph = torch.load(filename)
        elif filename.endswith('.gt'):
            graph = load_graph(filename)
        else:
            raise TypeError("Filename must be either .gt or .pt")
        return graph



def prepare_gdm_graph(network, features=None, targets=None):
    # Retrieve node features and targets
    # features_property = network.vertex_properties["features"]
    all_features = ["num_vertices", "num_edges", "degree", "clustering_coefficient", "eigenvectors", "chi_degree",
                    "chi_lcc", "pagerank_out", "betweenness_centrality", "kcore"]
    # TODO IMPROVE ME
    if features is None:
        # features = [feature for feature in list(network.vertex_properties.keys()) if feature in all_features]
        features = features_they_use
    if "None" in features:
        x = np.ones((network.num_vertices(), 1))
    else:
        x = np.column_stack(
            tuple(
                network.vertex_properties[feature].get_array() for feature in features
            )
        )
    x = torch.from_numpy(x).to(torch.float).to(device)
    #  NEW
    target_they_use = "t_0.18"
    if targets is None:
        targets = target_they_use

    if targets not in list(network.vertex_properties.keys()):
        y = None
    else:
        targets = network.vertex_properties[targets]

        y = targets.get_array().copy()
        y = torch.from_numpy(y).to(torch.float).to(device)

    edge_index = np.empty((2, 2 * network.num_edges()), dtype=np.int32)
    i = 0
    for e in network.edges():
        # TODO Can we replace the index here?
        # network.edge_index[e]
        edge_index[:, i] = (network.vertex_index[e.source()], network.vertex_index[e.target()])
        edge_index[:, i + 1] = (network.vertex_index[e.target()], network.vertex_index[e.source()])

        i += 2

    edge_index = torch.from_numpy(edge_index).to(torch.long).to(device)

    data = Data(x=x, y=y, edge_index=edge_index)

    return data


def remove_node_from_pyg_graph(graph: Data, node_idx: int, device_to_use = device) -> Data:
    """
    Remove a node (with index `node_idx`) from a PyTorch Geometric Data object.
    This function removes the node features, its label, and all edges connected to it,
    then re-indexes the remaining nodes.

    Parameters:
        graph (Data): A PyTorch Geometric Data object with attributes `x`, `edge_index`, and `y`.
        node_idx (int): The index of the node to be removed.

    Returns:
        Data: A new Data object with the node removed.
    """
    # Determine the original number of nodes.
    num_nodes = graph.x.size(0) if graph.x is not None else None
    if num_nodes is None:
        raise ValueError("Graph does not have node features (graph.x), cannot determine number of nodes.")

    if node_idx < 0 or node_idx >= num_nodes:
        raise IndexError("node_idx is out of bounds.")

    # Remove the node from the node features.
    if graph.x is not None:
        new_x = torch.cat([graph.x[:node_idx], graph.x[node_idx + 1:]], dim=0).to(device_to_use)
    else:
        new_x = None

    # Remove the node from the labels.
    if graph.y is not None:
        # If y is a 1D tensor
        if graph.y.dim() == 1:
            new_y = torch.cat([graph.y[:node_idx], graph.y[node_idx + 1:]], dim=0).to(device_to_use)
        else:
            # If y has more dimensions, remove the row corresponding to the node.
            new_y = torch.cat([graph.y[:node_idx], graph.y[node_idx + 1:]], dim=0).to(device_to_use)
    else:
        new_y = None

    # Remove all edges incident to the node.
    edge_index = graph.edge_index
    # Create a mask for edges that do not involve the node.
    mask = (edge_index[0] != node_idx) & (edge_index[1] != node_idx)
    new_edge_index = edge_index[:, mask].clone().to(device_to_use)

    # Adjust the indices of remaining nodes:
    # Any node index greater than node_idx should be decremented by 1.
    new_edge_index[new_edge_index > node_idx] -= 1

    # Construct and return the new Data object.
    new_graph = Data(x=new_x, edge_index=new_edge_index, y=new_y)
    return new_graph


