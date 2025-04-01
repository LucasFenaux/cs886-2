import torch
from global_settings import device


class LinkPredictor(torch.nn.Module):
    def __init__(self, embedding_model, embedding_size: int):
        super(LinkPredictor, self).__init__()
        self.embedding_model = embedding_model
        self.embedding_size = embedding_size
        self.q_value_estimator = torch.nn.Sequential(torch.nn.Linear(2*self.embedding_size, self.embedding_size),
                                                     torch.nn.ReLU(), torch.nn.Linear(self.embedding_size, 1))

    # def edge_q_value(self, z_1, z_2):
    #     return self.q_value_estimator(torch.cat([z_1, z_2], dim=0)).squeeze()

    def forward(self, data):
        # compute the Q-values for all edges
        z = self.embedding_model.encode(data)
        num_nodes = z.size(0)

        # Generate all pairs of nodes efficiently
        z_i = z.repeat_interleave(num_nodes, dim=0)  # shape: [num_nodes*num_nodes, embedding_size]
        z_j = z.repeat(num_nodes, 1)  # shape: [num_nodes*num_nodes, embedding_size]

        # Compute all q-values in a single batch call
        q_values = self.q_value_estimator(torch.cat([z_i, z_j], dim=1)).squeeze()
        return q_values

    @staticmethod
    def construct_edge_mask(edge_index, num_nodes):
        assert num_nodes == 309  # TODO: debugging, remember to remove
        adj_mask = torch.zeros((num_nodes*num_nodes), dtype=torch.bool).to(device)
        # we mask existing edges
        for l in range(len(edge_index[0])):
            i, j = edge_index[0][l], edge_index[1][l]
            adj_mask[i+(num_nodes*j)] = True
        return adj_mask

    def get_action(self, data):
        q_values = self.forward(data)
        adj_mask = self.construct_edge_mask(data.edge_index, data.num_nodes)
        q_values[adj_mask] = -torch.inf
        best_q_value_idx = torch.argmax(q_values, dim=0).item()
        assert q_values[best_q_value_idx] != -torch.inf
        return best_q_value_idx

        # self.predictor = lambda x_i, x_j: (x_i * x_j).sum(dim=-1)
    #
    # def encode(self, x, edge_index):
    #     return self.embedding_model.encode(x, edge_index)
    #
    # def decode(self, z, edge_index):
    #     return self.predictor(z[edge_index[0]], z[edge_index[1]])
    #
    # def decode_all(self, z):
    #     prob_adj = z@z.t()
    #     return prob_adj


