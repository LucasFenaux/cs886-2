#   This file is part of GDM (Graph Dismantling with Machine learning),
#   proposed in the paper "Machine learning dismantling and
#   early-warning signals of disintegration in complex systems"
#   by M. Grassia, M. De Domenico and G. Mangioni.
#
#   GDM is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   GDM is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with GDM.  If not, see <http://www.gnu.org/licenses/>.

# from collections import defaultdict

import torch
from torch.nn import functional as F
# from common import dotdict
from torch_geometric.data import Batch
from torch_geometric.nn import GATConv, global_mean_pool

from global_settings import features_they_use, device
# from models.base import BaseModel
# from network_dismantling.machine_learning.pytorch.common import DefaultDict
from models.gdm.base import BaseModel


class GDM_GATArgs:
    def __init__(self, conv_layers: list[int] = (10, ), heads: list[int] = (1, ), fc_layers: list[int] = (100, ),
                 concat: bool = True, negative_slope: float = 0.2, dropout: float = 0.3, bias: bool = True,
                 seed_train: int = 0, use_sigmoid: bool = True, for_rl: bool = False):
        self.conv_layers = conv_layers
        self.heads = heads
        self.fc_layers = fc_layers
        self.concat = [concat]*len(conv_layers)
        self.negative_slope = [negative_slope]*len(conv_layers)
        self.dropout = [dropout]*len(conv_layers)
        self.bias = [bias]*len(conv_layers)
        self.features = features_they_use
        self.seed_train = seed_train
        self.use_sigmoid = use_sigmoid
        self.for_rl = for_rl


class GAT_Model(BaseModel):
    _model_parameters = ["conv_layers", "heads", "fc_layers", "concat", "negative_slope", "dropout", "bias"]
    _affected_by_seed = False

    # def __getstate__(self):
    #     # Copy the object's state from self.__dict__ which contains
    #     # all our instance attributes. Always use the dict.copy()
    #     # method to avoid modifying the original state.
    #     state = self.__dict__.copy()
    #     # Remove the unpicklable entries.
    #     del state['add_model_parameters']
    #     del state["parameters_combination_validator"]
    #     print(state)
    #     return state
    #
    # def __setstate__(self, state):
    #     # Restore instance attributes (i.e., filename and lineno).
    #     self.__dict__.update(state)

    def __init__(self, args):

        assert len(args.conv_layers) == len(args.heads)

        super(GAT_Model, self).__init__()

        self.features = args.features
        self.num_features = len(self.features)
        self.conv_layers = args.conv_layers
        self.heads = args.heads
        self.fc_layers = args.fc_layers
        self.concat = args.concat
        self.negative_slope = args.negative_slope
        self.dropout = args.dropout
        self.bias = args.bias
        self.seed_train = args.seed_train
        self.use_sigmoid = args.use_sigmoid
        self.for_rl = args.for_rl
        self.to_scores = None
        if self.for_rl:
            self.to_scores = torch.Linear(1, 1)
        # Call super

        self.convolutional_layers = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        self.fullyconnected_layers = torch.nn.ModuleList()

        # TODO support non constant concat values
        for i in range(len(self.conv_layers)):
            num_heads = self.heads[i - 1] if ((self.concat[i - 1] is True) and (i > 0)) else 1
            in_channels = self.conv_layers[i - 1] * num_heads if i > 0 else self.num_features
            self.convolutional_layers.append(
                GATConv(in_channels=in_channels,
                        out_channels=self.conv_layers[i],
                        heads=self.heads[i],
                        concat=self.concat[i],
                        negative_slope=self.negative_slope[i],
                        dropout=self.dropout[i],
                        bias=self.bias[i])
            )

            num_out_heads = self.heads[i] if self.concat[i] is True else 1
            self.linear_layers.append(
                torch.nn.Linear(in_features=in_channels, out_features=self.conv_layers[i] * num_out_heads)
            )

        # Regressor

        # If last layer output is not a regressor, append a layer
        if self.fc_layers[-1] != 1:
            self.fc_layers.append(1)

        for i in range(len(self.fc_layers)):
            num_heads = self.heads[-1] if ((self.concat[-1] is True) and (i == 0)) else 1
            in_channels = self.fc_layers[i - 1] if i > 0 else self.conv_layers[-1] * num_heads
            self.fullyconnected_layers.append(
                torch.nn.Linear(in_features=in_channels, out_features=self.fc_layers[i])
            )

    def forward(self, x, edge_index):

        for i in range(len(self.convolutional_layers)):
            x = F.elu(self.convolutional_layers[i](x, edge_index) + self.linear_layers[i](x))

        x = x.view(x.size(0), -1)
        for i in range(len(self.fullyconnected_layers)):
            # TODO ELU?
            x = F.elu(self.fullyconnected_layers[i](x))

        x = x.view(x.size(0))
        # TODO PUT BACK SIGMOID IF USING MSELOSS
        if self.use_sigmoid:
            x = torch.sigmoid(x)

        if self.for_rl:
            x = self.to_scores(x)

        # print(x.size())
        return x

    def add_run_parameters(self, run: dict):
        for parameter in self._model_parameters:
            if parameter != "fc_layers":
                num_layers = len(self.conv_layers)
            else:
                num_layers = len(self.fc_layers)

            run[parameter] = ','.join(str(vars(self)[parameter][i]) for i in range(num_layers)) + ","

        # run["seed"] = self.seed_test

    def model_name(self):
        name = []
        for parameter in self._model_parameters:
            if parameter != "fc_layers":
                num_layers = len(self.conv_layers)
            else:
                num_layers = len(self.fc_layers)

            name.append("{}{}".format(''.join(x[0].upper() for x in parameter.split("_")),
                                      '_'.join(str(vars(self)[parameter][i]) for i in range(num_layers))
                                      )
                        )
        name.append("S{}".format(self.seed_train))

        return '_'.join(name)


    @classmethod
    def train_model(cls, online_net, target_net, optimizer, batch, gamma):
        # can't do it batched because problems
        batch_size = len(batch.state)
        optimizer.zero_grad()
        avg_loss = 0.
        for i in range(batch_size):
            state = batch.state[i]
            next_state = batch.next_state[i]
            action = torch.tensor(batch.action[i], dtype=torch.long).to(device)
            reward  = batch.reward[i]

            pred = online_net(state)
            next_pred = target_net(next_state)

            pred = pred.gather(0, action)

            target = reward + gamma * next_pred.max()

            loss = F.mse_loss(pred, target.detach()) / batch_size  # we scale by the batch size to avoid explosion
            avg_loss += loss.item()
            loss.backward()
        optimizer.step()
        return avg_loss

    @classmethod
    def train_model_batched(cls, online_net, target_net, optimizer, batch, gamma):

        batch_size = len(batch.state)
        # states = torch.stack(batch.state).contiguous()
        states = Batch.from_data_list(batch.state)
        # next_states = torch.stack(batch.next_state).contiguous()
        next_states = Batch.from_data_list(batch.next_state)
        actions = torch.tensor(batch.action, dtype=torch.long).contiguous().view(batch_size, -1)
        rewards = torch.tensor(batch.reward).contiguous().view(batch_size)




        pred = online_net(states)
        # pred = global_mean_pool(pred, states.batch)  # shape: [batch_size, num_actions]

        next_pred = target_net(next_states)
        # pred = global_mean_pool(next_pred, next_states.batch)  # shape: [batch_size, num_actions]

        state_list = states.to_data_list()
        next_step_states = next_states.to_data_list()

        def reshape_pred(pred_to_reshape, pred_state_list):
            new_pred = []
            index = 0
            for data in pred_state_list:
                part_pred = pred_to_reshape[index:index + data.num_nodes]
                new_pred.append(part_pred)
                index += data.num_nodes
            return new_pred  # can't concatenate them because could be different dims

        reshaped_pred = reshape_pred(pred, state_list)
        reshaped_next_pred = reshape_pred(next_pred, next_step_states)

        actions = actions.to(device)
        rewards = rewards.to(device)

        # pred = pred.gather(1, actions)
        pred = []
        for i, graph_pred in enumerate(reshaped_pred):
            pred.append(graph_pred[actions[i]])
        pred = torch.cat(pred)

        next_pred_max = []
        for i, next_graph_pred in enumerate(reshaped_next_pred):
            next_pred_max.append(next_graph_pred.max().unsqueeze(0))
        next_pred_max = torch.cat(next_pred_max)

        target = rewards + gamma * next_pred_max

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        # trying to stabilize training
        torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
        optimizer.step()

        return loss.item()