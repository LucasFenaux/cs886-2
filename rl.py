from collections import namedtuple, deque
import numpy as np
import torch
import copy
from util import get_largest_connected_component, remove_node_from_pyg_graph
from global_settings import device

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward'))


class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward):
        self.memory.append(
            Transition(state, next_state, action, reward))

    def sample(self, batch_size):
        batch_state, batch_next_state, batch_action, batch_reward = [], [], [], []
        p = np.array([len(episode) for episode in self.memory])
        p = p / p.sum()

        batch_indexes = np.random.choice(np.arange(len(self.memory)), batch_size, p=p)

        for batch_idx in batch_indexes:
            episode = self.memory[batch_idx]

            # start = random.randint(0, len(episode) - sequence_length)
            # transitions = episode[start:start + sequence_length]
            # batch = Transition(*zip(*episode))

            batch_state.append(episode.state)
            batch_next_state.append(episode.next_state)
            batch_action.append(episode.action)
            batch_reward.append(episode.reward)

        return Transition(batch_state, batch_next_state, batch_action, batch_reward)

    def __len__(self):
        return len(self.memory)


class GraphEnv:
    def __init__(self, dataset, threshold_fn, shuffle: bool = False):
        self.dataset = dataset
        self.threshold_fn = threshold_fn
        self.counter = None
        self.idx = list(range(len(dataset)))
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.idx)

        self.removals = []

        self.current_graph = None
        self.current_lcc_size = None
        self.start_lcc_size = None
        self.threshold = None
        self.node_ids = None
        # self.set_graph_as_current()


    def get_random_action(self):
        return np.random.randint(0, high=self.current_graph.num_nodes)

    def set_graph_as_current(self):
        self.current_graph = self.dataset[self.idx[self.counter]]
        self.start_lcc_size = get_largest_connected_component(self.current_graph).num_nodes
        self.current_lcc_size = self.start_lcc_size
        self.threshold = self.threshold_fn(self.current_lcc_size)
        self.node_ids = list(range(self.current_graph.num_nodes))

    def get_removals(self):
        return copy.deepcopy(self.removals)

    def get_current_threshold(self):
        return self.threshold

    def get_start_lcc_size(self):
        return self.start_lcc_size

    def epoch_reset(self):
        self.idx = list(range(len(self.dataset)))
        if self.shuffle:
            np.random.shuffle(self.idx)

        self.removals = []
        self.counter = None
        self.current_graph = None
        self.current_lcc_size = None
        self.start_lcc_size = None
        self.threshold = None
        self.node_ids = None

    def reset(self):

        if self.counter is None:
            self.counter = 0
        else:
            self.counter += 1

        done = self.counter >= (len(self.dataset) - 1)  # we are at the last graph in the dataset
        # if self.counter >= len(self.dataset) - 1:
        #     self.counter = 0
        #     if self.shuffle:
        #         # new epoch, we re shuffle the indices
        #         np.random.shuffle(self.idx)
        #     done = True
        # else:
        #     done = False

        self.set_graph_as_current()
        self.removals = []
        return self.current_graph, done

    def step(self, action):
        """ Takes an action in the current graph. Removes action node. Then computes the new lcc size and derives the
        score as the difference in size of the lcc before and after. Add the node id removed to the list of removals.
        We are done if the new current_lcc_size is smaller or equal to the threshold for this graph.
        :param action: Should be an integer that represents the index in the graph feature matrix of the node to remove.
        :return: new_state (new graph): pyg.Data, reward: int, done: bool
        """
        # remove the node from the graph and update the removal list for potential reinsertion
        new_graph = remove_node_from_pyg_graph(self.current_graph, action).to(device)
        node_id = self.node_ids.pop(action)
        self.removals.append(node_id)

        # compute the reward
        new_lcc_size = get_largest_connected_component(new_graph).num_nodes
        reward = (self.current_lcc_size - new_lcc_size)/self.start_lcc_size  # normalize the reward to prevent explosion
        # reward = self.current_lcc_size - new_lcc_size
        # update the env state
        self.current_graph = new_graph
        self.current_lcc_size = new_lcc_size

        done = self.current_lcc_size <= self.threshold

        return new_graph, reward, done

    def __len__(self):
        return len(self.dataset)
