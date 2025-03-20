from run_next_node_dm import get_parser as get_pre_parser
from run_next_node_dm import pretrain
import torch
from data import GDMTrainingData, GDMTestData
from torch_geometric.loader import DataLoader
from models.models import get_model, GDM_GATArgs
from models.gdm.GDM_GAT import GAT_Model
from global_settings import device, lcc_threshold_fn
from tqdm import tqdm
from rl import Memory
import numpy as np
from models.link_prediction import LinkPredictor
from run_rl_v3 import get_parser as get_pre_parser
from util import get_largest_connected_component, add_edge_to_pyg_graph
from smoothed_value import SmoothedValue
import random
import copy
import itertools
from run_next_node_dm import evaluate as evaluate_dismantler
import matplotlib.pyplot as plt


edge_add_count_fn = lambda x: max(int(x.num_edges*0.001), 1)


class EdgeGraphEnv:
    """
    Stores the environment for one graph
    """
    def __init__(self, graph, threshold_fn, dismantler, alpha: float = 1., edge_count_to_add: int = 50):
        self.threshold_fn = threshold_fn
        self.starting_graph = graph
        self.start_lcc_size = get_largest_connected_component(graph).num_nodes
        self.threshold = self.threshold_fn(self.start_lcc_size)
        self.dismantler = dismantler
        self.dismantler.eval()
        self.current_graph = None
        self.current_removal = None
        self.current_removal_with_reinsert = None
        self.current_lcc_size = None
        self.node_ids = None
        self.edges = None
        self.edges_to_add = None
        self.best_removal = None
        self.best_removal_with_reinsert = None
        self.starting_removal = None
        self.starting_removal_with_reinsert = None
        self.edges_added = []
        self.alpha = alpha
        self.edge_count_to_add = edge_count_to_add
        self.edges_added_count = 0
        self.reset()

    def copy(self):
        return EdgeGraphEnv(self.starting_graph.clone().detach(), self.threshold_fn, self.dismantler,
                            self.alpha, self.edge_count_to_add)

    def get_random_action(self):
        # get a random edge that doesn't already exist
        random_edge = random.sample(tuple(self.edges_to_add), 1)[0]
        return random_edge[0] + (self.current_graph.num_nodes * random_edge[1])

    def get_edges_added(self):
        return copy.deepcopy(self.edges_added)

    def get_current_threshold(self):
        return self.threshold

    def get_start_lcc_size(self):
        return self.start_lcc_size

    def edge_is_in_current_graph(self, edge):
        in_edges = edge in self.edges
        in_added = edge in self.edges_added
        in_to_add = edge in self.edges_to_add
        in_current = False
        for i in range(len(self.current_graph.edge_index[0])):
            if edge == (self.current_graph.edge_index[0][i].item(), self.current_graph.edge_index[1][i].item()):
                in_current = True
        in_starting = False
        for i in range(len(self.starting_graph.edge_index[0])):
            if edge == (self.starting_graph.edge_index[0][i].item(), self.starting_graph.edge_index[1][i].item()):
                in_current = True
        return in_edges, in_added, in_to_add, in_current, in_starting


    def reset(self):
        self.current_graph = self.starting_graph.clone().detach()
        self.current_lcc_size = self.start_lcc_size
        self.node_ids = list(range(self.starting_graph.num_nodes))
        self.edges = set((self.current_graph.edge_index[0][i].item(), self.current_graph.edge_index[1][i].item()) for i in range(len(self.current_graph.edge_index[0])))
        self.edges_to_add = set(itertools.product(range(self.starting_graph.num_nodes), repeat=2)) - self.edges
        self.edges_added_count = 0
        self.edges_added = []
        loader = DataLoader([self.current_graph.clone().detach()], batch_size=1)
        self.starting_removal, self.starting_removal_with_reinsert = evaluate_dismantler(self.dismantler, loader, True, lcc_threshold_fn)
        self.starting_removal = self.starting_removal.global_avg
        self.starting_removal_with_reinsert = self.starting_removal_with_reinsert.global_avg
        self.current_removal = self.starting_removal
        self.current_removal_with_reinsert = self.starting_removal_with_reinsert
        self.best_removal = self.starting_removal
        self.best_removal_with_reinsert = self.starting_removal_with_reinsert

    def get_state(self):
        return self.current_graph.clone().detach()

    def get_start_state(self):
        return self.starting_graph.clone().detach()

    def step(self, action):
        source = action % self.current_graph.num_nodes
        target = action // self.current_graph.num_nodes

        # mask = LinkPredictor.construct_edge_mask(self.current_graph.edge_index, num_nodes=self.current_graph.num_nodes)
        # mask_works = not mask[source + target * self.current_graph.num_nodes].item()
        # if not mask_works:
        #     print("Edge already in graph")
        new_graph = add_edge_to_pyg_graph(self.current_graph, source, target, device_to_use=device)
        # mask = LinkPredictor.construct_edge_mask(self.current_graph.edge_index, num_nodes=self.current_graph.num_nodes)
        # mask_works = mask[source + target * self.current_graph.num_nodes].item()

        # found = False
        # for i in range(len(self.current_graph.edge_index[0])):
        #     if (source, target) == (new_graph.edge_index[0][i].item(), new_graph.edge_index[1][i].item()):
        #         found = True
        # if not found:
        #     print(f"AYAYAYA: {action} | {(source, target)}")
        # we don't add the reverse edge for now
        # new_graph = add_edge_to_pyg_graph(new_graph, target, source, device_to_use=device)

        new_loader = DataLoader([new_graph.clone().detach()], batch_size=1)
        new_removed, new_with_reinsert = evaluate_dismantler(self.dismantler, new_loader, True, lcc_threshold_fn)
        new_removed = new_removed.global_avg
        new_with_reinsert = new_with_reinsert.global_avg
        # if new_removed < new_with_reinsert:
        #     print("BUUUUUUUUUUUUG")
        reward_removed = new_removed - self.current_removal
        reward_with_reinsert = new_with_reinsert - self.current_removal_with_reinsert

        reward = reward_removed + self.alpha*reward_with_reinsert
        reward = reward*0.5


        # we recompute the lcc size as it could have changed
        self.current_lcc_size = get_largest_connected_component(new_graph).num_nodes
        # we update the current_removed with and without reinsert and the best if needed
        self.current_removal = new_removed
        self.current_removal_with_reinsert = new_with_reinsert
        if self.current_removal > self.best_removal:
            self.best_removal = self.current_removal
        if self.current_removal_with_reinsert > self.best_removal_with_reinsert:
            self.best_removal_with_reinsert = self.current_removal_with_reinsert
        # we add the edge to the set of current edges and the set of edges added while removing it from the set of
        # edges we can add
        # for now, we only had edges in one direction
        # if (source, target) not in self.edges_to_add:
        #     print((source, target))
        #     if (source, target) in self.edges:
        #         print("Edge already in the graph")
        #     if (source, target) in self.edges_added:
        #         print("Edge was already added to the graph")
        #         print(self.edges_added)
        #     check if the mask worked properly
            # mask = LinkPredictor.construct_edge_mask(self.current_graph.edge_index, num_nodes=self.current_graph.num_nodes)
            # print(mask[source + target*self.current_graph.num_nodes])

        # we now update the graph
        self.current_graph = new_graph.clone().detach()
        self.edges.add((source, target))
        self.edges_added.append((source, target))
        self.edges_to_add.remove((source, target))
        self.edges_added_count += 1
        # if source != target:
        #     don't want to add self edges twice
            # self.edges.add((target, source))
            # self.edges_added.append((target, source))
            # self.edges_to_add.remove((target, source))
            # self.edges_added_count += 1

        # print("B", len(self.edges), len(self.edges_added), len(self.edges_to_add))
        done = self.edges_added_count >= self.edge_count_to_add  # we divide by two because we're adding two edges at a time since undirected
        return new_graph, reward, done


def evaluate(graph_env, model):
    model.eval()
    graph_env.reset()
    done = False
    with torch.no_grad():
        while not done:
            state = graph_env.get_state()
            action = model.get_action(state)
            next_state, reward, done = graph_env.step(action)
    return (graph_env.starting_removal, graph_env.starting_removal_with_reinsert, graph_env.best_removal,
            graph_env.best_removal_with_reinsert)


def load_pretrained_model(args, graph_file_name):
    pass


def get_action(states, online_net, epsilon, env):
    # action = online_net.get_action(states)
    # rand = True
    if np.random.rand() <= epsilon:
        action = env.get_random_action()
    else:
        # rand =False
        action = online_net.get_action(states)
    i, j = action % states.num_nodes, action // states.num_nodes
    # if env.edge_is_in_current_graph((i,j)) != (False, False, True, False, False):
    #     print(env.edge_is_in_current_graph((i, j)))
    #     print("Something very wrong")
    #     print(f"Random action was selected: {rand}")
    return action


def update_target_model(online_model, target_model, tau):
    # Target <- Net
    # target_net.load_state_dict(online_net.state_dict())
    for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
        target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)


def train(args, online_model, target_model, train_env, optimizer, scheduler = None):
    no_reverse_edge = []
    self_edges = []
    for edge in train_env.edges:
        if (edge[1], edge[0]) not in train_env.edges:
            no_reverse_edge.append(edge)
        if edge[0] == edge[1]:
            self_edges.append(edge)
    print(f"No reverse edge: {no_reverse_edge}")
    print(f"Self edges: {self_edges}")
    memory = Memory(args.replay_memory_capacity)
    epsilon = 1.0
    steps = 0
    total_loss = SmoothedValue(fmt='{avg:.3e}')
    max_diff = 0
    max_diff_with_reinsert = 0
    rewards = SmoothedValue(window_size=20, fmt='{avg:.3f}')
    pbar = tqdm(range(args.num_steps))
    per_epoch_steps = (args.num_steps - args.initial_exploration) // args.epochs
    latest_diff = 0
    latest_diff_with_reinsert = 0
    best_model_state = None

    episode_diffs = []
    episode_diffs_reinsert = []
    episode_rewards = []  # to record cumulative reward per episode
    episode_reward = 0.0  # accumulator for the current episode

    while steps < args.num_steps:
        online_model.train()
        target_model.train()
        state = train_env.get_state()
        action = get_action(state, online_model, epsilon, train_env)
        # print(f"Action: {action} | ({action % train_env.current_graph.num_nodes}, {action // train_env.current_graph.num_nodes})")
        next_state, reward, done = train_env.step(action)

        episode_reward += reward
        rewards.update(reward, n=1)

        memory.push(state.clone().to("cpu"), next_state.clone().to("cpu"), action, reward)

        if done:
            # we gather the stats
            best_removal, best_removal_with_reinsert = train_env.best_removal, train_env.best_removal_with_reinsert
            best_diff = best_removal - train_env.starting_removal
            best_diff_with_reinsert = best_removal_with_reinsert - train_env.starting_removal_with_reinsert
            if best_diff > max_diff:
                max_diff = best_diff
            if best_diff_with_reinsert > max_diff_with_reinsert:
                max_diff_with_reinsert = best_diff_with_reinsert
                best_model_state = online_model.state_dict()

            latest_diff = best_diff
            latest_diff_with_reinsert = best_diff_with_reinsert

            episode_diffs.append(best_diff)
            episode_diffs_reinsert.append(best_diff_with_reinsert)
            episode_rewards.append(episode_reward)
            episode_reward = 0.0  # reset the accumulator

            train_env.reset()
            if args.reinsert:
                pbar.set_description(
                    f"Step: {steps} | Eps: {epsilon:.2f} | Reward: {rewards} | Loss: {total_loss} | Base (No-R/R) {train_env.starting_removal}/{train_env.starting_removal_with_reinsert} "
                    f"Diff: {latest_diff}--{max_diff}, "
                    f"Diff-R: {latest_diff_with_reinsert}--{max_diff_with_reinsert} ")
            else:
                pbar.set_description(
                    f"Step: {steps} | Eps: {epsilon:.2f} | Reward: {rewards} | Loss: {total_loss} | Base (No-R) {train_env.starting_removal} "
                    f"Diff: {latest_diff}--{max_diff}")
        else:
            if args.reinsert:
                pbar.set_description(
                    f"Step: {steps} | Eps: {epsilon:.2f} | Reward: {rewards} | Loss: {total_loss} | Base (No-R/R) {train_env.starting_removal}/{train_env.starting_removal_with_reinsert} "
                    f"Diff: {latest_diff}--{max_diff}, "
                    f"Diff-R: {latest_diff_with_reinsert}--{max_diff_with_reinsert} ")
            else:
                pbar.set_description(
                    f"Step: {steps} | Eps: {epsilon:.2f} | Reward: {rewards} | Loss: {total_loss} | Base (No-R) {train_env.starting_removal} "
                    f"Diff: {latest_diff}--{max_diff}")

        if steps > args.initial_exploration and len(memory) > args.batch_size:
            # epsilon -= 0.00005
            # epsilon -= 0.0005
            epsilon -= 0.005

            epsilon = max(epsilon, args.max_epsilon)

            batch = memory.sample(args.batch_size)
            loss = GAT_Model.train_model(online_model, target_model, optimizer, batch, args.gamma)

            total_loss.update(loss, n=args.batch_size)

            if steps % args.update_target == 0:
                update_target_model(online_model, target_model, args.tau)

            if scheduler is not None:
                # we need to decrease the lr regularly, we do so by simulating how often we would do it with epochs
                if steps % per_epoch_steps == 0:
                    scheduler.step()

            if (steps) % (args.evaluate_every * per_epoch_steps) == 0:
                (val_starting_removal, val_starting_removal_with_reinsert, val_best_removal,
                 val_best_removal_with_reinsert) = evaluate(train_env.copy(), online_model)
                if args.reinsert:
                    print(
                        f"Base (No-R/R): {val_starting_removal}/{val_starting_removal_with_reinsert} | "
                        f"Diff: {val_best_removal-val_starting_removal} | "
                        f"Diff-R: {val_best_removal_with_reinsert-val_best_removal_with_reinsert} | ")
                else:
                    print(
                        f"Base (No-R): {val_starting_removal} | "
                        f"Diff: {val_best_removal-val_starting_removal} | ")
                print(
                    f"Epoch global metrics: Loss  {total_loss.global_avg:.4f}")

        steps += 1
        pbar.update(1)

    plt.figure(figsize=(10, 6))
    plt.plot(episode_diffs, label='Best Diff (No Reinserts)', marker='o')
    if args.reinsert:
        plt.plot(episode_diffs_reinsert, label='Best Diff (With Reinserts)', marker='o')

    # Compute running window mean for best differences
    window_size = 20  # adjust the window size as needed
    if len(episode_diffs) >= window_size:
        running_diff = np.convolve(episode_diffs, np.ones(window_size) / window_size, mode='valid')
        # Plot the running mean line (aligning the x-axis appropriately)
        plt.plot(range(window_size - 1, len(episode_diffs)), running_diff,
                 label=f'Running Mean ({window_size}) No-R', linestyle='--', color='black')

    if len(episode_diffs_reinsert) >= window_size:
        running_diff_with_reinsert = np.convolve(episode_diffs_reinsert, np.ones(window_size) / window_size, mode='valid')
        # Plot the running mean line (aligning the x-axis appropriately)
        plt.plot(range(window_size - 1, len(episode_diffs_reinsert)), running_diff_with_reinsert,
                 label=f'Running Mean ({window_size}) R', linestyle='--', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Best Difference')
    plt.title('RL Defense Success Throughout Training')
    plt.legend()
    plt.savefig('rl_defense_training_success.png')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards, label='Cumulative Episode Reward', marker='o')
    # Compute running window mean for rewards
    if len(episode_rewards) >= window_size:
        running_rewards = np.convolve(episode_rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(range(window_size - 1, len(episode_rewards)), running_rewards,
                 label=f'Running Mean ({window_size})', linestyle='--', color='black')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Reward Signal Throughout Training')
    plt.legend()
    plt.savefig('rl_defense_training_rewards.png')
    plt.show()
    pbar.close()
    (val_starting_removal, val_starting_removal_with_reinsert, val_best_removal,
     val_best_removal_with_reinsert) = evaluate(train_env.copy(), online_model)
    print(
        f"Final Performance: Base (No-R/R): {val_starting_removal}/{val_starting_removal_with_reinsert} | "
        f"Diff: {val_best_removal - val_starting_removal} | "
        f"Diff-R: {val_best_removal_with_reinsert - val_starting_removal_with_reinsert} | ")
    print(
        f"Best performance throughout training | Diff (No-R/R): {max_diff}/{max_diff_with_reinsert}")
    online_model.load_state_dict(best_model_state)
    (cache_val_starting_removal, cache_val_starting_removal_with_reinsert, cache_val_best_removal,
     cache_val_best_removal_with_reinsert) = evaluate(train_env.copy(), online_model)
    print(
        f"Best cached model performance| Base (No-R/R): {cache_val_starting_removal}/{cache_val_starting_removal_with_reinsert} | "
        f"Diff: {cache_val_best_removal - cache_val_starting_removal} | "
        f"Diff-R: {cache_val_best_removal_with_reinsert - cache_val_starting_removal_with_reinsert} | ")

    return (max(cache_val_best_removal - cache_val_starting_removal, val_best_removal - val_starting_removal, max_diff),
            max(cache_val_best_removal_with_reinsert - cache_val_starting_removal_with_reinsert,
                val_best_removal_with_reinsert - val_starting_removal_with_reinsert, max_diff_with_reinsert))


def main(args):
    train_dataset = GDMTrainingData()
    test_dataset = GDMTestData(size=args.test_size, test_dataset=args.test_set)  # only want the corruption network for now

    embedding_dimension = 30*10  # last conv layer * num heads

    if args.model_folder is None:
        # no pretrained RL-based models, instead we just quickly train a GDM dismantler
        attack_model, _ = pretrain(args, epochs=10)
    best_no_reinserts = []
    best_with_reinserts = []
    for i, graph in enumerate(test_dataset):
        edge_count_to_add_val = edge_add_count_fn(graph)
        print(f"Running graph {i} with {graph.num_nodes} nodes and {graph.num_edges} edges | Edges to add: {edge_count_to_add_val}")
        if args.model_folder is not None:
            attack_model = load_pretrained_model(args, test_dataset.files[i])
        model_args = GDM_GATArgs(conv_layers=[40, 30], heads=[10, 10], fc_layers=[100, 100], use_sigmoid=True,
                                 for_rl=False)
        defender_embedder = get_model(args.model, in_channel=train_dataset.num_features,
                                      out_channel=args.hidden_dim,
                                      num_classes=train_dataset.num_classes, num_layers=args.num_layers,
                                      gdm_args=model_args).to(device)
        online_model = LinkPredictor(embedding_model=defender_embedder, embedding_size=embedding_dimension).to(device)
        target_model = LinkPredictor(embedding_model=defender_embedder, embedding_size=embedding_dimension).to(device)
        optimizer = torch.optim.SGD(online_model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/10)
        train_env = EdgeGraphEnv(graph, lcc_threshold_fn, attack_model, alpha=1, edge_count_to_add=edge_count_to_add_val)
        best_diff, best_diff_with_reinsert = train(args, online_model, target_model, train_env, optimizer, scheduler)
        best_no_reinserts.append(best_diff)
        best_with_reinserts.append(best_diff_with_reinsert)
        print(f"Graph {i + 1}/{len(test_dataset)}: Best Diff: {best_diff} | Best With Reinserts: {best_diff_with_reinsert}")
    print(f"No reinserts: {best_no_reinserts}")
    print(f"With reinserts: {best_with_reinserts}")
    print(f"{len(test_dataset)} graphs")
    print(f"Sum no reinserts: {sum(best_no_reinserts)}")
    print(f"Sum with reinserts: {sum(best_with_reinserts)}")
    # return no_reinsert, reinsert



def get_parser():
    parser = get_pre_parser()
    parser.add_argument("--model_folder", type=str, default=None, help="path to folder of pretrained attack models (one per graph since assumed they were trained with RL)")
    # parser.add_argument("--edges_to_add", type=int, default=10, help="number of edges to add")
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)



# ways of training it:
# - RL: straightforward-ish but not guaranteed to work
# - Direct optimization: Try every change in the adjacency matrix and see which one mitigates the final lcc destruction the most
# - Approximate optimization: Try every change in the adjacency matrix and see which one mitigates the change in lcc size by the removal of the next node

# now that RL works for attacking, we can try it for defending
