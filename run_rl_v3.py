# We want to train one model per graph
import copy
import os
import numpy as np
from run_next_node_dm import get_parser as get_pre_parser
from data import GDMTrainingData, GDMTestData
from torch_geometric.loader import DataLoader
from models.models import get_model, GDM_GATArgs
from models.gdm.GDM_GAT import GAT_Model
from global_settings import device, lcc_threshold_fn
from tqdm import tqdm
import torch
from smoothed_value import SmoothedValue
from reinsertion import reinsertion
from rl import Memory, GraphEnv
from run_next_node_dm import pretrain
from util import get_largest_connected_component, remove_node_from_pyg_graph


def get_action(states, online_net, epsilon, env):
    # action = online_net.get_action(states)
    if np.random.rand() <= epsilon:
        return env.get_random_action()
    else:
        action = online_net.get_action(states)
        return action


def update_target_model(online_model, target_model, tau):
    # Target <- Net
    # target_net.load_state_dict(online_net.state_dict())
    for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
        target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)



def evaluate(model, graph, lcc_threshold_fn):
    model.eval()
    with torch.no_grad():
        new_data = graph.clone()
        start_lcc_size = get_largest_connected_component(new_data).num_nodes
        current_lcc_size = start_lcc_size
        lcc_threshold = lcc_threshold_fn(start_lcc_size)
        nodes_removed = 0
        node_ids = list(range(new_data.num_nodes))
        removals = []
        removals_with_reinsert = []
        while current_lcc_size > lcc_threshold:
            pred = model(new_data).squeeze()
            # now we got to remove the node we predicted out
            node_idx = pred.argmax(dim=0)
            new_data = remove_node_from_pyg_graph(new_data, node_idx).to(device)
            node_id = node_ids.pop(node_idx)
            nodes_removed += 1
            removals.append(node_id)
            current_lcc_size = get_largest_connected_component(new_data).num_nodes

        new_data, reinserted_nodes, sub_history = reinsertion(graph.clone(), new_data, lcc_threshold, removals, start_lcc_size)
        removals_with_reinsert = [node_id for node_id in removals if node_id not in reinserted_nodes]

    return removals, removals_with_reinsert



def train(args, online_model, target_model, train_env, graph, optimizer, scheduler = None, file_basename=None):
    memory = Memory(args.replay_memory_capacity)
    epsilon = 1.0
    steps = 0
    total_loss = SmoothedValue(fmt='{avg:.3e}')
    total_nodes_removed = SmoothedValue(window_size=1, fmt='{avg:.2f}')
    total_nodes_removed_with_reinsert = SmoothedValue(window_size=1, fmt='{avg:.2f}')
    rewards = SmoothedValue(window_size=20, fmt='{avg:.3f}')
    pbar = tqdm(range(args.num_steps))
    per_epoch_steps = (args.num_steps-args.initial_exploration) // args.epochs
    latest_removals = np.inf
    latest_removals_with_reinsert = np.inf
    best_removal = np.inf
    best_removals_with_reinsert = np.inf
    best_removal_ids = None
    best_removals_with_reinsert_ids = None
    best_removal_model = None
    best_removal_with_reinsert_model = None
    best_eval_removals = np.inf
    best_eval_removals_ids = []
    best_eval_removals_with_reinsert = np.inf
    best_eval_removals_with_reinsert_ids = []
    best_eval_removals_model = None
    best_eval_removals_with_reinsert_model = None
    num_completions = 0
    done_condition = False
    state = train_env.reset()
    while not done_condition:
        online_model.train()
        target_model.train()

        action = get_action(state, online_model, epsilon, train_env)
        next_state, reward, done = train_env.step(action)
        rewards.update(reward, n=1)

        memory.push(state.clone().to("cpu"), next_state.clone().to("cpu"), action, reward)
        state = next_state
        if done:
            if steps > args.initial_exploration and epsilon <= args.max_epsilon:
                num_completions += 1
            # we gather the stats
            removals = train_env.get_removals()
            total_nodes_removed.update(len(removals), n=1)
            latest_removals = len(removals)
            if len(removals) < best_removal:
                best_removal = len(removals)
                best_removal_ids = copy.deepcopy(removals)
                best_removal_model = copy.deepcopy(online_model.state_dict())
            if args.reinsert:
                # we do reinsertion
                start_state = train_env.get_start_state()
                cur_state = train_env.get_state()
                new_data, reinserted_nodes, sub_history = reinsertion(start_state, cur_state,
                                                                      train_env.get_current_threshold(),
                                                                      removals, train_env.get_start_lcc_size())
                post_reinsertion_removals = [node_id for node_id in removals if node_id not in reinserted_nodes]
                nodes_removed_after_reinsertion = len(post_reinsertion_removals)
                total_nodes_removed_with_reinsert.update(nodes_removed_after_reinsertion, n=1)
                latest_removals_with_reinsert = nodes_removed_after_reinsertion
                # train_env.update_best_removals(removals, post_reinsertion_removals, model=online_model.state_dict())
                if nodes_removed_after_reinsertion < best_removals_with_reinsert:
                    best_removals_with_reinsert = nodes_removed_after_reinsertion
                    best_removal_with_reinsert_model = copy.deepcopy(online_model.state_dict())
                    best_removals_with_reinsert_ids = copy.deepcopy(post_reinsertion_removals)
                pbar.set_description(
                    f"Step: {steps} | Eps: {epsilon:.2f} | Reward: {rewards} | Loss: {total_loss} | {num_completions}/{args.num_completions}; Removed: {latest_removals}--{best_removal}, "
                    f"reinsert: {latest_removals_with_reinsert}--{best_removals_with_reinsert}")
            else:
                # train_env.update_best_removals(removals, model=online_model.state_dict())
                pbar.set_description(
                    f"Step: {steps} | Eps: {epsilon:.2f} | Reward: {rewards} | Loss: {total_loss} | {num_completions}/{args.num_completions}; Removed: {latest_removals}--{best_removal}")
            state = train_env.reset()
            #  we evaluate
            eval_removals, eval_removals_with_reinsert = evaluate(online_model, graph.clone(), lcc_threshold_fn)
            if len(eval_removals) < best_eval_removals:
                best_eval_removals = len(eval_removals)
                best_eval_removals_ids = copy.deepcopy(eval_removals)
                best_eval_removals_model = copy.deepcopy(online_model.state_dict())
            if len(eval_removals_with_reinsert) < best_eval_removals_with_reinsert:
                best_eval_removals_with_reinsert = len(eval_removals_with_reinsert)
                best_eval_removals_with_reinsert_model = copy.deepcopy(online_model.state_dict())
                best_eval_removals_with_reinsert_ids = copy.deepcopy(eval_removals_with_reinsert)
        else:
            if args.reinsert:
                pbar.set_description(
                    f"Step: {steps} | Eps: {epsilon:.2f} | Reward: {rewards} | Loss: {total_loss} | {num_completions}/{args.num_completions}; Removed:{latest_removals}--{best_removal}, "
                    f"reinsert: {latest_removals_with_reinsert}--{best_removals_with_reinsert}")
            else:
                pbar.set_description(
                    f"Step: {steps} | Eps: {epsilon:.2f} | Reward: {rewards} | Loss: {total_loss} | {num_completions}/{args.num_completions}; Removed:{latest_removals}--{best_removal}")

        if steps > args.initial_exploration and len(memory) > args.batch_size:
            epsilon -= 0.00005
            # epsilon -= 0.0005

            epsilon = max(epsilon, args.max_epsilon)

            batch = memory.sample(args.batch_size)
            loss = GAT_Model.train_model_batched(online_model, target_model, optimizer, batch, args.gamma)

            total_loss.update(loss, n=args.batch_size)

            if steps % args.update_target == 0:
                update_target_model(online_model, target_model, args.tau)

            if scheduler is not None:
                # we need to decrease the lr regularly, we do so by simulating how often we would do it with epochs
                if steps % per_epoch_steps == 0:
                    scheduler.step()

            if (steps) % (args.evaluate_every * per_epoch_steps) == 0:
                eval_removals, eval_removals_with_reinsert = evaluate(online_model, graph.clone(), lcc_threshold_fn)
                if len(eval_removals) < best_eval_removals:
                    best_eval_removals = len(eval_removals)
                    best_eval_removals_ids = copy.deepcopy(eval_removals)
                    best_eval_removals_model = copy.deepcopy(online_model.state_dict())
                if len(eval_removals_with_reinsert) < best_eval_removals_with_reinsert:
                    best_eval_removals_with_reinsert = len(eval_removals_with_reinsert)
                    best_eval_removals_with_reinsert_model = copy.deepcopy(online_model.state_dict())
                    best_eval_removals_with_reinsert_ids = copy.deepcopy(eval_removals_with_reinsert)
                print(
                        f"Eval Nodes Removed: {len(eval_removals):.2f}--{best_eval_removals} | with reinsert: {len(eval_removals_with_reinsert):.2f}--{best_eval_removals_with_reinsert}")

                print(
                    f"Epoch global metrics: Loss  {total_loss.global_avg:.4f} | Nodes removed: {total_nodes_removed.global_avg:.2f} | with reinsert: {total_nodes_removed_with_reinsert.global_avg:.2f}")
            # else:
            #     print(
            #         f"Epoch global metrics: Loss  {total_loss.global_avg:.4f} | Nodes removed: {total_nodes_removed.global_avg:.2f} | with reinsert: {total_nodes_removed_with_reinsert.global_avg:.2f}")
        steps += 1
        pbar.update(1)
        if steps >= args.num_steps:
            done_condition = True
        if args.num_completions is not None and num_completions >= args.num_completions:
            done_condition = True

    eval_removals, eval_removals_with_reinsert = evaluate(online_model, graph.clone(), lcc_threshold_fn)
    if len(eval_removals) < best_eval_removals:
        best_eval_removals = len(eval_removals)
        best_eval_removals_ids = copy.deepcopy(eval_removals)
        best_eval_removals_model = copy.deepcopy(online_model.state_dict())
    if len(eval_removals_with_reinsert) < best_eval_removals_with_reinsert:
        best_eval_removals_with_reinsert = len(eval_removals_with_reinsert)
        best_eval_removals_with_reinsert_model = copy.deepcopy(online_model.state_dict())
        best_eval_removals_with_reinsert_ids = copy.deepcopy(eval_removals_with_reinsert)
    pbar.close()
    print(f"Final Eval performance: Nodes removed: {len(eval_removals):.2f}--{best_eval_removals} | with reinsert: {len(eval_removals_with_reinsert):.2f}--{best_eval_removals_with_reinsert}")
    print(f"Best performance throughout training | Removal: {best_removal:.2f} | With reinsert: {best_removals_with_reinsert:.2f}")
    if args.save:
        folder = f"./saved_models/{file_basename}/"
        os.makedirs(folder, exist_ok=True)
        # saving the best eval model and removals
        torch.save(best_eval_removals_model, os.path.join(folder, f"best_eval_model_{args.num_completions}_{args.num_steps}.pt"))
        np.save(os.path.join(folder, f"best_eval_removal_nodes.npy"), np.array(best_eval_removals_ids))
        # saving the best eval model and removals with reinsert
        torch.save(best_eval_removals_with_reinsert_model, os.path.join(folder, f"best_eval_with_reinsert_model_{args.num_completions}_{args.num_steps}.pt"))
        np.save(os.path.join(folder, f"best_eval_removal_with_reinsert_nodes.npy"), np.array(best_eval_removals_with_reinsert_ids))
        # saving the best model and removals
        torch.save(best_removal_model, os.path.join(folder, f"best_model_{args.num_completions}_{args.num_steps}.pt"))
        np.save(os.path.join(folder, f"best_removal_nodes.npy"), np.array(best_removal_ids))
        # saving the best model and removals with reinsert
        torch.save(best_removal_with_reinsert_model, os.path.join(folder, f"best_with_reinsert_model_{args.num_completions}_{args.num_steps}.pt"))
        np.save(os.path.join(folder, f"best_removal_with_reinsert_nodes.npy"), np.array(best_removals_with_reinsert_ids))
    return min(best_removal, best_eval_removals), min(best_removals_with_reinsert, best_eval_removals_with_reinsert)

def main(args):
    args.use_sigmoid = False  # doesn't work with what we are trying to do
    args.evaluate_every = np.inf  # we don't care about evaluating in the middle anymore
    train_dataset = GDMTrainingData()
    test_dataset = GDMTestData(size=args.test_size, test_dataset=args.test_set)
    # train_env = DatasetEnv(train_dataset, threshold_fn=lcc_threshold_fn, shuffle=True)
    # train_env = DatasetEnvModified(test_dataset, threshold_fn=lcc_threshold_fn, shuffle=False)

    # eval_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    model_args = GDM_GATArgs(conv_layers=[40, 30], heads=[10, 10], fc_layers=[100, 100], use_sigmoid=False, for_rl=True)
    best_no_reinserts = []
    best_with_reinserts = []
    for i, graph in enumerate(test_dataset):
        print(f"Running graph {i+1}/{len(test_dataset)} with {graph.num_nodes} nodes")
        if os.path.exists(os.path.join("./saved_models", os.path.splitext(os.path.basename(test_dataset.files[i]))[0], f"best_removal_with_reinsert_nodes.npy")):

            continue
        # env = DatasetEnvModified([graph], threshold_fn=lcc_threshold_fn, shuffle=False)
        env = GraphEnv(graph, lcc_threshold_fn)
        online_model = get_model(args.model, in_channel=train_dataset.num_features, out_channel=args.hidden_dim,
                          num_classes=train_dataset.num_classes, num_layers=args.num_layers, gdm_args=model_args).to(device)
        target_model = get_model(args.model, in_channel=train_dataset.num_features, out_channel=args.hidden_dim,
                          num_classes=train_dataset.num_classes, num_layers=args.num_layers, gdm_args=model_args).to(device)

        if args.pretrain:
            model, args = pretrain(args)
            online_model.load_state_dict(model.state_dict())
            target_model.load_state_dict(model.state_dict())

        optimizer = torch.optim.SGD(online_model.parameters(), lr=args.lr)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/10)
        scheduler = None
        no_reinsert, reinsert = train(args, online_model, target_model, env, graph.clone(), optimizer, scheduler, os.path.splitext(os.path.basename(test_dataset.files[i]))[0])
        best_no_reinserts.append(no_reinsert)
        best_with_reinserts.append(reinsert)
        print(f"Graph {i+1}/{len(test_dataset)}: Best Removals: {no_reinsert} | Best With Reinserts: {reinsert}")
    print(f"No reinserts: {best_no_reinserts}")
    print(f"With reinserts: {best_with_reinserts}")
    print(f"{len(test_dataset)} graphs")
    print(f"Sum no reinserts: {sum(best_no_reinserts)}")
    print(f"Sum with reinserts: {sum(best_with_reinserts)}")
    # return no_reinsert, reinsert


def get_parser():
    parser = get_pre_parser()
    parser.add_argument('--replay_memory_capacity', type=int, default=2000)
    parser.add_argument("--initial_exploration", type=int, default=1000)
    parser.add_argument("--max_epsilon", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--update_target", type=int, default=200)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--num_steps", default=1e5, type=int)
    parser.add_argument("--num_completions", default=None, type=int)
    parser.add_argument("--save", action="store_true", help="Save model")
    # parser.add_argument("--pretrain", action="store_true")
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.num_steps = int(args.num_steps)
    main(args)