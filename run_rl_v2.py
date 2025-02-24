import numpy as np
from run_next_node_dm import get_parser as get_pre_parser
from run_next_node_dm import evaluate
import argparse
from data import GDMTrainingData, GDMTestData
from torch_geometric.loader import DataLoader
from models.models import get_model, GDM_GATArgs
from models.gdm.GDM_GAT import GAT_Model
from global_settings import device, threshold, lcc_threshold_fn
from tqdm import tqdm
import torch
from util import get_largest_connected_component, remove_node_from_pyg_graph
from smoothed_value import SmoothedValue
from reinsertion import reinsertion
from rl import DatasetEnvModified, Memory
from run_gdm import main as run_gdm_main
from run_gdm import get_parser as get_gdm_parser


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

def train(args, online_model, target_model, train_env, eval_loader, optimizer, scheduler = None):
    memory = Memory(args.replay_memory_capacity)
    epsilon = 1.0
    steps = 0
    total_loss = SmoothedValue(fmt='{avg:.3e}')
    total_nodes_removed = SmoothedValue(fmt='{avg:.2f}')
    total_nodes_removed_with_reinsert = SmoothedValue(fmt='{avg:.2f}')
    pbar = tqdm(range(args.num_steps))
    per_epoch_steps = args.num_steps // args.epochs

    while steps < args.num_steps:
        online_model.train()
        target_model.train()
        train_env.swap_graph()
        state = train_env.get_state()

        action = get_action(state, online_model, epsilon, train_env)
        next_state, reward, done = train_env.step(action)

        memory.push(state.clone().to("cpu"), next_state.clone().to("cpu"), action, reward)

        if done:
            # we gather the stats
            removals = train_env.get_removals()
            total_nodes_removed.update(len(removals), n=1)

            if args.reinsert:
                # we do reinsertion
                start_state = train_env.get_current_graph_start_state()
                cur_state = train_env.get_state()
                new_data, reinserted_nodes, sub_history = reinsertion(start_state, cur_state,
                                                                      train_env.get_current_threshold(),
                                                                      removals, train_env.get_start_lcc_size())
                post_reinsertion_removals = [node_id for node_id in removals if node_id not in reinserted_nodes]
                nodes_removed_after_reinsertion = len(post_reinsertion_removals)
                total_nodes_removed_with_reinsert.update(nodes_removed_after_reinsertion, n=1)

                pbar.set_description(
                    f"Step: {steps} | Loss: {total_loss} | Removed: {total_nodes_removed}, "
                    f"reinsert: {total_nodes_removed_with_reinsert}")
            else:
                pbar.set_description(
                    f"Step: {steps} | Loss: {total_loss} | Removed: {total_nodes_removed}")
        else:
            pbar.set_description(
                f"Step: {steps} | Loss: {total_loss} | Removed: {total_nodes_removed}")


        if steps > args.initial_exploration and len(memory) > args.batch_size:
            epsilon -= 0.00005
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
            total_nodes_removed_pre_reinsert, total_nodes_removed_post_reinsert = evaluate(online_model, eval_loader,
                                                                                           args.reinsert,
                                                                                           lcc_threshold_fn)
            if args.reinsert:
                print(
                    f"Eval Nodes Removed: {total_nodes_removed_pre_reinsert.global_avg:.2f} | with reinsert: {total_nodes_removed_post_reinsert.global_avg:.2f}")
            else:
                print(f"\nEval Nodes Removed: {total_nodes_removed_pre_reinsert.global_avg:.2f}")

            print(
                f"Epoch global metrics: Loss  {total_loss.global_avg:.4f} | Nodes removed: {total_nodes_removed.global_avg:.2f} | with reinsert: {total_nodes_removed_with_reinsert.global_avg:.2f}")
        # else:
        #     print(
        #         f"Epoch global metrics: Loss  {total_loss.global_avg:.4f} | Nodes removed: {total_nodes_removed.global_avg:.2f} | with reinsert: {total_nodes_removed_with_reinsert.global_avg:.2f}")
        steps += 1
        pbar.update(1)

    total_nodes_removed_pre_reinsert, total_nodes_removed_post_reinsert = evaluate(online_model, eval_loader,
                                                                                   args.reinsert, lcc_threshold_fn)
    return total_nodes_removed_pre_reinsert, total_nodes_removed_post_reinsert


def main(args):
    args.use_sigmoid = False  # doesn't work with what we are trying to do

    train_dataset = GDMTrainingData()
    test_dataset = GDMTestData(size=args.test_size, test_dataset=args.test_set)  # only want the corruption network for now
    # train_env = DatasetEnv(train_dataset, threshold_fn=lcc_threshold_fn, shuffle=True)
    train_env = DatasetEnvModified(test_dataset, threshold_fn=lcc_threshold_fn, shuffle=True)

    eval_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)
    model_args = GDM_GATArgs(conv_layers=[40, 30], heads=[10, 10], fc_layers=[100, 100], use_sigmoid=args.use_sigmoid)

    online_model = get_model(args.model, in_channel=train_dataset.num_features, out_channel=args.hidden_dim,
                      num_classes=train_dataset.num_classes, num_layers=args.num_layers, gdm_args=model_args).to(device)
    target_model = get_model(args.model, in_channel=train_dataset.num_features, out_channel=args.hidden_dim,
                      num_classes=train_dataset.num_classes, num_layers=args.num_layers, gdm_args=model_args).to(device)

    if args.pretrain:
        print("GDM pre-training")
        epochs = args.epochs
        lr = args.lr
        wd = args.wd
        test_size = args.test_size
        evaluate_every = args.evaluate_every
        args.epochs = 1
        args.lr = 0.003
        args.wd = 1e-5
        args.test_size = 1
        args.evaluate_every = args.epochs * 2

        online_model = run_gdm_main(args)
        target_model.load_state_dict(online_model.state_dict())

        args.epochs = epochs
        args.lr = lr
        args.wd = wd
        args.test_size = test_size
        args.evaluate_every = evaluate_every
        print("Done with GDM pre-training")

    optimizer = torch.optim.Adam(online_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/10)
    # scheduler = None
    no_reinsert, reinsert = train(args, online_model, target_model, train_env, eval_loader, optimizer, scheduler)
    return no_reinsert, reinsert


def get_parser():
    parser = get_pre_parser()
    parser.add_argument('--replay_memory_capacity', type=int, default=4000)
    parser.add_argument("--initial_exploration", type=int, default=2000)
    parser.add_argument("--max_epsilon", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--update_target", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--num_steps", default=1e5, type=int)
    parser.add_argument("--pretrain", action="store_true")
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.num_steps = int(args.num_steps)
    main(args)