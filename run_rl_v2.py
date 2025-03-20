# train one model for all graphs, each graph is being dismantled simultaneously and is reset when it is done
import numpy as np
from run_next_node_dm import get_parser as get_pre_parser
from run_next_node_dm import evaluate
from data import GDMTrainingData, GDMTestData
from torch_geometric.loader import DataLoader
from models.models import get_model, GDM_GATArgs
from models.gdm.GDM_GAT import GAT_Model
from global_settings import device, lcc_threshold_fn
from tqdm import tqdm
import torch
from smoothed_value import SmoothedValue
from reinsertion import reinsertion
from rl import DatasetEnvModified, Memory
from run_next_node_dm import pretrain


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
    total_nodes_removed = SmoothedValue(window_size=1, fmt='{avg:.2f}')
    total_nodes_removed_with_reinsert = SmoothedValue(window_size=1, fmt='{avg:.2f}')
    rewards = SmoothedValue(window_size=20, fmt='{avg:.3f}')
    pbar = tqdm(range(args.num_steps))
    per_epoch_steps = (args.num_steps-args.initial_exploration) // args.epochs
    latest_removals = [np.inf]*len(train_env)
    latest_removals_with_reinsert = [np.inf]*len(train_env)
    num_completions = 0
    done_condition = False
    while not done_condition:
        online_model.train()
        target_model.train()
        train_env.swap_graph()
        state = train_env.get_state()

        action = get_action(state, online_model, epsilon, train_env)
        next_state, reward, done = train_env.step(action)
        rewards.update(reward, n=1)

        memory.push(state.clone().to("cpu"), next_state.clone().to("cpu"), action, reward)

        if done:
            if steps > args.initial_exploration:
                num_completions += 1
            # we gather the stats
            removals = train_env.get_removals()
            total_nodes_removed.update(len(removals), n=1)
            latest_removals[train_env.current_graph_idx] = len(removals)
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

                train_env.update_best_removals(len(removals), nodes_removed_after_reinsertion, model=online_model.state_dict())
                latest_removals_with_reinsert[train_env.current_graph_idx] = nodes_removed_after_reinsertion
                pbar.set_description(
                    f"Step: {steps} | Eps: {epsilon:.2f} | Reward: {rewards} | Loss: {total_loss} | {num_completions}/{args.num_completions}; Removed: {np.ma.masked_invalid(latest_removals).sum()}--{train_env.get_best_removals()}, "
                    f"reinsert: {np.ma.masked_invalid(latest_removals_with_reinsert).sum()}--{train_env.get_best_removals_with_reinsert()}")
            else:
                train_env.update_best_removals(len(removals), model=online_model.state_dict())
                pbar.set_description(
                    f"Step: {steps} | Eps: {epsilon:.2f} | Reward: {rewards} | Loss: {total_loss} | {num_completions}/{args.num_completions}; Removed: {np.ma.masked_invalid(latest_removals).sum()}--{train_env.get_best_removals()}")
        else:
            if args.reinsert:
                pbar.set_description(
                    f"Step: {steps} | Eps: {epsilon:.2f} | Reward: {rewards} | Loss: {total_loss} | {num_completions}/{args.num_completions}; Removed:{np.ma.masked_invalid(latest_removals).sum()}--{train_env.get_best_removals()}, "
                    f"reinsert: {np.ma.masked_invalid(latest_removals_with_reinsert).sum()}--{train_env.get_best_removals_with_reinsert()}")
            else:
                pbar.set_description(
                    f"Step: {steps} | Eps: {epsilon:.2f} | Reward: {rewards} | Loss: {total_loss} | {num_completions}/{args.num_completions}; Removed:{np.ma.masked_invalid(latest_removals).sum()}--{train_env.get_best_removals()}")

        if steps > args.initial_exploration and len(memory) > args.batch_size:
            # epsilon -= 0.00005
            epsilon -= 0.0005

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
        if steps >= args.num_steps:
            done_condition = True
        if num_completions >= args.num_completions:
            done_condition = True


    total_nodes_removed_pre_reinsert, total_nodes_removed_post_reinsert = evaluate(online_model, eval_loader,
                                                                                   args.reinsert, lcc_threshold_fn)
    print(f"Final Eval performance: Nodes removed: {total_nodes_removed_pre_reinsert} | With reinsert: {total_nodes_removed_post_reinsert}")
    print(f"Best performance throughout training | Removal: {train_env.get_best_removals()} | With reinsert: {(train_env.get_best_removals_with_reinsert())}")
    if train_env.best_model is not None:
        online_model.load_state_dict(train_env.best_model)
    best_pre_reinsert, best_post_reinsert = evaluate(online_model, eval_loader, args.reinsert, lcc_threshold_fn)
    print(f"Best cached model performance| Removal: {best_pre_reinsert} | With reinsert: {best_post_reinsert}")

    return min(min(total_nodes_removed_pre_reinsert.global_avg, best_pre_reinsert.global_avg), train_env.get_best_removals()), min(min(total_nodes_removed_post_reinsert.global_avg, best_post_reinsert.global_avg), train_env.get_best_removals_with_reinsert())


def main(args):
    args.use_sigmoid = False  # doesn't work with what we are trying to do

    train_dataset = GDMTrainingData()
    test_dataset = GDMTestData(size=args.test_size, test_dataset=args.test_set)
    # train_env = DatasetEnv(train_dataset, threshold_fn=lcc_threshold_fn, shuffle=True)
    train_env = DatasetEnvModified(test_dataset, threshold_fn=lcc_threshold_fn, shuffle=False)

    eval_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False)
    model_args = GDM_GATArgs(conv_layers=[40, 30], heads=[10, 10], fc_layers=[100, 100], use_sigmoid=False, for_rl=True)

    online_model = get_model(args.model, in_channel=train_dataset.num_features, out_channel=args.hidden_dim,
                      num_classes=train_dataset.num_classes, num_layers=args.num_layers, gdm_args=model_args).to(device)
    target_model = get_model(args.model, in_channel=train_dataset.num_features, out_channel=args.hidden_dim,
                      num_classes=train_dataset.num_classes, num_layers=args.num_layers, gdm_args=model_args).to(device)

    if args.pretrain:
        model, args = pretrain(args)
        online_model.load_state_dict(model.state_dict())
        target_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.SGD(online_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr/10)
    # scheduler = None
    no_reinsert, reinsert = train(args, online_model, target_model, train_env, eval_loader, optimizer, scheduler)
    return no_reinsert, reinsert


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
    # parser.add_argument("--pretrain", action="store_true")
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.num_steps = int(args.num_steps)
    main(args)