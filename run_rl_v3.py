# We want to train one model per graph
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
from rl import DatasetEnvModified, Memory, GraphEnv
from run_next_node_dm import pretrain
from run_rl_v2 import train


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
        env = DatasetEnvModified([graph], threshold_fn=lcc_threshold_fn, shuffle=False)
        eval_loader = DataLoader([graph], batch_size=1, shuffle=False)
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
        no_reinsert, reinsert = train(args, online_model, target_model, env, eval_loader, optimizer, scheduler)
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
    # parser.add_argument("--pretrain", action="store_true")
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.num_steps = int(args.num_steps)
    main(args)