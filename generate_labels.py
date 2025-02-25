from torch_geometric.data import Data
from util import get_largest_connected_component, remove_node_from_pyg_graph
import itertools
from global_settings import device, mp_num_cpu
import torch
import multiprocessing as mp
from functools import partial
import random
import math


def evaluate_subset(args):
    """
    Evaluates a single subset by cloning the network, removing the nodes,
    and checking whether the largest connected component (LCC) falls below
    the threshold.

    :param args: Tuple containing (network, subset, lcc_threshold)
    :return: The subset if it passes the threshold condition, else None.
    """
    network, subset, lcc_threshold = args
    network = network.clone()
    removed = []
    # NOTE: Iterating over a set yields an arbitrary order. If order matters, sort the subset.
    for node_idx in subset:
        shifted_node_idx = node_idx
        for already_removed in removed:
            if node_idx > already_removed:
                shifted_node_idx -= 1
        network = remove_node_from_pyg_graph(network, shifted_node_idx, device_to_use=torch.device('cpu'))
        removed.append(node_idx)

    lcc_size = get_largest_connected_component(network).num_nodes
    if lcc_size <= lcc_threshold:
        return subset
    return None


def candidate_worker(successful_subsets):
    """
    Randomly selects a successful subset, removes one random element,
    and returns the candidate as a frozenset.
    """
    ss = random.choice(successful_subsets)
    # Remove one random element without modifying the original subset.
    elem_to_remove = random.choice(tuple(ss))
    candidate = ss - {elem_to_remove}
    return frozenset(candidate)


def sample_candidates(successful_subsets, num_candidates):
    """
    Parallelized random candidate generation.

    Instead of enumerating all candidate subsets and filtering them,
    we repeatedly generate candidates in parallel until we have
    at least num_candidates unique candidates.

    :param successful_subsets: A list of sets (each representing a successful subset).
    :param num_candidates: The desired number of unique candidate subsets.
    :return: A list of candidate subsets (each as a set).
    """
    candidates = set()
    # Create a pool using the desired number of CPU processes.
    pool = mp.Pool(processes=mp_num_cpu)
    count = 0
    warning = False

    try:
        # Continue generating until we have enough unique candidates.
        while len(candidates) < num_candidates:
            # Calculate how many candidates are needed.
            needed = num_candidates - len(candidates)
            # Over-generate candidates; factor of 10 is arbitrary and can be adjusted.
            num_to_generate = max(needed, mp_num_cpu)
            # Prepare a list of arguments; each worker gets the same successful_subsets.
            args = [successful_subsets] * num_to_generate
            # pool.map will run candidate_worker on each argument in parallel.
            results = pool.map(candidate_worker, args)
            # Add the new candidates (as frozensets) to the candidates set.
            candidates.update(results)
            count += mp_num_cpu
            if count >= 1000 * num_candidates and not warning:
                print(
                    f'warning: infinite loop detected | len_ss : {len(successful_subsets)} | candidates: {len(candidates)} | wanted candidates: {num_candidates}')
                warning = True
    finally:
        pool.close()
        pool.join()

    # Convert the unique frozensets back to regular sets.
    # Also, return exactly num_candidates candidates.
    return [set(c) for c in list(candidates)[:num_candidates]]


def non_dist_sample_candidates(successful_subsets, num_candidates):
    """
    Randomly generate a set of candidate subsets of a given size.
    Each candidate is generated by randomly selecting one of the
    successful_subsets (if it is large enough) and then sampling
    subset_size elements from it.

    :param successful_subsets: A list of sets (each representing a successful subset).
    :param subset_size: The desired size for the candidate subset.
    :param num_candidates: How many candidates to generate.
    :return: A list of candidate subsets (each as a set).
    """
    candidates = set()
    count = 0
    warning = False
    # We'll use a frozenset so we can add candidates to a set for uniqueness.
    while len(candidates) < num_candidates:
        # Randomly pick a successful subset
        ss = set(random.choice(successful_subsets))
        elem_to_remove = random.choice(tuple(ss))
        # I know all my subsets are of the same size and I want 1 down
        candidate = ss - {elem_to_remove}
        # candidate = frozenset(random.sample(ss, len(ss)-1))
        candidates.add(frozenset(candidate))
        count += 1
        if count >= 1000*num_candidates and not warning:
            print(f'warning: infinite loop detected | len_ss : {len(successful_subsets)} | candidates: {len(candidates)} | wanted candidates: {num_candidates}')
            warning = True
    # Convert frozensets back to regular sets
    return [set(c) for c in candidates]

def check_candidate(candidate, successful_subsets):
    """
    Checks if a candidate (a tuple of node indices) is eligible for further evaluation.
    The candidate is considered eligible if it is a superset of any of the current
    successful_subsets (i.e. if the candidate contains at least one successful subset).

    :param candidate: Tuple of node indices.
    :param successful_subsets: List of sets (each representing a successful subset).
    :return: The candidate as a set if eligible, else None.
    """
    cand = set(candidate)
    for ss in successful_subsets:
        if cand.issubset(ss):
            return cand
    return None

def count_unique_candidates(successful_subsets):
    unique_candidates = set()
    for ss in successful_subsets:
        for x in ss:
            # Create candidate by removing element x from ss.
            candidate = frozenset(ss - {x})
            unique_candidates.add(candidate)
    return len(unique_candidates)

def generate_keystone_labels(network: Data, lcc_threshold, paths_to_evaluate, return_probs=False):
    """
    # keystones are the nodes that are part of every single removals
    # if there are None, we take the node(s) with the highest number of paths they are a part of
    :param network: the network
    :param lcc_threshold: the threshold for the largest connected component
    :return:
    """
    # check if it already has statistics for the current graph
    if hasattr(network, "y") and network.y is not None and network.y.max() == 1:
        probs = network.y
    else:
        start_lcc = get_largest_connected_component(network).num_nodes
        # lcc_threshold = lcc_threshold_fn(start_lcc)
        successful_subsets = [set(range(network.num_nodes))]
        # we have to compute the statistic ourselves
        node_ids = set(range(network.num_nodes))

        pool = mp.Pool(processes=mp_num_cpu)
        try:
            # print("starting pool")
            # for subset_size in range(1, network.num_nodes + 1):
            for subset_size in range(network.num_nodes-1, 0, -1):
                # Generate candidate subsets (as tuples) of the given size
                # candidates = list(itertools.combinations(node_ids, subset_size))
                # print(f"Pruning candidates from {len(candidates)} to {paths_to_evaluate}")
                # random.shuffle(candidates)
                # # Parallelize the check: determine which candidates are eligible
                # check_func = partial(check_candidate, successful_subsets=successful_subsets)
                # to_check = []
                # candidate_pool = mp.Pool(processes=mp_num_cpu)
                # try:
                #     # imap_unordered returns results as they become available.
                #     for res in candidate_pool.imap_unordered(check_func, candidates):
                #         if res is not None:
                #             to_check.append(res)
                #             if len(to_check) >= paths_to_evaluate:
                #                 print(f"Reached {paths_to_evaluate} passing candidates, stopping early.")
                #                 candidate_pool.terminate()
                #                 break
                # finally:
                #     candidate_pool.close()
                #     candidate_pool.join()
                # print(f"Counting max number of candidates")
                num_candidates = count_unique_candidates(successful_subsets)
                # print(f"Num candidates: {num_candidates}")
                if paths_to_evaluate >= num_candidates:
                    # there are subset_size+1 number of subset_size subsets for a set of size subset_size+1
                    # we have len(successful_subsets) such sets
                    # print(f"Enumerate ({(subset_size+1)*len(successful_subsets)}): computing subsets for size", subset_size)
                    # there are only so many subsets, therefore we can try them all
                    candidates = list(itertools.combinations(node_ids, subset_size))
                    check_func = partial(check_candidate, successful_subsets=successful_subsets)
                    results = pool.map(check_func, candidates)
                    to_check = [res for res in results if res is not None]
                else:
                    # print("Random sampling: computing subsets for size", subset_size)
                    to_check = sample_candidates(successful_subsets, num_candidates=paths_to_evaluate)
                # with mp.Pool(processes=mp_num_cpu) as candidate_pool:
                #     candidate_results = candidate_pool.map(check_func, candidates)
                # Filter out candidates that didn't pass the check (i.e. are None)
                # to_check = [res for res in candidate_results if res is not None]



                # print("computing subsets")
                # subsets = set(itertools.combinations(node_ids, subset_size))
                # to_check = []
                # for subset in subsets:
                #     found = False
                #     for successful_subset in successful_subsets:
                #         # check if we already have a successful subset as part of the subsets
                #         # if successful_subset.issubset(subset):
                #         if set(subset).issubset(successful_subset):
                #             found = True
                #             break
                #     # if not found:
                #     if found:
                #         to_check.append(set(subset))
                if len(to_check) == 0:
                    # we terminate, there is no more possible set of nodes to check
                    break
                else:
                    # print("running tasks")

                    # Prepare tasks: each task is (network, subset, lcc_threshold).
                    data = network.clone().cpu()
                    # print(len(to_check))
                    tasks = [(data, subset, lcc_threshold) for subset in to_check]
                    results = pool.map(evaluate_subset, tasks)
                    # Keep only those subsets that pass the threshold test.
                    to_add = [res for res in results if res is not None]

                    # to_add = []
                    # for subset in to_check:
                    #     new_data = network.clone()
                    #     removed = []
                    #     for node_idx in subset:
                    #         shifted_node_idx = node_idx
                    #         # we need to adjust the node idx. The already removed nodes affect the structure and ordering of the edges
                    #         for already_removed in removed:
                    #             if node_idx > already_removed:
                    #                 # node_idx is greater therefore the already removed was before it in the order, so we need to shift down
                    #                 shifted_node_idx -= 1
                    #         new_data = remove_node_from_pyg_graph(new_data, shifted_node_idx).to(device)
                    #         removed.append(node_idx)
                    #
                    #     lcc_size = get_largest_connected_component(new_data).num_nodes
                    #     if lcc_size <= lcc_threshold:
                    #         to_add.append(subset)
                    if len(to_add) == 0:
                        # it means that there is no subset of that size that breaks the model, therefore it's pointless to search
                        # further as any smaller subsets wouldn't be able to either
                        break
                    else:
                        # we can just replace because the bigger ones are already represented in the smaller ones that passed
                        # any smaller one that passes in the future must pass from the to_add ones iff they pass from the bigger ones
                        successful_subsets = to_add
        finally:
            pool.close()
            pool.join()
        # print("done with pool")
        # now we count occurrences
        probs = torch.zeros(network.num_nodes).to(device)

        for node_id in range(network.num_nodes):
            probs[node_id] = 0

        for successful_subset in successful_subsets:
            for node_id in successful_subset:
                probs[node_id] += 1

        probs /= len(successful_subsets)

    if return_probs:
        return probs
    max_value = probs.max()
    mask = probs >= max_value
    new_y = torch.zeros(network.num_nodes, dtype=torch.int32).to(device)
    new_y[mask] = 1

    return new_y




