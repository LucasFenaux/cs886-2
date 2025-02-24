# cs886-2
Repo for coding project for CS886: Graph Neural Networks

## Data
In the global setting, one can set the <code>data_root</code> variable to indicate the location of the graphs to use. The graphs can be found in the original GDM repository
 <url>https://github.com/NetworkScienceLab/GDM</url> under the dataset folder.
 
## Experiments
To produce the results of the GDM experiment, run the <code>run_gdm.py</code> file. To use reinsertion, add the <code>-r</code> flag. To run it only on the Brazilian corruption network, run it with the <code>--test_size 1</code> flag. 

To run the same grid as is offered in the GDM repository to find a good set of hyperparameters for their GNN model, run <code>run_gdm_grid.py</code>. We include 5 runs of that grid without reinsertion (grid_results_i.json with i in [1,2,3,4,5]) and 1 run with reinsertion (grid_results.json).

### Next-node prediction
To train a next node prediction model, run <code>run_next_node_dm.py</code>. To reproduce the table of results in the paper for the grid search we performed, run <code>run_grid.py</code>.
We include the results to this grid in the results folder.

### Reinforcement learning-based next-node prediction
To run the most up to date experiment on reinforcement learning, run <code>run_rl_v2.py</code>. It differs from <code>run_rl_v1.py</code> in how the graphs are used as training environments. In V1, the models learn from one graph at a time, dismantling each one node at a time before moving to the next. In V2, it learns to dismantle all the graphs at the same time, sampling at random one of the graph at each step and removing one node from it. 
