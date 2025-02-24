import torch

device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"

device = torch.device(device_name)
# print(device)
data_root = "/scratch/lprfenau/datasets/graph_data/dataset"
# data_root = "./"

datasets = ["synth_train_NEW", "test", "test_ew", "test_large", "test_synth"]
threshold = 0.1
features_they_use = ["degree", "clustering_coefficient", "kcore", "chi_degree"]
mp_num_cpu = 16
# paths_to_evaluate = 1000
lcc_threshold_fn = lambda x: max(int(threshold * x), 1)