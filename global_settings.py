import torch

device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"


device = torch.device(device_name)

# data_root = "/scratch/lprfenau/datasets/graph_data/dataset"
data_root = "./"

datasets = ["synth_train_NEW", "test", "test_ew", "test_large", "test_synth"]
threshold = 0.1
