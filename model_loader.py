import os
import cifar10.model_loader

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    return net

def loadMLP(input_dim, hidden_dims, output_dim, batch_norm_enabled = False, bias_enabled = True):
    return cifar10.model_loader.load_MLP(input_dim, hidden_dims, output_dim, batch_norm_enabled, bias_enabled)