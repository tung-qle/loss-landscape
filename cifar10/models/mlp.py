import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, batch_norm_enabled = False, bias_enabled = True):
        super(MLP, self).__init__()
        
        # Saving initialization parameters 
        self.input_size = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.batch_norm_enabled = batch_norm_enabled
        self.bias_enabled = bias_enabled

        # Define network paramterers
        list_dims = [input_dim] + hidden_dims + [output_dim] 
        self.network = self._build_sequence_fc_layers(list_dims, batch_norm_enabled, bias_enabled)

    def forward(self, x):
        out = x.view(out.size(0), -1)
        out = self.network(out)
        return out

    def _build_sequence_fc_layers(self, list_dims, batch_norm_enabled, bias_enabled):
        layers = []
        for idx in range(len(list_dims) - 1):
            layers += self._build_one_fc_layer(list_dims[idx], list_dims[idx + 1], batch_norm_enabled, bias_enabled)
        return nn.Sequential(*layers)

    def _build_one_fc_layer(self, indim, outdim, batch_normed, bias_enabled):
        layers = [nn.Linear(indim, outdim, bias = bias_enabled)]
        if batch_normed:
            layers += [nn.BatchNorm1d(outdim)]
        layers += [nn.ReLU(inplace = True)]
        return layers