# implement a rate-based GRU from scratch
# with reference to
# https://www.codegenes.net/blog/gru-from-scratch-pytorch/

import torch.nn as nn


class RateGRU(nn.Module):
    """
    RateGRU = rate-based GRU layer

    example usage:
    batch_size = 32
    n_tbins = 500
    input_size = 100 # nfeatures
    hidden_size = 50
    x = torch.randn(batch_size, n_tbins, input_size)
    h_prev = torch.zeros(batch_size, hidden_size)

    gru = RateGRU(input_size, hidden_size)
    outputs, h = gru(x, h_prev)
    print(outputs.shape)
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.readout(out)

        return out, h
