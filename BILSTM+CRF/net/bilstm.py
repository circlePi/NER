import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np

import config.config as config

torch.manual_seed(2018)
torch.cuda.manual_seed(2018)
torch.cuda.manual_seed_all(2018)
np.random.seed(2018)

os.environ["CUDA_VISIBLE_DEVICE"] = "%d"%config.device

class RNN(nn.Module):
    def __init__(self,
                 hidden_size, bi_flag,
                 num_layer,
                 input_size,
                 cell_type,
                 dropout,
                 num_tag):
        super(RNN, self).__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.dropout = dropout


        if torch.cuda.is_available():
            self.device = torch.device("cuda")



        if cell_type == "LSTM":
            self.rnn_cell = nn.LSTM(input_size=input_size,
                                    hidden_size=hidden_size,
                                    num_layers=num_layer,
                                    batch_first=True,
                                    dropout=dropout,
                                    bidirectional=bi_flag)
        elif cell_type == "GRU":
            self.rnn_cell = nn.GRU(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layer,
                                   batch_first=True,
                                   dropout=dropout,
                                   bidirectional=bi_flag)
        else:
            raise TypeError("RNN: Unknown rnn cell type")

        # 是否双向
        self.bi_num = 2 if bi_flag else 1
        self.linear = nn.Linear(in_features=hidden_size*self.bi_num, out_features=num_tag)

    def forward(self, embeddings, length):
        # 去除padding元素
        # embeddings_packed: (batch_size*time_steps, embedding_dim)
        embeddings_packed = pack_padded_sequence(embeddings, length, batch_first=True)
        output, _ = self.rnn_cell(embeddings_packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.linear(output)
        output = F.dropout(output, p=self.dropout, training=self.training)
        # output = F.tanh(output)
        return output
