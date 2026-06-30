import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DetectorAttentionGRU(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 32, num_layers = 2, num_classes = 1, dropout = 0.1, bidirectional = False):
        super(DetectorAttentionGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout,
            bidirectional = bidirectional
        )

        self.output_size = hidden_size * (2 if bidirectional else 1)

        self.attention = nn.Sequential(
            nn.Linear(self.output_size, 1)
        )

        self.fc = nn.Linear(self.output_size, num_classes) 

        
    def forward(self, x, return_attention = False):
        #h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_() 
        out, h_n = self.gru(x)

        attn = self.attention(out)
        attention_weights = F.softmax(attn, dim = 1)

        context = torch.sum(attention_weights * out, dim = 1)  # [batch_size, output_size]
        context = self.dropout(context)
        output = self.fc(context)

        if return_attention:
            return output, attention_weights

        return output
    