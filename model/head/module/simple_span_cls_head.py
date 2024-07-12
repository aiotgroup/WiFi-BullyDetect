import torch.nn as nn


class SimpleSpanCLSHead(nn.Module):
    def __init__(self, hidden_dim, n_class):
        super(SimpleSpanCLSHead, self).__init__()

        self.head_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.head_2 = nn.Linear(hidden_dim, n_class, bias=False)

        nn.init.xavier_uniform_(self.head_1.weight)
        nn.init.xavier_uniform_(self.head_2.weight)

    def forward(self, features):
        return self.head_2(self.head_1(features))
