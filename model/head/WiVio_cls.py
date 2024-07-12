import torch.nn as nn
from ..model_config import ModelConfig

class WiVio_cls_config(ModelConfig):
    label_n_classes = 7
    def __init__(self, model_name):
        super(WiVio_cls_config, self).__init__(model_name)

class WiVio_cls(nn.Module):
    def __init__(self, hidden_dim, config: WiVio_cls_config):
        super(WiVio_cls, self).__init__()
        self.hidden_dim = hidden_dim
        self.model_name = config.model_name
        self.label_n_classes = config.label_n_classes
        self.label_head = SimpleSpanCLSHead(hidden_dim, config.label_n_classes)

    def forward(self, features):
        return {
            'label': self.label_head(features),
        }

    def get_model_name(self):
        return self.model_name


class SimpleSpanCLSHead(nn.Module):
    def __init__(self, hidden_dim, n_class):
        super(SimpleSpanCLSHead, self).__init__()

        self.head_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.head_2 = nn.Linear(hidden_dim, n_class, bias=False)

        nn.init.xavier_uniform_(self.head_1.weight)
        nn.init.xavier_uniform_(self.head_2.weight)

    def forward(self, features):
        return self.head_2(self.head_1(features))