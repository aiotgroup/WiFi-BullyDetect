import torch
import torch.nn as nn
import torch.nn.functional as F


class LateFusionSpanCLSHead(nn.Module):
    """
    pooling_method:
        max: Extract the maximum value feature along the convolutional dimension
        late: First extract features along the convolutional dimension to logits, then take the maximum
        slow:
        local: Take the maximum feature of adjacent convolutional dimensions, then compute logits through the head, and finally average
        conv: Convolve along the non-convolutional dimension, extract the maximum value feature, then compute logits through the head
    """
    def __init__(self, hidden_dim, n_class, pooling_method, downsample_factor=2, conv_in_channel=0):
        super(LateFusionSpanCLSHead, self).__init__()

        self.pooling_method = pooling_method
        self.downsample_factor = downsample_factor
        self.conv_in_channel = conv_in_channel
        if self.conv_in_channel != 0:
            self.default_conv_channel = 128
            self.out_conv = nn.Conv1d(self.conv_in_channel, self.default_conv_channel,
                                      kernel_size=3, stride=1, padding=1)
            self.head_1 = nn.Linear(self.default_conv_channel, self.default_conv_channel, bias=False)
            self.head_2 = nn.Linear(self.default_conv_channel, n_class, bias=False)
        else:
            self.head_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.head_2 = nn.Linear(hidden_dim, n_class, bias=False)

        nn.init.xavier_uniform_(self.head_1.weight)
        nn.init.xavier_uniform_(self.head_2.weight)

    def forward(self, features):
        # batch_size, channel, time
        if self.pooling_method == 'max':
            features = F.adaptive_max_pool1d(features, 1).squeeze()
            return self.head_2(self.head_1(features))
        elif self.pooling_method == 'late':
            features = features.permute(0, 2, 1)
            logits = self.head_2(self.head_1(features))
            return F.adaptive_max_pool1d(logits.permute(0, 2, 1), 1).squeeze()
        elif self.pooling_method == 'slow':
            assert features.size(2) % self.downsample_factor == 0
            features = F.adaptive_max_pool1d(features, features.size(2) // self.downsample_factor)
            features = self.head_1(features.permute(0, 2, 1)).permute(0, 2, 1)
            features = F.adaptive_max_pool1d(features, 1).squeeze()
            return self.head_2(features)
        elif self.pooling_method == 'local':
            assert features.size(2) % self.downsample_factor == 0
            features = F.adaptive_max_pool1d(features, features.size(2) // self.downsample_factor)
            return torch.mean(self.head_2(self.head_1(features.permute(0, 2, 1))), dim=1)
        elif self.pooling_method == 'conv':
            features = self.out_conv(features.permute(0, 2, 1))
            features = F.adaptive_max_pool1d(features, 1).squeeze()
            return self.head_2(self.head_1(features))
