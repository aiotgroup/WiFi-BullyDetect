import torch.nn as nn
import torch.nn.functional as F
from util import log_f_ch
from model import ModelConfig
from model import DTW
import logging
logger = logging.getLogger()

class DTWknn_config(ModelConfig):

    def __init__(self, config: dict):
        super(DTWknn_config, self).__init__(config["model"]["backbone_name"])
        self.backbone_name  = config["model"]["backbone_name"]

        if self.backbone_name == 'DTW':
            self.model = DTW(n_neighbors= config['model']['backbone_setting']['n_neighbors'],
                             max_warping_window=config['model']['backbone_setting']['max_warping_window'],
                             subsample_step=config['model']['backbone_setting']['subsample_step'])

        logger.info(log_f_ch('初始化 model:', self.backbone_name, '[model]', str_len=[15, 10, 10]))


class DTWknn(object):
    def __init__(self, strategy_config: DTWknn_config):
        super(DTWknn, self).__init__()
        self.model_name = strategy_config.model_name
        self.model = strategy_config.model

    # def forward(self, input):

