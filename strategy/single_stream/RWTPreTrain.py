import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F
from model import ModelConfig
from util import log_f_ch

import logging
logger = logging.getLogger()

class RWTPreTrain_config(ModelConfig):

    def __init__(self, config: dict):
        super(RWTPreTrain_config, self).__init__(config["model"]["backbone_name"])
        self.backbone_name  = config["model"]["backbone_name"]
        self.head_name      = config["model"]["head_name"]
