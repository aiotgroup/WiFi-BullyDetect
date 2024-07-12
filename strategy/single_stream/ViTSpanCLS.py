import torch
import importlib
import torch.nn as nn
import torch.nn.functional as F
from model import ModelConfig
from util import log_f_ch

import logging
logger = logging.getLogger( )


class ViTSpanCLS_config(ModelConfig):
    """
        model_name format: vit_span_cls_(calc_data)
    """

    def __init__(self, config: dict):
        super(ViTSpanCLS_config, self).__init__(config["model"]["backbone_name"])

        self.backbone_name  = config["model"]["backbone_name"]
        self.head_name      = config["model"]["head_name"]

        # patch_size
        self.patch_size = config["model"]["backbone_setting"]["patch_size"]
        # 计算的数据: raw|freq
        self.calc_data = config["model"]["strategy_setting"]["calc_data"]
        # LOSS
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

        # 得到 Backbone_config 和 Head_config
        Model = importlib.import_module("model")
        Backbone_config = getattr(Model, f"{self.backbone_name}_config")
        Head_config = getattr(Model, f"{self.head_name}_config")

        # 初始化 Backbone_config 和 Head_config -----------------------------------
        logger.info(log_f_ch('初始化 backbone:', self.backbone_name, '[config]', str_len=[15, 10, 10]))
        self.backbone_config = Backbone_config(config)
        logger.info(log_f_ch('初始化 head:', self.head_name, '[config]', str_len=[15, 10, 10]))
        self.head_config = Head_config(config)
        # ------------------------------------------------------------------------

        self.init_backbone_head(config)

    def init_backbone_head(self, config):

        if (self.calc_data=="raw"):
            self.backbone_config.n_channel = config["dataset"]["dataset_info"]["n_channel"]
            self.backbone_config.seq_len = config["dataset"]["dataset_info"]["seq_len"]

        self.backbone_config.num_classes = config["dataset"]["dataset_info"]["label_n_class"]
        self.head_config.label_n_class = config["dataset"]["dataset_info"]["label_n_class"]


class ViTSpanCLS(nn.Module):
    def __init__(self, strategy_config: ViTSpanCLS_config):
        super(ViTSpanCLS, self).__init__()
        self.model_name = strategy_config.model_name

        # 得到 Backbone 和 Head ---------------------------------------------------------------
        Model = importlib.import_module("model")
        Backbone = getattr(Model, strategy_config.backbone_name)
        Head     = getattr(Model, strategy_config.head_name)

        logger.info(log_f_ch('初始化 backbone:', strategy_config.backbone_name, str_len=[15, 10]))
        logger.info(log_f_ch('初始化 head:', strategy_config.head_name, str_len=[15, 10]))
        # backbone ----------------------------------------------------------------------------
        self.backbone = Backbone(strategy_config.backbone_config)
        # head --------------------------------------------------------------------------------
        self.head = Head(self.backbone.get_output_size(), strategy_config.head_config)
        # -------------------------------------------------------------------------------------
        self.strategy_config = strategy_config


    def to_backbone_to_head(self, input):
        batch_data = None
        if self.strategy_config.calc_data == 'raw':
            batch_data = input['data']
        elif self.strategy_config.calc_data == 'freq':
            batch_data = input['freq_data']
        elif self.strategy_config.calc_data == 'test':
            batch_data = input

        # 补0 ---------------------------------------------------------------------------------
        if batch_data.size(2) % self.strategy_config.patch_size != 0:
            batch_data = F.pad(batch_data,
                               (0, self.strategy_config.patch_size - (batch_data.size(2) % self.strategy_config.patch_size)),
                               'constant', 0)
        # -------------------------------------------------------------------------------------
        features = self.backbone(batch_data) # VIT
        logits = self.head(features) # 分类

        return logits

    def forward(self, input):
        logits = self.to_backbone_to_head(input)
        loss = None
        for key in logits.keys():
            if loss is None:
                loss = self.strategy_config.loss_fn(logits[key], input[key])
            else:
                loss = loss + self.strategy_config.loss_fn(logits[key], input[key])
        return loss

    def predict(self, input):
        logits = self.to_backbone_to_head(input)
        result = {}
        for key in logits.keys():
            result[key] = F.softmax(logits[key], dim=-1)
        return result

    def get_model_name(self):
        return self.model_name