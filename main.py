import os
import argparse
import logging
import torch
from training import train, pre_train, pre_Trainer2
from testing import test
from util import load_setting, write_setting, update_time


logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s-%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def init_configs():
    parser = argparse.ArgumentParser(description="RWT study")
    parser.add_argument('--is_train', dest="is_train", required=False, type=bool, default=False,
                        help="是否训练")
    parser.add_argument('--config_path', dest="config_path", required=True, type=str, help="config path")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = init_configs()
    config = load_setting(args.config_path)

    if args.is_train:
        if config["training"]["pretrain"]["enable"]:
            pre_train(config=config, Trainer=pre_Trainer2)
        elif "trainer" in config["training"].keys():
            train(config=config)
        else:
            train(config=config)
    else:
        test(config=config)