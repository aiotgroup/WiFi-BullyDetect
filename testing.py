import os
import torch
import logging

from torch.utils.data.dataloader import DataLoader

from pipeline import Tester as Tester_gpu

import init_util

logger = logging.getLogger(__name__)


def test(config: dict, Tester=Tester_gpu):

    train_dataset, eval_dataset = init_util.init_dataset(config, is_test=True)
    train_dataset.update_config(config)
    strategy = init_util.init_model(config)

    if config["training"]["pretrain"]["enable"]:
        strategy.load_state_dict(torch.load(os.path.join(config["path"]["result_path"],
                                                         '%s-final-finetune-%.2f.pt' % (config["model"]["backbone_setting"]["backbone_setting"],
                                                                                        config["training"]["pretrain"]["ratio"]))))
    elif config["testing"]["Specified"]:
        strategy.load_state_dict(torch.load(config["testing"]["path"]))
    else:
        strategy.load_state_dict(torch.load(os.path.join(config["path"]["result_path"],
                                                         "%s-final" % (config["model"]["backbone_setting"]["backbone_setting"]))))

    print('Test dataset'.center(100, '='))

    tester = Tester(
        strategy=strategy,
        eval_data_loader=DataLoader(eval_dataset, batch_size=config["learning"]["test_batch_size"], shuffle=False),
        n_classes=eval_dataset.get_n_classes(),
        output_path=os.path.join(config["path"]["result_path"], 'Test_dataset'),
        use_gpu=True,
        backbone_setting=config["model"]["backbone_setting"]["backbone_setting"]
    )

    tester.testing()

    print('Train dataset'.center(100, '='))

    tester = Tester(
        strategy=strategy,
        eval_data_loader=DataLoader(train_dataset, batch_size=config["learning"]["test_batch_size"], shuffle=False),
        n_classes=eval_dataset.get_n_classes(),
        output_path=os.path.join(config["path"]["result_path"], 'Train_dataset'),
        use_gpu=True,
        backbone_setting=config["model"]["backbone_setting"]["backbone_setting"]
    )

    tester.testing()
    # 90 1000