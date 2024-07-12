import importlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(filename)s-%(levelname)s: %(message)s')

logger = logging.getLogger(__name__)

def init_dataset(config: dict, is_test: bool):
    Dataset = importlib.import_module(f"dataset")
    Dataset = getattr(Dataset, config["dataset"]["dataset_name"])
    dataset = Dataset(config, is_test)
    train_dataset, test_dataset = dataset.get_Dataset()
    return train_dataset, test_dataset

def init_model(config: dict):

    strategy_name = config["model"]["strategy_name"]
    logger.info('初始化训练策略: %s' % strategy_name)

    Strategy = importlib.import_module("strategy")
    Strategy_config = getattr(Strategy, f"{strategy_name}_config")
    strategy_config = Strategy_config(config)

    Strategy = getattr(Strategy, strategy_name)
    model = Strategy(strategy_config)

    return model

