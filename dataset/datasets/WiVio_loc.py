import os
import torch
import pandas as pd
from torch.utils.data.dataset import Dataset
from util import log_f_ch, load_mat
from dataset import Dataset_temple


import logging
logger = logging.getLogger( )

class WiVioFT(Dataset_temple):
    def __init__(self, config, is_test):
        super(WiVioFT, self).__init__(config["path"]["datasource_path"], config["path"]["result_path"])
        self.config = config
        self.loc = config["dataset"]["dataset_setting"]["loc"]
        self.pre_ratio = config["dataset"]["dataset_setting"]["ratio"]
        self.is_test = is_test

    def load_wifi_Vio_data(self):

        train_data = {"data_path": os.path.join(self.datasource_path, 'data'),
                      "list_path": os.path.join(self.datasource_path, f'ft_loc_{self.loc}_{"%.1f"%(self.pre_ratio)}_train_list.csv'),
                      "save_path": os.path.join(self.save_path, 'Train_dataset', 'train_dataset.csv')}
        test_data = {"data_path": os.path.join(self.datasource_path, 'data'),
                     "list_path": os.path.join(self.datasource_path, f'ft_loc_{self.loc}_{"%.1f"%(self.pre_ratio)}_test_list.csv'),
                     "save_path": os.path.join(self.save_path, 'Test_dataset', 'test_dataset.csv')}

        for data in [train_data, test_data]:
            if os.path.exists(data['save_path']):
                os.remove(data['save_path'])

        if not os.path.exists(os.path.join(self.save_path, 'Test_dataset')):
            os.makedirs(os.path.join(self.save_path, 'Test_dataset'))
        if not os.path.exists(os.path.join(self.save_path, 'Train_dataset')):
            os.makedirs(os.path.join(self.save_path, 'Train_dataset'))

        return train_data, test_data

    def get_Dataset(self):
        train_data, test_data = self.load_wifi_Vio_data()
        train_dataset, test_dataset = WiFiVioDataset(train_data, self.is_test), WiFiVioDataset(test_data, self.is_test)
        return train_dataset, test_dataset

class WiVioLoc(Dataset_temple):
    def __init__(self, config, is_test):
        super(WiVioLoc, self).__init__(config["path"]["datasource_path"], config["path"]["result_path"])
        self.config = config
        self.loc = config["dataset"]["dataset_setting"]["loc"]
        self.is_test = is_test

    def load_wifi_Vio_data(self):

        train_data = {"data_path": os.path.join(self.datasource_path, 'data'),
                      "list_path": os.path.join(self.datasource_path, f'loc_{self.loc}_train_list.csv'),
                      "save_path": os.path.join(self.save_path, 'Train_dataset', 'train_dataset.csv')}
        test_data = {"data_path": os.path.join(self.datasource_path, 'data'),
                     "list_path": os.path.join(self.datasource_path, f'loc_{self.loc}_test_list.csv'),
                     "save_path": os.path.join(self.save_path, 'Test_dataset', 'test_dataset.csv')}

        for data in [train_data, test_data]:
            if os.path.exists(data['save_path']):
                os.remove(data['save_path'])

        if not os.path.exists(os.path.join(self.save_path, 'Test_dataset')):
            os.makedirs(os.path.join(self.save_path, 'Test_dataset'))
        if not os.path.exists(os.path.join(self.save_path, 'Train_dataset')):
            os.makedirs(os.path.join(self.save_path, 'Train_dataset'))

        return train_data, test_data

    def get_Dataset(self):
        train_data, test_data = self.load_wifi_Vio_data()
        train_dataset, test_dataset = WiFiVioDataset(train_data, self.is_test), WiFiVioDataset(test_data, self.is_test)
        return train_dataset, test_dataset

class WiVioPerson(Dataset_temple):
    def __init__(self, config, is_test):
        super(WiVioPerson, self).__init__(config["path"]["datasource_path"], config["path"]["result_path"])
        self.config = config
        self.person = config["dataset"]["dataset_setting"]["person"]
        self.is_test = is_test

    def load_wifi_Vio_data(self):

        train_data = {"data_path": os.path.join(self.datasource_path, 'data'),
                      "list_path": os.path.join(self.datasource_path, f'per_{self.person}_train_list.csv'),
                      "save_path": os.path.join(self.save_path, 'Train_dataset', 'train_dataset.csv')}
        test_data = {"data_path": os.path.join(self.datasource_path, 'data'),
                     "list_path": os.path.join(self.datasource_path, f'per_{self.person}_test_list.csv'),
                     "save_path": os.path.join(self.save_path, 'Test_dataset', 'test_dataset.csv')}

        for data in [train_data, test_data]:
            if os.path.exists(data['save_path']):
                os.remove(data['save_path'])

        if not os.path.exists(os.path.join(self.save_path, 'Test_dataset')):
            os.makedirs(os.path.join(self.save_path, 'Test_dataset'))
        if not os.path.exists(os.path.join(self.save_path, 'Train_dataset')):
            os.makedirs(os.path.join(self.save_path, 'Train_dataset'))

        return train_data, test_data

    def get_Dataset(self):
        train_data, test_data = self.load_wifi_Vio_data()
        train_dataset, test_dataset = WiFiVioDataset(train_data, self.is_test), WiFiVioDataset(test_data, self.is_test)
        return train_dataset, test_dataset

class WiVio(Dataset_temple):
    def __init__(self, config, is_test):
        super(WiVio, self).__init__(config["path"]["datasource_path"], config["path"]["result_path"])
        self.config = config
        self.is_test = is_test

    def load_wifi_Vio_data(self):
        train_data = {"data_path": os.path.join(self.datasource_path, 'train'),
                      "list_path": os.path.join(self.datasource_path, f'train_list.csv'),
                      "save_path": os.path.join(self.save_path, 'Train_dataset', 'train_dataset.csv')}
        test_data = {"data_path": os.path.join(self.datasource_path, 'test'),
                     "list_path": os.path.join(self.datasource_path, f'test_list.csv'),
                     "save_path": os.path.join(self.save_path, 'Test_dataset', 'test_dataset.csv')}

        for data in [train_data, test_data]:
            if os.path.exists(data['save_path']):
                os.remove(data['save_path'])

        if not os.path.exists(os.path.join(self.save_path, 'Test_dataset')):
            os.makedirs(os.path.join(self.save_path, 'Test_dataset'))
        if not os.path.exists(os.path.join(self.save_path, 'Train_dataset')):
            os.makedirs(os.path.join(self.save_path, 'Train_dataset'))

        return train_data, test_data

    def get_Dataset(self):
        train_data, test_data = self.load_wifi_Vio_data()
        train_dataset, test_dataset = WiFiVioDataset(train_data, self.is_test), WiFiVioDataset(test_data, self.is_test)
        return train_dataset, test_dataset

class WiFiVioDataset(Dataset):
    def __init__(self, data, is_test: bool):
        super(WiFiVioDataset, self).__init__()

        if is_test: logger.info(f'加载WiFiVio数据集{" [for TEST]"}')
        else: logger.info(f'加载WiFiVio数据集{" [for TRAIN]"}')
        self.is_test = is_test
        self.data_path = data['data_path']
        self.data_list = pd.read_csv(data['list_path'])
        self.save_path = data['save_path']

        self.n_channel, self.seq_len, self.num_sample = self.init_n(self.data_list, self.data_path)
        self.label_n_class = 7
        self.freq_n_channel, self.freq_seq_len = None, None

        logger.info(log_f_ch('num_sample: ', str(self.num_sample)))
        logger.info(log_f_ch('n_class: ', str(self.label_n_class)))
        logger.info(log_f_ch('seq_len: ', str(self.seq_len)))
        logger.info(log_f_ch('n_channel: ', str(self.n_channel)))

    def __getitem__(self, index):
        data = load_mat(os.path.join(self.data_path, f'{self.data_list.iloc[index]["file"]}.h5'))

        if self.is_test:
            pd.DataFrame([[index, f'{self.data_list.iloc[index]["file"]}']]).to_csv(self.save_path, mode='a',index=False, header=False)
            return {
                'data': torch.from_numpy(data['amp']).float(),
                'label': torch.from_numpy(data['label']).long()-1,
                'index': torch.tensor(index)
            }
        else:
            return {
                'data': torch.from_numpy(data['amp']).float(),
                'label': torch.from_numpy(data['label']).long()-1,
            }

    def aug_fun(self, data):

        return data

    # def loss_fliter(self, data, frequency=1000, highpass=100):
    #     [b, a] = signal.butter(3, highpass / frequency * 2, 'lowpass')
    #     Signal_pro = signal.filtfilt(b, a, data)
    #     return Signal_pro

    def __len__(self):
        return self.num_sample

    def get_n_channels(self):
        return {
            'data': self.n_channel,
            'freq_data': self.freq_n_channel,
        }

    def get_seq_lens(self):
        return {
            'data': self.seq_len,
            'freq_data': self.freq_seq_len,
        }

    def get_n_classes(self):
        return {
            'label': self.label_n_class,
        }

    def init_n(self, data_list: pd.DataFrame, data_path):
        tmp_data_name = data_list.iloc[0]['file']
        tmp_data = load_mat(os.path.join(data_path, f'{tmp_data_name}.h5'))

        n_channel, seq_len = tmp_data['amp'].shape
        num_sample = len(data_list)

        return n_channel, seq_len, num_sample

    def update_config(self, config)->None:
        config["dataset"]["dataset_info"] = {
            "n_channel": self.n_channel,
            "seq_len":  self.seq_len,
            "label_n_class": self.label_n_class
        }