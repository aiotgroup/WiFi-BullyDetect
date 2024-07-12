
from torch.utils.data.dataset import Dataset

class Dataset_temple():
    def __init__(self, datasource_path, save_path):
        self.datasource_path = datasource_path
        self.save_path = save_path

    def get_Dataset(self)->Dataset:
        pass
