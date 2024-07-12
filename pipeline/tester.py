import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader


def calc_accuracy(confusion, n_classes):
    correct = 0
    for i in range(n_classes):
        correct += confusion[i][i]
    return correct / np.sum(confusion)


def calc_precision_recall_f1(confusion, n_classes):
    precision = [0 for _ in range(n_classes)]
    recall = [0 for _ in range(n_classes)]
    f1 = [0 for _ in range(n_classes)]

    for i in range(n_classes):
        precision[i] = confusion[i][i] / np.sum(confusion[i, :]) if np.sum(confusion[i, :]) != 0 else 0
        recall[i] = confusion[i][i] / np.sum(confusion[:, i]) if np.sum(confusion[:, i]) != 0 else 0

    for i in range(n_classes):
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0

    return precision, recall, f1


class Tester(object):
    def __init__(self,
                 strategy: nn.Module,
                 eval_data_loader: DataLoader,
                 n_classes: dict,
                 output_path: os.path,
                 backbone_setting: str,
                 use_gpu=True
                 ):
        super(Tester, self).__init__()

        self.strategy = strategy

        self.eval_data_loader = eval_data_loader

        self.use_gpu = use_gpu

        self.n_classes = n_classes
        self.confusion = {
            key: np.zeros((n_class, n_class), dtype=np.int32) for key, n_class in n_classes.items()
        }
        self.file_name = {
            key: [[[] for _ in range (n_class)] for _ in range(n_class)] for key, n_class in n_classes.items()
        }
        self.output_path = output_path
        self.backbone_setting = backbone_setting


    def _to_var(self, data: dict):
        if self.use_gpu:
            for key, value in data.items():
                data[key] = Variable(value.cuda())
        else:
            for key, value in data.items():
                data[key] = Variable(value)
        return data

    def testing(self):
        if self.use_gpu:
            self.strategy = self.strategy.cuda()
        self.strategy.eval()
        with torch.no_grad():
            for data in tqdm(self.eval_data_loader):
                data = self._to_var(data)
                prob = self.strategy.predict(data)
                for key in self.n_classes.keys():
                    prediction = torch.max(prob[key], dim=1)[1]
                    label = data[key]
                    index = data['index']
                    for pred, gt, i in zip(prediction, label, index):
                        self.confusion[key][pred][gt] += 1
                        self.file_name[key][pred][gt].append(i.item())

        for key in self.n_classes.keys():
            print(key.center(100, '='))
            print('Confusion Matrix: '.center(100, '='))
            print(self.confusion[key])
            accuracy = calc_accuracy(self.confusion[key], self.n_classes[key])
            precision, recall, f1 = calc_precision_recall_f1(self.confusion[key], self.n_classes[key])
            print('Precision: ' + str(precision))
            print('Recall: ' + str(recall))
            print('F1: ' + str(f1))
            print('mAccuracy: ' + str(accuracy))
            print('mPricision: ' + str(np.mean(precision)))
            print('mRecall: ' + str(np.mean(recall)))
            print('mF1: ' + str(np.mean(f1)))

            result = pd.DataFrame(self.confusion[key])
            result.to_csv(os.path.join(self.output_path, '%s-%s-confusion_matrix.csv' % (
                self.backbone_setting,
                key,
            )), index=False, header=False)

            pd.DataFrame(self.file_name[key]).to_csv(os.path.join(self.output_path,
                                                                  '%s-%s-file_name.csv' % (self.backbone_setting,
                                                                                              key)), index=False, header=False)
