import logging
import os.path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
import random
from util import init_distributed_mode, dist, cleanup, reduce_value, augmentation
import numpy as np


class Trainer(object):
    def __init__(self,
                 strategy: nn.Module,
                 train_dataset: Dataset,
                 eval_dataset: Dataset,
                 batch_size: int,
                 num_epoch: int,
                 opt_method: str,
                 lr_rate: float,
                 lr_rate_adjust_epoch: int,
                 lr_rate_adjust_factor: float,
                 weight_decay: float,
                 save_epoch: int,
                 eval_epoch: int,
                 patience: int,
                 check_point_path: os.path,
                 use_gpu=True,
                 backbone_setting='',
                 aug = 'None'):
        super(Trainer, self).__init__()
        # model--------------------------------------------------------------------
        self.strategy = strategy
        # dataset -----------------------------------------------------------------
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        # -------------------------------------------------------------------------

        self.batch_size = batch_size
        self.num_epoch = num_epoch
        # learning config ---------------------------------------------------------
        self.opt_method = opt_method
        self.lr_rate = lr_rate
        self.lr_rate_adjust_epoch = lr_rate_adjust_epoch
        self.lr_rate_adjust_factor = lr_rate_adjust_factor
        self.weight_decay = weight_decay

        # training setting --------------------------------------------------------
        self.save_epoch = save_epoch
        self.eval_epoch = eval_epoch
        self.patience = patience
        self.use_gpu = use_gpu
        self.check_point_path = check_point_path
        # -------------------------------------------------------------------------

        self.writer = SummaryWriter(os.path.join(self.check_point_path, f'tb_{self.strategy.backbone.get_model_name()}'))

        # DDP setting -------------------------------------------------------------
        self.dist_url = 'env://'
        self.rank = 0
        self.world_size = 0
        self.gpu=0

        self.device = 'cuda'
        # -------------------------------------------------------------------------
        self.backbone_setting = backbone_setting
        self.aug = aug

    def _init_optimizer(self):
        params = [
            {'params': self.strategy.module.backbone.parameters()},
            {'params': self.strategy.module.head.parameters()},
        ]
        if self.opt_method == 'adam':
            self.optimizer = torch.optim.Adam(params=params,
                                              lr=self.lr_rate,
                                              weight_decay=self.weight_decay)
        elif self.opt_method == 'adamw':
            self.optimizer = torch.optim.AdamW(params=params,
                                               lr=self.lr_rate,
                                               weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(params=params,
                                             lr=self.lr_rate,
                                             weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         self.lr_rate_adjust_epoch,
                                                         self.lr_rate_adjust_factor)

    def _to_var(self, data: dict, device):
        if self.use_gpu:
            for key, value in data.items():
                data[key] = Variable(value.to(device))
        else:
            for key, value in data.items():
                data[key] = Variable(value)
        return data

    def _train_one_step(self, data):
        self.optimizer.zero_grad()

        loss = self.strategy(data)

        loss.backward()

        # 同步
        loss = reduce_value(loss, average=False)

        self.optimizer.step()

        return loss.item()

    def training(self):
        # 给不同的进程分配不同的、固定的随机数种子
        self.set_seed(2023+self.rank)
        init_distributed_mode(args=self)

        # if self.use_gpu:
            # self.strategy = self.strategy.cuda()

        if self.rank == 0:
            print(f'world_size: {self.world_size}, gpu: {self.gpu}')

        device = torch.device(self.device)
        # 给每个rank对应的进程分配训练的样本索引 -------------------------------------------------------------
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(self.eval_dataset)

        # 将样本索引每batch_size个元素组成一个list ----------------------------------------------------------
        train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, self.batch_size, drop_last=True)

        nw = min([os.cpu_count(), self.batch_size if self.batch_size > 1 else 0, 8])  # number of workers
        if self.rank == 0:
            print('Using {} dataloader workers every process'.format(nw))
        # dataset loader -------------------------------------------------------------------------------
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_sampler=train_batch_sampler,
                                                   pin_memory=True,
                                                   num_workers=nw)

        val_loader = torch.utils.data.DataLoader(self.eval_dataset,
                                                 batch_size=self.batch_size,
                                                 sampler=val_sampler,
                                                 pin_memory=True,
                                                 num_workers=nw)
        # load model ------------------------------------------------------------------------------------
        self.strategy = self.strategy.to(device=device)
        # init model weight -----------------------------------------------------------------------------
        if self.rank == 0:
            torch.save(self.strategy.state_dict(), os.path.join(self.check_point_path, "initial_weights.pt"))
        # wait dist
        dist.barrier()

        self.strategy.load_state_dict(torch.load(os.path.join(self.check_point_path, "initial_weights.pt"),
                                                 map_location=device))
        # 转为DDP模型 --------------------------------------------------------------------------------------
        self.strategy = torch.nn.parallel.DistributedDataParallel(self.strategy, device_ids=[self.gpu])

        self._init_optimizer()

        patience_count = 0
        mini_train_loss = float('inf')
        for epoch in range(self.num_epoch):

            train_sampler.set_epoch(epoch+self.rank) # 打乱分配的数据
            np.random.seed(epoch+self.rank)
            self.strategy.train()
            if self.rank == 0:
                log_info = 'Epoch: %d. ' % (epoch + 1)
                tbar = tqdm(train_loader)
            else:
                tbar = train_loader

            train_loss = 0

            for data in tbar:

                data_aug = data.copy() # 浅拷贝
                # train ---------------------------------------------------------------------------
                data = self._to_var(data, device)
                train_loss += self._train_one_step(data)

                # train with aug ------------------------------------------------------ ---------------------
                self.augment(data_aug)
                data_aug = self._to_var(data_aug, device)
                train_loss += self._train_one_step(data_aug)

                if self.rank ==0:
                    tbar.set_description('%s: Epoch: %d: ' % (self.backbone_setting,epoch + 1))
                    tbar.set_postfix(train_loss=train_loss)

            if self.rank==0:
                tbar.close()

            # 等待所有进程计算完毕
            torch.cuda.synchronize(device)

            self.scheduler.step()

            if self.rank == 0:
                log_info += 'Train Loss: %f. ' % train_loss
                self.writer.add_scalar("Train Loss", train_loss, epoch)

            if (epoch + 1) % self.eval_epoch == 0:
                self.strategy.eval()
                with torch.no_grad():
                    eval_loss = 0
                    for data in tqdm(val_loader):
                        data = self._to_var(data, device)
                        eval_loss += self.strategy(data)
                if self.rank == 0:
                    log_info += 'Eval Loss: %f.' % eval_loss
                    self.writer.add_scalar("Eval Loss", eval_loss, epoch)
            if (epoch + 1) % self.save_epoch == 0:
                if self.rank == 0:
                    torch.save(self.strategy.module.state_dict(),
                               os.path.join(self.check_point_path, '%s-%d' % (self.backbone_setting,
                                                                                 epoch + 1)))
            # 如果启用patience机制
            if self.patience != 0:
                if train_loss < mini_train_loss:
                    mini_train_loss = train_loss
                    if self.rank == 0:
                        log_info += 'best-save '
                        # torch.save(self.strategy.module.state_dict(),
                        #            os.path.join(self.check_point_path, '%s-%s-best' % (self.strategy.module.backbone.get_model_name(),
                        #                                                                self.strategy.module.head.get_model_name())))
                    patience_count = 0
                else:
                    patience_count += 1

                if self.rank == 0:
                    log_info += 'Patience Count: %d.' % patience_count

                if patience_count > self.patience:
                    if self.rank == 0:
                        log_info += 'Stop Early, patience has been running out.'
                        print(log_info)
                    break
            if self.rank == 0:
                print(log_info)

        if self.rank == 0:
            torch.save(self.strategy.module.state_dict(),
                       os.path.join(self.check_point_path, '%s-final' % (self.backbone_setting)))

        if self.rank == 0:
            if os.path.exists(os.path.join(self.check_point_path, "initial_weights.pt")) is True:
                os.remove(os.path.join(self.check_point_path, "initial_weights.pt"))

        cleanup()
        print('dist.destroy_process_group()')

    def augment(self, data_aug)->None:
        rand_i = np.random.rand()

        if self.aug == 'None':
            data_aug['data'] = data_aug['data'].numpy()
        elif self.aug == 'i-jitter':
            data_aug['data'] = augmentation2.jitter(data_aug['data'].numpy())
        elif self.aug == 'i-window-s':
            data_aug['data'] = augmentation2.window_slice(data_aug['data'].numpy())
        elif self.aug == 'i-window-w':
            data_aug['data'] = augmentation2.window_warp(data_aug['data'].numpy())
        elif self.aug == 'i-magwarp':
            data_aug['data'] = augmentation2.magnitude_warp(data_aug['data'].numpy())
        elif self.aug == 'i-scaling':
            data_aug['data'] = augmentation2.scaling(data_aug['data'].numpy())
        elif self.aug == 'i-window-w-j':
            if rand_i > 0.4:
                data_aug['data'] = augmentation2.window_warp(data_aug['data'].numpy())
            else:
                data_aug['data'] = augmentation2.jitter(data_aug['data'].numpy())
        elif self.aug == 'i-window-w-m-j':
            if rand_i < 0.3:
                data_aug['data'] = augmentation2.jitter(data_aug['data'].numpy())
            elif rand_i > 0.66:
                data_aug['data'] = augmentation2.window_warp(data_aug['data'].numpy())
            else:
                data_aug['data'] = augmentation2.magnitude_warp(data_aug['data'].numpy())
        elif self.aug == 'i-window-w-s':
            if rand_i > 0.5:
                data_aug['data'] = augmentation2.window_warp(data_aug['data'].numpy())
            else:
                data_aug['data'] = augmentation2.window_slice(data_aug['data'].numpy())
        elif self.aug == 'i-window-w-m':
            if rand_i > 0.5:
                data_aug['data'] = augmentation2.window_warp(data_aug['data'].numpy())
            else:
                data_aug['data'] = augmentation2.magnitude_warp(data_aug['data'].numpy())

        data_aug['data'] = torch.from_numpy(data_aug['data']).float()


    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic =True



# def init_seeds(seed=0, cuda_deterministic=True):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
#     if cuda_deterministic:  # slower, more reproducible
#         cudnn.deterministic = True
#         cudnn.benchmark = False
#     else:  # faster, less reproducible
#         cudnn.deterministic = False
#         cudnn.benchmark = True