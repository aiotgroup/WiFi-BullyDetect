import sys
import os
import torch

project_path = '/home/lanbo/RWT_wifi_code/'
sys.path.append(project_path)

from util import load_setting, update_time, get_time, get_log_path, get_result_path, get_day, write_setting
from setting import get_dataset_setting, get_model_setting, Run_config


if __name__ == '__main__':

    day = get_day()

    '''
        {backbone}_{atten}_{layer}_{scale}_{patch_size}_{dropout}_{droppath}
    '''
    '''
        {model}, {batch_size}, {epoch}
    '''
    model_str_list = [
        ('RWT_waveres_8_s_8_0.4_0.1', 64, 500),
        ('RWT_waveres_8_s_16_0.4_0.1', 64, 500),
        ('RWT_waveres_8_s_32_0.4_0.1', 64, 500),
        ('RWT_waveres_8_b_16_0.4_0.1', 64, 500),
        ('RWT_waveres_8_b_32_0.4_0.1', 64, 500),
        ('RWT_waveres_8_b_64_0.4_0.1', 64, 500),

        ('RWT_timm_8_s_8_0.4_0.1', 64, 500),
        ('RWT_timm_8_s_16_0.4_0.1', 64, 500),
        ('RWT_timm_8_s_32_0.4_0.1', 64, 500),
        ('RWT_timm_8_b_16_0.4_0.1', 64, 500),
        ('RWT_timm_8_b_32_0.4_0.1', 64, 500),
        ('RWT_timm_8_b_64_0.4_0.1', 64, 500),
    ]

    dataset_str_list = [

        # 'WiVio_None',
        # 'WiVio_i-jitter',
        # 'WiVio_i-window-s',
        # 'WiVio_i-window-w',
        # 'WiVio_i-magwarp',
        # 'WiVio_i-scaling',
        # 'WiVio_i-window-w-j',
        # 'WiVio_i-window-w-m',
        # 'WiVio_i-window-w-s',
        # 'WiVio_None',
        # 'WiVio_i-jitter',
        # 'WiVio_i-window-s',
        # 'WiVio_i-window-w',
        # 'WiVio_i-magwarp',
        # 'WiVio_i-scaling',
        # 'WiVio_i-window-w-j',
        # 'WiVio_i-window-w-m',
        'WiVio_i-window-w-s',
    ]


    for dataset_str in dataset_str_list:
        dataset_setting = get_dataset_setting(dataset_str)
        for model_str in model_str_list:
            model_set   = model_str[0]
            batch_size  = model_str[1]
            epoch       = model_str[2]

            backbone_setting = get_model_setting(model_set)

            config = load_setting(r'/home/lanbo/RWT_wifi_code/basic_setting.json')

            config['datetime'] = get_time()

            config["training"]["DDP"]["enable"] = True
            config["training"]["DDP"]["devices"] = [3]
            test_gpu = 3

            # TAG ===============================================================================================
            tag = f'model_size'

            # 数据集路径 ==========================================================================================
            # config['path']['datasource_path'] = "/home/lanbo/dataset/wifi_violence_processed_loc_class/"
            config['path']['datasource_path'] = '/home/lanbo/dataset/wifi_violence_processed_loc/'

            config['path']['log_path']      = get_log_path(config, day, dataset_str, model_set, tag)
            config['path']['result_path']   = get_result_path(config, day, dataset_str, model_set, tag)
            # ===================================================================================================

            config['dataset']['dataset_name']    = dataset_setting['dataset_name']
            config['dataset']['dataset_setting'] = dataset_setting

            # model setting =====================================================================================
            config['model']['backbone_name'] = backbone_setting['backbone_name']
            config['model']['backbone_setting'] = backbone_setting

            config['model']['head_name']     = 'WiVio_cls'
            config['model']['strategy_name'] = 'ViTSpanCLS'
            config["model"]["strategy_setting"]["calc_data"] = "raw"
            # ===================================================================================================

            config['learning']['train_batch_size'] = int(batch_size)
            config['learning']['test_batch_size'] = int(batch_size)
            config['learning']['save_epoch']  = 400

            # epoch =============================================================================================
            config["learning"]["num_epoch"] = epoch
            # ===================================================================================================

            write_setting(config, os.path.join(config['path']['result_path'], 'setting.json'))

            # TRAIN =============================================================================================
            run = Run_config(config, "train")

            os.system(
                f"CUDA_VISIBLE_DEVICES={run.ddp_devices} {run.python_path} -m torch.distributed.launch --nproc_per_node {run.nproc_per_node} "
                f"--master_port='29501' --use_env "
                f"{run.main_path} --is_train true --config_path {run.config_path} "
                f"> {run.log_path}"
            )
            # os.system(
            #     f"{run.python_path} "
            #     f"{run.main_path} --is_train true --config_path {run.config_path} "
            #     f"> {run.log_path}"
            # )
            # TEST ==============================================================================================
            # run = Run_config(config, "test")
            #
            # os.system(
            #     f"CUDA_VISIBLE_DEVICES={test_gpu} {run.python_path} "
            #     f"{run.main_path} --config_path {run.config_path} "
            #     f"> {run.log_path}"
            # )

