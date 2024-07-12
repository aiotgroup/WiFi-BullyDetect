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
    model_str_list = [
        ('RWT_waveres_8_s_16_0.4_0.1', 64),
        # ('RWT_waveres_8_s_16_0.4_0.1', 64, 0.3),
        # ('RWT_waveres_8_s_16_0.4_0.1', 64, 0.4),
    ]

    dataset_str_list = [
        # ('WiVioLoc-1_i-window-w-s', "/home/lanbo/RWT_wifi_code/result/05-25/pretrain-2/WiVioFT-2-0.2_i-window-w-s_RWT_waveres_8_s_16_0.4_0.1/RWT_waveres_8_s_16_0.4_0.1-final-finetune-0.20.pt"),
        # ('WiVioLoc-1_i-window-w-s', "/home/lanbo/RWT_wifi_code/result/05-25/pretrain/WiVioFT-1-0.2_i-window-w-s_RWT_waveres_8_s_16_0.4_0.1/RWT_waveres_8_s_16_0.4_0.1-final-finetune-0.20.pt"),
        ('WiVioPerson-1_i-window-w-s', "50"),
        ('WiVioPerson-1_i-window-w-s', "100"),
        ('WiVioPerson-1_i-window-w-s', "150"),
        ('WiVioPerson-1_i-window-w-s', "200"),
        ('WiVioPerson-1_i-window-w-s', "250"),
        ('WiVioPerson-1_i-window-w-s', "300")
    ]

    for dataset_str in dataset_str_list:
        dataset_str, model_path = dataset_str
        dataset_setting = get_dataset_setting(dataset_str)
        for model_str in model_str_list:
            model_set   = model_str[0]
            batch_size  = model_str[1]

            backbone_setting = get_model_setting(model_set)

            config = load_setting(r'/home/lanbo/RWT_wifi_code/basic_setting.json')

            config['datetime'] = get_time()

            test_gpu = 3
            tag = f'testing-epoch-{model_path}'

            # TEST model path ===================================================================================
            config["testing"]["Specified"] = True
            config["testing"]["path"] = f"/home/lanbo/RWT_wifi_code/result/05-26/person-1-epoch-300/WiVioPerson-1_i-window-w-s_RWT_waveres_8_s_16_0.4_0.1/RWT_waveres_8_s_16_0.4_0.1-{model_path}"


            # 数据集路径 ==========================================================================================
            config['path']['datasource_path'] = "/home/lanbo/dataset/wifi_violence_processed_loc_class/"

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

            # epoch =============================================================================================
            config["learning"]["num_epoch"] = 0
            # ===================================================================================================

            write_setting(config, os.path.join(config['path']['result_path'], 'setting.json'))

            # TEST ==============================================================================================
            run = Run_config(config, "test")

            os.system(
                f"CUDA_VISIBLE_DEVICES={test_gpu} {run.python_path} "
                f"{run.main_path} --config_path {run.config_path} "
                f"> {run.log_path}"
            )

