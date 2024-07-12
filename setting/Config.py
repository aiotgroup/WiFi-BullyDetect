

class ModelConfig(object):
    def __init__(self, model_name: str):
        self.model_name = model_name



class Run_config():
    def __init__(self, config:dict, type):
        self.python_path = config["path"]["basic_path"]["python_path"]
        self.main_path = config["path"]["basic_path"]["project_path"]+'main.py'
        if type == 'test':
            self.ddp_devices, self.master_port, self.nproc_per_node = self.get_ddp_config(config["training"]["DDP"]["devices"][:1])
        else:
            self.ddp_devices, self.master_port, self.nproc_per_node = self.get_ddp_config(config["training"]["DDP"]["devices"])
        self.log_path = config["path"]["log_path"][type]
        self.config_path = config['path']['result_path'] + '/setting.json'

    def get_ddp_config(self, devices: list):
        nproc_per_node = len(devices)
        master_port = '29501'
        ddp_devices = ''
        for i in range(nproc_per_node - 1):
            ddp_devices += f'{devices[i]},'
        ddp_devices += f'{devices[nproc_per_node - 1]}'

        return ddp_devices, master_port, nproc_per_node
