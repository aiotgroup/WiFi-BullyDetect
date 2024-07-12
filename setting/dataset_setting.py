


def get_dataset_setting(dataset_setting: str)->dict:

    if dataset_setting.startswith('WiVio'):
        if dataset_setting.startswith('WiVioLoc'):
            return get_wivio_loc_setting(dataset_setting)
        elif dataset_setting.startswith('WiVioFT'):
            return get_wivio_ft_loc_setting(dataset_setting)
        elif dataset_setting.startswith('WiVioPerson'):
            return get_wivio_person_setting(dataset_setting)
        return get_wivio_setting(dataset_setting)

def get_wivio_person_setting(dataset_set:str)->dict:
    dataset_name, augment, *others = dataset_set.split('_')
    dataset_name, person = dataset_name.split('-')
    dataset_setting = {
        'dataset_name': dataset_name,
        'person': int(person),
        'augment': augment,
        'others': {}
    }
    return  dataset_setting

def get_wivio_loc_setting(dataset_set:str)->dict:
    dataset_name, augment, *others = dataset_set.split('_')
    dataset_name, loc = dataset_name.split('-')
    dataset_setting = {
        'dataset_name': dataset_name,
        'loc': int(loc),
        'augment': augment,
        'others': {}
    }
    return  dataset_setting

def get_wivio_setting(dataset_set:str)->dict:
    dataset_name, augment, *others = dataset_set.split('_')
    dataset_setting = {
        'dataset_name': dataset_name,
        'augment': augment,
        'others': {}
    }
    return dataset_setting

def get_wivio_ft_loc_setting(dataset_set:str)->dict:
    dataset_name, augment, *others = dataset_set.split('_')
    dataset_name, loc, ratio = dataset_name.split('-')
    dataset_setting = {
        'dataset_name': dataset_name,
        'loc': int(loc),
        'ratio': float(ratio),
        'augment': augment,
        'others': {}
    }
    return  dataset_setting