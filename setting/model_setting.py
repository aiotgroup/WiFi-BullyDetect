

def get_model_setting(model_set:str)->dict:
    if model_set.startswith('RWT'):
        return get_rwt_setting(model_set)
    elif model_set.startswith('DTW'):
        return get_dtw_setting(model_set)

def get_rwt_setting(model_set:str)->dict:
    '''
        {backbone}_{atten}_{layer}_{scale}_{patch_size}_{dropout}_{droppath}
    '''
    backbone_name, atten, layer, scale, patch_size, dropout, droppath, *others = model_set.split('_')

    backbone_setting = {
        "backbone_name": backbone_name,
        "attn_type": atten,
        "attn_type_layer": int(layer),
        "scale": scale,
        "patch_size": int(patch_size),
        "dropout": float(dropout),
        "droppath": float(droppath),
        "backbone_setting": model_set,
        "others": {
            "high_ratio": float(1.0)
        }
    }

    if len(others) != 0:
        backbone_setting["others"]["high_ratio"] = float(others[0])

    return backbone_setting

def get_dtw_setting(model_set:str)->dict:
    backbone_name, n_neighbors, max_warping_window, subsample_step = model_set.split('_')
    backbone_setting = {
        "backbone_name": backbone_name,
        "n_neighbors": int(n_neighbors),
        "max_warping_window": int(max_warping_window),
        "subsample_step": int(subsample_step),
        "backbone_setting": model_set,
    }
    return backbone_setting