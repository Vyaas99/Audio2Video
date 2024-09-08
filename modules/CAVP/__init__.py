import importlib
import os
import subprocess
import torch

from omegaconf import OmegaConf

CAVP_CONFIG_PATH = "./modules/CAVP/CAVP.yaml"
CKPT_PATH = "./models/CAVP/cavp_epoch66.ckpt"

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    # print(module)
    # print(cls)
    # print(string)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def init_first_from_ckpt(model_object, path):
    model = torch.load(path, map_location="cpu")
    if "state_dict" in list(model.keys()):
        model = model["state_dict"]
    # Remove: module prefix
    new_model = {}
    for key in model.keys():
        new_key = key.replace("module.","")
        new_model[new_key] = model[key]
    missing, unexpected = model_object.load_state_dict(new_model, strict=False)
    print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
    if len(missing) > 0:
        print(f"Missing Keys: {missing}")
    if len(unexpected) > 0:
        print(f"Unexpected Keys: {unexpected}")
    return model_object

def load_CAVP(device:str="cuda"):
    # initialize CAVP
    CAVP_config = OmegaConf.load(CAVP_CONFIG_PATH)
    CAVP = instantiate_from_config(CAVP_config.model).to(device)

    # loading weight
    if os.path.exists(CKPT_PATH):
        CAVP = init_first_from_ckpt(CAVP, CKPT_PATH).eval()
    else:
        command = "huggingface-cli download SimianLuo/Diff-Foley diff_foley_ckpt/cavp_epoch66.ckpt --local-dir ./models/CAVP"
        subprocess.run(command.split())
        CAVP = init_first_from_ckpt(CAVP, CKPT_PATH).eval()
    return CAVP

