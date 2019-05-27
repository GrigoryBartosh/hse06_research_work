import os
import shutil

import torch

from common.config import PATH

__all__ = ["save_unimodel", "load_state", "load_unimodel_state", "load_unimodel"]

ALL_UNIMODEL_DIR = PATH["MODELS"]["UNIMODEL_DIR"]
NAME_UNIMODEL_STATE = "state.pth"
NAME_UNIMODEL = "model.pth"


def remove_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_unimodel_dir_path(name):
    return os.path.join(ALL_UNIMODEL_DIR, name)

def get_unimodel_state_path(name):
    return os.path.join(ALL_UNIMODEL_DIR, name, NAME_UNIMODEL_STATE)

def get_unimodel_path(name):
    return os.path.join(ALL_UNIMODEL_DIR, name, NAME_UNIMODEL)

def save_unimodel(model, name):
    path_dir = get_unimodel_dir_path(name)
    path_unimodel_state = get_unimodel_state_path(name)
    path_unimodel = get_unimodel_path(name)
    remove_dir(path_dir)
    make_dir(path_dir)
    torch.save(model.state_dict(), path_unimodel_state)
    torch.save(model, path_unimodel)

def load_state(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))

def load_unimodel_state(model, name, device='cpu'):
    path = get_unimodel_state_path(name)
    load_state(model, path)

def load_unimodel(name, device='cpu'):
    path = get_unimodel_path(name)
    return torch.load(path, map_location=device)
