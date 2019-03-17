import torch
from hw2p2 import paths

def save_checkpoint(model, current_epoch):
    """Save checkpoint if a new best is achieved"""
    model_filepath = str(paths.model + str(current_epoch) +'.pth')
    torch.save(model, model_filepath)
    print ("=> Saved model")