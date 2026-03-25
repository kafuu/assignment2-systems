import torch.nn as nn
import torch.optim as optim
import os
import typing
import torch


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] ) :
    model_ = model.state_dict()
    optimizer_ = optimizer.state_dict()
    output = {"model":model_,"optimizer":optimizer_,"iteration":iteration}
    torch.save(output,out)
    

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    inp = torch.load(src)
    model.load_state_dict(inp["model"])
    optimizer.load_state_dict(inp["optimizer"])
    return inp["iteration"]

def load_model(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module):
    inp = torch.load(src)
    model.load_state_dict(inp["model"])