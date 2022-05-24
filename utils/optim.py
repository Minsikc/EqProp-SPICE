import torch

def create_optimizer(params, name, **kwargs):
    """
    Create optimizer for the given model.

    Args:
        model: nn.Module whose parameters will be optimized
        name: Name of the optimizer to be used

    Returns:
        torch.optim.Optimizer instance for the given model
    """
    if name == "adagrad":
        return torch.optim.Adagrad(params, **kwargs)
    elif name == "adam":
        return torch.optim.Adam(params, **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(params, **kwargs)
    elif name == "radam":
        return torch.optim.RAdam(params, **kwargs)
    elif name == "adamw":
        return torch.optim.AdamW(params, **kwargs)
    elif name == "adamp":
        from adamp import AdamP
        return AdamP(params, **kwargs)
    else:
        raise ValueError("Optimizer \"{}\" undefined".format(name))