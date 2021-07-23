import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from collections import OrderedDict, defaultdict
import numpy as np

class GlobalMagGrad(prune.BasePruningMethod):

    PRUNING_TYPE = 'global'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()

def global_mag_grad(module, inputs, outputs):
    params = {module: get_params(module) for module in prunable_modules(module)}
    grads = get_param_gradients(module, inputs, outputs)

    importances = { mod: 
                    {p: np.abs(params[mod][p]*grads[mod][p]) 
                        for p in mod_params}
                    for mod, mod_params in params.items()}
    
    print(type(importances))

    importance_scores = importances
    GlobalMagGrad.apply(module, importance_scores=importance_scores, name="weight")

def prunable_modules(module):
    prunable = [module for module in module.modules() if can_prune(module)]
    return prunable

def can_prune(module):
    if hasattr(module, 'is_classifier'):
        return not module.is_classifier
    #if isinstance(module, (MaskedModule, nn.Linear, nn.Conv2d)): ## Removed Masked Module
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        return True
    return False

def get_params(model, recurse=False):
    """Returns dictionary of paramters

        Arguments:
            model {torch.nn.Module} -- Network to extract the parameters from

        Keyword Arguments:
            recurse {bool} -- Whether to recurse through children modules

        Returns:
            Dict(str:numpy.ndarray) -- Dictionary of named parameters their
                                    associated parameter arrays
    """
    params = {k: v.detach().cpu().numpy().copy()
              for k, v in model.named_parameters(recurse=recurse)}
    return params

def get_param_gradients(model, inputs, outputs, loss_func=None, by_module=True):

    gradients = OrderedDict()

    if loss_func is None:
        loss_func = nn.CrossEntropyLoss()

    training = model.training
    model.train()
    pred = model(inputs)
    loss = loss_func(pred, outputs)
    loss.backward()

    if by_module:
        gradients = defaultdict(OrderedDict)
        for module in model.modules():
            assert module not in gradients
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad and param.grad is not None:
                    gradients[module][name] = param.grad.detach().cpu().numpy().copy()

    else:
        gradients = OrderedDict()
        for name, param in model.named_parameters():
            assert name not in gradients
            if param.requires_grad and param.grad is not None:
                gradients[name] = param.grad.detach().cpu().numpy().copy()

    model.zero_grad()
    model.train(training)

    return gradients
