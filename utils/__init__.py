import torch.optim as optim
import torch


def set_optimizer_scheduler(model, lr, weight_decay, amsgrad, epochs):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, scheduler

def nll_error(log_density):
    return -torch.mean(log_density)