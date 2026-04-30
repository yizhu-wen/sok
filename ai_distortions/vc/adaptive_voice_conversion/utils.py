import torch


def cc(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return net.to(device)
