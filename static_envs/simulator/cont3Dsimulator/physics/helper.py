import torch
import numpy as np
import scipy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
floatType = torch.float64

def tensor(val, data_type=floatType, device=device):
    return torch.tensor(val, dtype=data_type, requires_grad=False, device=device)

def sparse(A, shape=None, data_type=floatType, device=device):
    if shape is None:
        shape = A.shape
    if device.type != "mps":
        return torch.sparse_coo_tensor(torch.LongTensor(np.vstack((A.row, A.col))),
                                   torch.tensor(A.data, dtype=data_type),
                                   torch.Size(shape)).to(device)
    else:
        return torch.tensor(A.todense(), dtype=data_type, device=device)

def array(val):
    return val.detach().cpu().numpy()

def arange(n, m,data_type=int, device=device):
    return torch.arange(n, m, dtype=data_type, device=device)

def ones(n, m = None, l = None, data_type=floatType, device=device):
    if m is None:
        return torch.ones(n, device=device, dtype=data_type, requires_grad=False)
    elif l is None:
        return torch.ones((n, m), device=device, dtype=data_type, requires_grad=False)
    else:
        return torch.ones((n, m, l), device=device, dtype=data_type, requires_grad=False)

def eye(n, data_type=floatType, device=device):
        return torch.eye(n, device=device, dtype=data_type, requires_grad=False)

def zeros(n, m = None, l = None, data_type=floatType, device=device):
    if m is None:
        return torch.zeros(n, device=device, dtype=data_type, requires_grad=False)
    elif l is None:
        return torch.zeros((n, m), device=device, dtype=data_type, requires_grad=False)
    else:
        return torch.zeros((n, m, l), device=device, dtype=data_type, requires_grad=False)