import os
import torch
import json


device = torch.device("cuda")
#use tf32 
torch.set_float32_matmul_precision('high') 
torch.backends.cudnn.allow_tf32 = True
floatType = torch.float32
intType = torch.long
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#os.environ["CUDA_LAUNCH_BLOCKING"] = '1'



