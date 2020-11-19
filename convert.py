import sys

import torch
import numpy as np

device = torch.device('cuda')
epsilon = 0.031
eme = epsilon - np.finfo(np.float32(1.0)).eps
x = np.load('data_attack/cifar10_X.npy')
x = torch.tensor(x, device=device)
xadv = torch.load(sys.argv[1], map_location=device)['adv_complete']
xadv = xadv.permute(0, 2, 3, 1).contiguous()
d = xadv - x
d = torch.clamp(d, -eme, eme)
xadv = torch.clamp(d + x, 0.0, 1.0)
np.save('data_attack/cifar10_X_adv.npy', xadv.cpu().numpy())
