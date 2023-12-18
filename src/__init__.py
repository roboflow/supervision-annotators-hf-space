from torch import device
from torch.cuda import is_available as cuda_is_available

DEVICE = device("cuda" if cuda_is_available() else "cpu")
