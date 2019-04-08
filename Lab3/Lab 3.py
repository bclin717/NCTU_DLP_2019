from __future__ import print_function

import torch
import torch.optim
from torch.cuda import device


def main():
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    main()
