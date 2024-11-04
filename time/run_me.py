import os


import torch
import numpy as np
import os
import minexr
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#Args:
#        spp (int, optional): the spp count. Defaults to 4.
#        mode (str, optional): the variant/baseline being run. Defaults to "".
#        conf (str, optional): the scale of every of the 3 network, where 0 is the scaled down version,
#            1 is the default, and 2 is the large version. Defaults to "111".
#        interval (list, optional): the validation interval. Defaults to [700, 800].
os.system("python main.py 4.0 vanilla 111 55 59")