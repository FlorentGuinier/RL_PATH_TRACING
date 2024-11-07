import os
import torch
import os

#Args:
#        spp (int, optional): the spp count. Defaults to 4.
#        mode (str, optional): the variant/baseline being run. Defaults to "vanilla".
#        conf (str, optional): the scale of every of the 3 network, where 0 is the scaled down version,
#            1 is the default, and 2 is the large version. Defaults to "111".
#        interval (list, optional): the validation interval. Defaults to [120, 140].
os.system("python main.py 4.0 vanilla 111 120 140")