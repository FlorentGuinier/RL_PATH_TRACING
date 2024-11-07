import time

import torch
import torch.nn as nn
import torch.optim as optim

from .model import *
from .dasr import get_model as dasr
from .ntas import get_model as ntas
from .model00 import get_model as a00
from .model01 import get_model as a01
from .model10 import get_model as a10
from .model21 import get_model as a21
from .model12 import get_model as a12


from .loss import *

# Worker function
def main_worker(inp=31, ic=32, mode=None, conf="111"): 
    """creates all objects needed to train the denoiser network

    Args:
        inp (int, optional): input number of channels. Defaults to 31.
        ic (int, optional): latent space number of channels. Defaults to 32.
        mode (str, optional): the variant/baseline being run. Defaults to None.
        conf (str, optional): the scale of every network. Defaults to "111". 0 means small, 1 default, 2 large.

    Returns:
        _type_: _description_
    """    
    if mode == "ntas":
        model = ntas(inp=inp, ic=ic)
    elif mode == "dasr":
        model = dasr(inp=inp, ic=ic)
    else:
        if conf[1:] == "01":
            model = a01( inp=inp, ic=ic)
        elif conf[1:] == "10":
            model = a10( inp=inp, ic=ic)
        elif conf[1:] == "21":
            model = a21( inp=inp, ic=ic)
        elif conf[1:] == "12":
            model = a12( inp=inp, ic=ic)
        elif conf[1:] == "00":
            model = a00( inp=inp, ic=ic)
        else:
            model = get_model( inp=inp, ic=ic)
    criterion = MixLoss([L1Loss(), MSSSIMLoss(weights=None)], [0.16, 0.84])

    optimizer = optim.Adam(model.parameters(), lr=1)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    model.apply(init_weights)

    lr_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-05,
        #total step seems to be 1 for init
        # then a variable amount seem from 24 to 36 so far (always a modulo of 4 because of batch size).
        # TODO understand, this might be related to training env (perf/async behavior)
        # or to the actual loss/reward of the result

        # assuming the above the number of RL training step need to be (dataset_size*num_epoch we want)/32
        # current training set size is 140*10 epoch = 1400/32 = 44 RL training steps
        
        # for the paper they had 3500 frames * 100 epoch => so need 350K backprop steps
        # however the code below is 200K steps, might be a scene was not there (aka not 3500 frame to 2000)
        # if so 200k/2500 training step they have is 80 (max sample step per training?)

        # for now will use the same as an heuristic
        # 20 epoch * 140 frame = 1400 steps / 80 = 17.5 RL training steps
        total_steps=140*100, # 10 * 200 * 100 #TODO based on dataset
        pct_start=0.15,
        anneal_strategy="cos",
        div_factor=(25.0),
        final_div_factor=1e4,
        last_epoch=-1,
    )
    valid_data = Dataset()
    return model, valid_data, criterion, optimizer, lr_scheduler
