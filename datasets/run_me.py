import os
import sys

path = "D:/GitRepos/rl-path-tracing/datasets/SunTemple_v4/SunTemple/SunTemple.blend"
#todo other scene and additional frames

import torch
import numpy as np
import os
import minexr
from tqdm import tqdm

def store_flow(path,out):    
    with open(path, 'rb') as fp:
      reader = minexr.load(fp)
      data = reader.select(['R','G','B'])
  
    #we want to express the motion vector so it can be fed to https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
    #in translation(self, i, data, transform=None) from dataset.py
    #grid_sample need a POSITION normalized as [-1,1],[-1,1] top left being 0,0
    #while blender provide pixels based motion vector [-xres,xres],[-yres,yres] BOTTOM left being 0,0

    xres=720
    yres=720
    #print('motion flow input shape from exr : ' + str(np.array(data).shape))
    data = torch.Tensor(np.array(data))[:xres,:yres]

    #this convert U motion from pixels [-xres, xres] to [-1,1] UV space with top left being 0,0
    data[:,:,0] = data[:,:,0]/xres
    
    #this convert V motion from pixels [-yres, yres] to [-1,1] UV space with top left being 0,0
    data[:,:,1] = -data[:,:,0]/yres

    #print('motion flow min value from exr : ' + str(torch.min(data)))
    #print('motion flow max value from exr : ' + str(torch.max(data)))
    
    xres_position = torch.Tensor(np.tile(np.arange(xres),(xres,1)))
    # buffer like this
    # 0 1 2 3 4 ... xres
    # 0 1 2 3 4 ... xres
    # ...
    xres_position = (xres_position*2-xres+1)/xres
    # buffer like this
    # -1.0 -0.99 .. 1.0
    # -1.0 -0.99 .. 1.0
    # ...

    yres_position = torch.Tensor(np.tile(np.arange(yres).T,(yres,1)).T)
    yres_position = (yres_position*2-yres+1)/yres
    #same but for the V/Y direction

    #this convert from a motion vector [-1,1],[-1, 1] top left [0,0] bottom right [1,1]
    #to a position to fetch normalized as [-1,1] top left being [-1,-1]
    data[:,:,0] = torch.Tensor(data[:,:,0]*2+xres_position)
    data[:,:,1] = torch.Tensor(data[:,:,1]*2+yres_position) 

    flow =  (torch.Tensor(data)[:,:,[0,1]].unsqueeze(0))
    torch.save(flow,out)

    #initial code:
    #print(np.array(data).shape)
    #data = torch.Tensor(np.array(data))[:720,:720]
    #print(torch.min(data))
    #a=720
    #b=720
    #bb = torch.Tensor(np.tile(np.arange(b),(720,1)))
    #aa = torch.Tensor(np.tile(np.arange(a).T,(720,1)).T)
    #data[:,:,0] = torch.Tensor((data[:,:,0]*(2)+bb*2-b+1)/b)
    #data[:,:,1] = torch.Tensor((-data[:,:,1]*(2) +aa*2-a+1)/a)
    #print(path)

if __name__ == "__main__":
    (start, stop) = sys.argv[-2:]

    motion_vector_file_path = 'd:/rl_dataset-gen/add/Motion'
    log_file_path = 'd:/rl_dataset-gen/log'

    for i in (pbar := tqdm(range(int(start),int(stop)), desc="Frame")):
        pbar.set_description(f"Processing frame {i} (1/3 - Ground truth and samples)")
        os.system("blender "+ path +" --background --python generate_gd.py -- " +str(i) + " > "+log_file_path+"/generate_gd"+str(i)+".txt")

        pbar.set_description(f"Processing frame {i} (2/3 - Additional buffers)")
        os.system("blender "+ path +" --background --python generate_add.py -- " +str(i) + " > "+log_file_path+"/generate_add"+str(i)+".txt")

        pbar.set_description(f"Processing frame {i} (3/3 - Motion vector convertion)")
        frame_number_str = str(i).zfill(4)
        store_flow(motion_vector_file_path+frame_number_str+".exr",motion_vector_file_path+frame_number_str+".pt")
