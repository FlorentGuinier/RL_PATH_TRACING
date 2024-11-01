import os

path = "D:/GitRepos/rl-path-tracing/datasets/SunTemple_v4/SunTemple/SunTemple.blend"
#todo other scene and additional frames
#len = 1201
len = 2

import torch
import numpy as np
import os
import minexr

def store_flow(path,out):    
    with open(path, 'rb') as fp:
      reader = minexr.load(fp)
      data = reader.select(['R','G','B','A'])
  
    print(np.array(data).shape)
    data = torch.Tensor(np.array(data))[:720,:720]
    print(torch.min(data))
    a=720
    b=720
    bb = torch.Tensor(np.tile(np.arange(b),(720,1)))
    aa = torch.Tensor(np.tile(np.arange(a).T,(720,1)).T)
    data[:,:,0] = torch.Tensor((data[:,:,0]*(2)+bb*2-b+1)/b)
    data[:,:,1] = torch.Tensor((-data[:,:,1]*(2) +aa*2-a+1)/a)

    flow =  (torch.Tensor(data)[:,:,[0,1]].unsqueeze(0))
    torch.save(flow,out)
    print(path)

motion_vector_file_path = 'd:/rl_dataset/add/Motion'

for i in range(0,len):
    print("* generate_gd: individual samples and ground truth ******************************")
    os.system("blender "+ path +" --background --python generate_gd.py -- " +str(i) )   

    print("* generate_add: normal, depth, albedo and motion vector as exr ******************")
    os.system("blender "+ path +" --background --python generate_add.py -- " +str(i))

    print("* convert motion vector to torch format *****************************************")
    frame_number_str = str(i).zfill(4)
    store_flow(motion_vector_file_path+frame_number_str+".exr",motion_vector_file_path+frame_number_str+".pt")
    