import numpy as np

import torch

#from torch.utils.data import Dataset
#import torchvision.transforms as T
#t = T.Resize((1080, 1920))
# should not be needed

## -----------------------------------------------------------------------------
## Training dataset
## -----------------------------------------------------------------------------
from PIL import Image

cs = 100

import functools

"""
gets one of the stored frames for the given frame number
Caching makes the code faster
"""
@functools.lru_cache(maxsize=cs)
def get_ith_image(path, i, frame_number):
    image = Image.open(
        path + str(i).zfill(2) + "-" + str(frame_number).zfill(4) + ".png"
    )
    return np.expand_dims(image, 0)[:, :720, :720] / 255.0


"""
gets the ground truth frame for the given frame number
Caching makes the code faster
"""


@functools.lru_cache(maxsize=cs)
def get_truth(path, frame_number):
    image = Image.open(path + "gd" + str(frame_number).zfill(4) + ".png")
    return np.array(image)[:720, :720] / 255.0


"""
gets the specified additional feature given strings that describe frame number and additional feature name.
"""


def get_add(a, b, c):
    #if b == "UVUV":
    #    b = "00UVUV"
    image = Image.open(a + b + c)
    #if np.array(image).shape[:2] == (720, 1280):
    #    image = t(image)
    z = np.array(image)[:720, :720]
    #temp = [np.sum(z[:, :, i]) for i in range(3)]
    if b == ("Depth"):
        z = np.expand_dims(z, axis=2)
        z = z[:, :, 0:1]
    z_min = np.min(z)
    z_max = np.max(z)
    if z_min == z_max:
        z_max = z_max if z_max else 1.0
        return z / z_max
        # return either 1 (if source is not 0) or 0 (source is 0) do we want this?
    else:
        return (z - z_min) / (z_max - z_min) * 1.0 
        # normalize to [0,1] (assuming source is >= 0 witch it is from png) 
        # todo check: this seens to normalize individual textur, do we really want that?

"""
gets all additional features for the given frame number
Caching makes the code faster
"""


@functools.lru_cache(maxsize=cs)
def get_aux(path, frame_number):
    f = path + "add/"
    end = str(frame_number).zfill(4) + ".png"
    imgs = np.concatenate(
        [
            get_add(f, g, end)
            for g in ["Normal", "Albedo", "Depth"]  # \
        ],
        -1,
    )
    return imgs


"""
gets the motion vector for the given frame number
Caching makes the code faster
"""


@functools.lru_cache(maxsize=cs)
def get_flow(path, frame_number):
    a = path + "Motion" + str(frame_number).zfill(4) + ".pt"
    return torch.load(a, weights_only=True)[:, :720, :720]


"""
class of the main dataset
"""


class Dataset():  # Todo rename validation dataset to something else
    def __init__(self):
        self.path = "D:/rl_dataset/"
        #self.path2 = "/home/ascardigli/blender-3.2.2-linux-x64/zeroday/"
        #self.path3 = "/home/ascardigli/blender-3.2.2-linux-x64/emerald/"
        #self.path4 = "/home/ascardigli/blender-3.2.2-linux-x64/bubble/"
        sampling = torch.arange(1, 9).reshape(8, 1, 1, 1).cuda(0)
        self.sampling = sampling.repeat(1, 1, 720, 720)
        self.num_images = 60
        self.bb = torch.Tensor(np.tile(np.arange(720), (720, 1))).cuda(0)
        self.aa = torch.Tensor(np.tile(np.arange(720).T, (720, 1)).T).cuda(0)

    def __len__(self):
        return self.num_images

    """
  Extract all path traced frames given a frame index
  """

    def data(self, index):
        samples = torch.cat(
            [
                torch.Tensor(
                    get_ith_image(self.get_path(index), i, self.get_i(index))
                ).cuda(0)
                for i in range(8)
            ],
            0,
        )
        return samples.permute(0, 3, 1, 2)[torch.randperm(8)]

    """
  Changes path as function of frame index
  """

    def get_path(self, i):
        """
        if i <= 1200:  # suntemple has 1200 images
            return self.path
        if i <= 1600:  # zeroday has 400 images
            return self.path2  # Fix TODO zeroday
        if i <= 3000:
            return self.path3  # emerald has 1400 images
        """
        return self.path

    """
  Changes scene index as function of frame index
  """
    def get_i(self, i):
        return i
        #if i <= 1200:
        #    return i
        #if i <= 1600:
        #    return i - 1200
        #if i <= 3000:
        #    return i - 1600
        #return i - 3000

    """
  Uses the motion vector to warp the given tensor
  """

    def translation(self, i, data, transform=None):
        data = data.reshape(-1, 720, 720)
        flow = get_flow(self.get_path(i) + "add/", self.get_i(i + 1)).cuda(
            0
        )  # getflow(i) gives the flow from i to i+1
        temp = torch.nn.functional.grid_sample(
            0.1 + data.unsqueeze(0), flow, align_corners=False
        )  # little hack there: grid samples gives value 0 for pixels out of grid.
        # So we shift pixels of value 0 to value .1, and then we shift back in the next two lines.
        temp[temp == 0] = -0.9  # TODO
        return (temp - 0.1).squeeze(0)

    """
  This function uses the recommendation from the network and the frames of samples, and outputs the simulated path 
  traced frames following the network's recommendation  
  """

    def generate(self, samples, idxs, i):
        samples = samples[torch.randperm(8)]
        samples = torch.cat([samples, -1 * torch.ones((1, 3, 720, 720)).cuda(0)], 0)
        idxs = idxs.reshape(1, 720, 720)
        sampling = idxs - self.sampling
        sampling[sampling < 0] = 8
        sampling = sampling.repeat(1, 3, 1, 1)
        temp = torch.take_along_dim(samples, sampling, dim=0)
        return temp

    """
  Extracts additional features and target frame for a given frame index
  """

    def get(self, index):
        target_image = get_truth(self.get_path(index), self.get_i(index))
        input_image = get_aux(self.get_path(index), self.get_i(index))
        input_image = input_image.reshape(*input_image.shape[:2], -1)
        return input_image, target_image
