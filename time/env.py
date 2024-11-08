#import time
import gym
import numpy as np
#import torchvision.transforms as T
from gym import spaces
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.functional import mean_squared_error
from physicalsimulation import PhysicSimulation
from training import train
from training_logger import training_logger_writer

psnr = PeakSignalNoiseRatio().cuda(0)

class Spec:
    def __init__(self, max_episode_steps):
        """this class has no utility but is needed by RayRLlib

        Args:
            max_episode_steps (int): the maximum number of episode per animation
        """        
        self.max_episode_steps = max_episode_steps
        self.id = "3Drenderingenv"


class CustomEnv(gym.Env):
    """this is an RL environment, accessed by the ray rllib library
    """
    metadata = {"render.modes": ["human"]} # from gym.Env

    def __init__(self, env_config):
        """initializes needed variables, including state and action spaces, the denoising model, 
        the Physical simulation instance and other needed variables ...

        Args:
            env_config (dict): the configuration needed to intialize the environment.
        """        
        super(CustomEnv, self).__init__()
        self.log_debug = False
        if self.log_debug:
            print("CustomEnv init")
        self.spp = env_config["spp"]  # Samples Per Pixel
        self.mode = env_config["mode"]  # The algorithm used (ours, ablation study, baseline...)
        self.conf = env_config["conf"]
        self.interval = env_config["interval"] # The validation interval
        self.HEIGHT = 720  # side size of the tile
        self.WIDTH = 720
        self.max = 20  # number of frames per animation
        self.train=True
        self.validating = False
        out_space = ( # out_space corresponds to the number of channels of the input of the sampling importance network 
            32 + 7
        )  # our sampling importance network takes additional data (7 chans) and the temporal latent state (32 chans) as input
        if (
            self.mode == "ntas"
        ):  # ntas-denoising takes additional data (7 chans) plus one image (3 chans), and temporal state (3 chans)
            (
                self.model,
                self.data,
                self.criterion,
                self.optimizer,
                self.scheduler,
            ) = train.main_worker(10, 3, conf=self.conf, mode=self.mode)
            out_space = 10  # adaptive rendering takes additional data (7 chans) plus temporal state (3 chans)
        elif (
            "notp" == self.mode
        ):  # ours without temporal state -denoising takes additional data (7 chans) plus current rendered images (24 chans)
            (
                self.model,
                self.data,
                self.criterion,
                self.optimizer,
                self.scheduler,
            ) = train.main_worker(31, 0, conf=self.conf)
            out_space = 7  # adaptive rendering takes additional data (7 chans) but does not include the previous rendered image (no temporal feedback)
        elif (
            "notp1" == self.mode
        ):  # ours without temporal state -denoising takes additional data (7 chans) plus current rendered images (24 chans)
            (
                self.model,
                self.data,
                self.criterion,
                self.optimizer,
                self.scheduler,
            ) = train.main_worker(31, 0, conf=self.conf)
            out_space = 31  # adaptive rendering takes additional data (7 chans) plus previous rendered images (24 chans) in this variational case
        elif (
            self.mode == "dasr"
        ):  # dasr-denoising takes addition data (7 chans) plus one image (3 chans)
            (
                self.model,
                self.data,
                self.criterion,
                self.optimizer,
                self.scheduler,
            ) = train.main_worker(10, 0, conf=self.conf, mode=self.mode)
            out_space = 7  # adaptive rendering takes addition data (7 chans)
        elif (
            self.mode == "imcduni"
        ):  # imcd denoiser takes previous image (3 chans) and latent encoding (32)
            (
                self.model,
                self.data,
                self.criterion,
                self.optimizer,
                self.scheduler,
            ) = train.main_worker(35, 0, conf=self.conf, mode=self.mode)
        elif self.mode == "D": # ours C-denoiser takes current averaged image (3 chans), additional data (7 chans), and latent encoding (32)
            (
                self.model,
                self.data,
                self.criterion,
                self.optimizer,
                self.scheduler,
            ) = train.main_worker(10, 32, conf=self.conf, mode=self.mode)
        else:  # The default case-denoising is to take additional data (7 chans) plus current renderd images (24 images), and temporal state (32 chans)
            (
                self.model,
                self.data,
                self.criterion,
                self.optimizer,
                self.scheduler,
            ) = train.main_worker(conf=self.conf)

        self.model = self.model.to("cuda:0")
        self.criterion = self.criterion.to("cuda:0")
        self.offset = 0

        self.simulation = PhysicSimulation(self)

        #from gym.Env
        self.action_space = spaces.Box(
            low=0, high=1, shape=(int(self.HEIGHT * self.WIDTH),) 
        )
        #from gym.Env
        self.observation_space = spaces.Box(
            low=-1.0001,
            high=1.0001,
            shape=(out_space, int(self.HEIGHT), int(self.WIDTH)),
            dtype=np.float32,
        )  # MACHINE PRECISION

        self.spec = Spec(self.max) #from gym.Env
        
        self.mses = []
        self.psnrs = []
        with open(
            "comp/"
            + str(self.spp)
            + "mses"
            + self.mode
            + "-"
            + str(self.interval[0])
            + ".txt",
            "w",
        ) as fp:
            fp.write("\n")
        with open(
            "comp/"
            + str(self.spp)
            + "psnrs"
            + self.mode
            + "-"
            + str(self.interval[0])
            + ".txt",
            "w",
        ) as fp:
            fp.write("\n")
        if self.log_debug:
            print("CustomEnv init - done")

    """
    def endeval(self):
        self.train=True
        self.offset=self.tempoffset
    def eval(self):
        self.train=False
        self.tempoffset=self.offset
        self.offset = self.interval[0] - 140 #TODO based on dataset
    """

    def step(self, action):
        """ Performs a step of the RL agent: given action, it updates the state, computes the reward,
        and outputs the observation for the next step. 
        This is also where results are logged. 

        Args:
            action (numpy array): the recommendation from the sampling importance network

        Returns:
            tuple: observation, reward, boolean of whether or not the episode is done
        """        
        if self.log_debug:
            print("CustomEnv step")

        self.simulation.new(
            self.simulation.count
        )  # we initialize the simulation for this iteration
        self.simulation.simulate(
            action
        )  # we get the pixel colors given the sampling hitmap
        new = self.simulation.render().reshape(-1,self.HEIGHT,self.WIDTH)
         
        (
            observation,
            gd,
        ) = self.simulation.observe()  # we get the output image and ground truth
        loss = self.simulation.loss # we compute the loss and get it
        new1 = 1 - loss
        mse_value = mean_squared_error(new.reshape(-1), gd.reshape(-1)).cpu()
        psnr_value = psnr(new, gd).cpu()
        training_logger_writer.add_scalar("metrics/mse", mse_value, PhysicSimulation.total_step)
        training_logger_writer.add_scalar("metrics/psnr", psnr_value, PhysicSimulation.total_step)
        if self.validating:  # if in validation set, record mse and psnrs
            self.mses.append(mse_value)
            self.psnrs.append(psnr_value)

        reward = 10 ** (
            new1
        )  # We transform the loss, this showed to work better for the learning of the RL agent
        training_logger_writer.add_scalar("metrics/reward", reward, PhysicSimulation.total_step)
        done = self.spec.max_episode_steps <= self.simulation.count
        if self.log_debug:
            print("CustomEnv step - done")
        return observation.numpy(), reward.detach().numpy(), done, {}

    def reset(self):
        """Resets the agent and the physical simulation. This happens after/before every animation (20 subsequent frames)

        Returns:
            numpy array: the initial observation for the next animation 
        """        
        if self.log_debug:
            print("CustomEnv reset")

        self.simulation = PhysicSimulation(self)  # reset the simulation
        temp, _ = self.simulation.observe()

        #if last context (before reset was validating, dump info)
        if self.validating:
            with open(
                "comp/"
                + str(self.spp)
                + "mses"
                + self.mode
                + "-"
                + str(self.interval[0])
                + ".txt",
                "a",
            ) as fp:
                fp.write("\nOffset " + str(self.simulation.offset))
                fp.write("\nStep " + str(PhysicSimulation.scheduler_step))
                fp.write("\n")
                fp.write("\n".join(str(item.item()) for item in self.mses))
                fp.write("\n")
            with open(
                "comp/"
                + str(self.spp)
                + "psnrs"
                + self.mode
                + "-"
                + str(self.interval[0])
                + ".txt",
                "a",
            ) as fp:
                fp.write("\nOffset " + str(self.simulation.offset))
                fp.write("\nStep " + str(PhysicSimulation.scheduler_step))
                fp.write("\n")
                fp.write("\n".join(str(item.item()) for item in self.psnrs))
                fp.write("\n")

        offset_before = self.offset
        self.offset += (
            21  # we have a rolling list of 4 animations to increase cache usage.
        )
        if self.offset % 20 == 4:
            self.offset -= 64
        if self.log_debug:
            print("CustomEnv reset - done. Offset " +str(offset_before) +" => " + str(self.offset))

        self.validating = False
        if self.simulation.inval(
            (self.offset % self.simulation.number) // 20 * 20
        ): #and not self.train:  
            self.validating = True
        self.mses = []
        self.psnrs = []

        return temp.numpy()

