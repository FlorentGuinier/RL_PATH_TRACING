import unet2
import unet1small
import unet1big
import env
import time
import ray.rllib.algorithms.appo as appo
from ray.rllib.algorithms.appo import APPOConfig
import torch
import sys
from tqdm import tqdm

config = (
    APPOConfig()
    .framework("torch")
    .resources(num_gpus=1, num_cpus_per_worker=64, num_gpus_per_worker=1)
    .environment(normalize_actions=False, env=env.CustomEnv)
    .rollouts(num_envs_per_worker=1, num_rollout_workers=1, rollout_fragment_length=4, no_done_at_end=False)
    .training(use_critic=True, use_gae=True, use_kl_loss=True, kl_coeff=5e-7, kl_target=5e-8, lambda_=0.2, clip_param=0.15, grad_clip=4, #appo
              lr=1e-3, train_batch_size=4, model={"custom_model": "UN"}, #optimizer="adabelief", #generic
              vtrace_drop_last_ts=False, replay_buffer_num_slots=40, vf_loss_coeff=0.5, entropy_coeff=1e-5 #impala
              )
    )

config_old = {
    "framework": "torch",             #DONE ported to framework   in 2.2
    "vtrace_drop_last_ts": False,     #DONE ported to training    in 2.2 impala specific              => not default
    "optimizer": "adabelief",         #NOT PORTED seems useless? 
    "use_critic": True,               #DONE ported to training    in 2.2 APPO specific                => DEFAULT
    "use_gae": True,                  #DONE ported to training    in 2.2 APPO specific                => DEFAULT
    "use_kl_loss": True,              #DONE ported to training    in 2.2 APPO specific                => not default
    "kl_coeff": 5e-7,                 #DONE ported to training    in 2.2 APPO specific                => not default
    "kl_target": 5e-8,                #DONE ported to training    in 2.2 APPO specific                => not default
    "lambda": 0.2,                    #DONE ported to training    in 2.2 APPO specific                => not default
    "clip_param": 0.15,               #DONE ported to training    in 2.2 APPO specific                => not default
    "lr": 1e-3,                       #DONE ported to training    in 2.2 generic                      => DEFAULT
    "grad_clip": 4,                   #DONE ported to training    in 2.2 APPO (but exist in impala)   => not default
    "replay_buffer_num_slots": 40,    #DONE ported to training    in 2.2 impala specific              => not default
    "vf_loss_coeff": 0.5,             #DONE ported to training    in 2.2 impala specific              => DEFAULT
    "entropy_coeff": 1e-5,            #DONE ported to training    in 2.2 impala specific              => not default
    "normalize_actions": False,       #DONE ported to environment in 2.2                              => not default
    "train_batch_size": 4, # Was 128  #DONE ported to training    in 2.2 generic                      => not default
    "no_done_at_end": False,          #DONE ported to rollout     in 2.2                              => DEFAULT
    "num_envs_per_worker": 1,         #DONE ported to rollouts    in 2.2                              => DEFAULT
    "num_workers": 1,   #Was 8        #DONE ported to rollouts    in 2.2                              => not default ALSO is called num_rollout_workers in new 2.2 code see https://github.com/ray-project/ray/pull/29546/files
    "num_gpus": 4,                    #DONE ported to ressource   in 2.2                              => not default
    "num_cpus_per_worker": 48,        #DONE ported to ressource   in 2.2                              => not default
    "num_gpus_per_worker": 4,         #DONE ported to ressource   in 2.2                              => not default
    "rollout_fragment_length": 4,     #DONE ported to ressource   in 2.2                              => not default
    "model":   {"custom_model": "UN"},#DONE ported to training    in 2.2 generic                      => not default
#    "evaluation_interval":2500,
#    "evaluation_duration": 100 
}

def train_ppo_model(
    spp=4, mode="", conf="111", interval=[35, 40] #interval=[700, 800] #TODO based on dataset
):
    """Main code for our framework: Networks/agents get initialized and trained, possibly concurently

    Args:
        spp (int, optional): the spp count. Defaults to 4.
        mode (str, optional): the variant/baseline being run. Defaults to "".
        conf (str, optional): the scale of every of the 3 network, where 0 is the scaled down version,
            1 is the default, and 2 is the large version. Defaults to "111".
        interval (list, optional): the validation interval. Defaults to [700, 800].
    """
    trainours = mode == "vanilla" or mode == "notp" or mode == "notp1" or mode=="D" #all of our variants that require RL training
    spp = float(spp)
    algos = {}
    if trainours:
        def a(x):
         x.eval()
         return x
        def b(x):
         x.endeval()
         return x
        config["env_config"] = {
            "spp": spp,
            "mode": mode,
            "conf": conf,
            "interval": interval,
        }

        #algos[mode] = appo.APPO(env=env.CustomEnv, config=config)
        algos[mode] = config.build()

        #FLO change iteration here was 2500 #TODO based on dataset
        for i in tqdm(range(10)): #hardcoded number of iterations that corresponds to the wanted number of epochs
#            if i==config["evaluation_interval"]-1:
#             algos[mode].workers.foreach_env(a)
#             algos[mode].train()
#             algos[mode].workers.foreach_env(b)
#            else:
             #print("******************** Train loop :" + str(i) + "*********************")
             algos[mode].train()
    else:
        modes = mode
        if not isinstance(modes,list):
            modes = [modes]
        sims = {}
        unets = {}
        for e, mode in enumerate(modes):
            sims[mode] = env.CustomEnv(
                {
                    "spp": spp,
                    "mode": mode,
                    "conf": conf,
                    "interval": interval,
                }
            )
            unet = unet2.UN(
                sims[mode].observation_space, sims[mode].action_space, None, None, None
            )
            unet.model_config = model
            unet.obs_space = sims[mode].observation_space
            unet.action_space = sims[mode].action_space
            unets[mode] = unet

        def f(x, mode):
            """returns the sampling importance recommendation

            Args:
                x (Tensor): the input of the sampling importance network
                mode (str): the mode description

            Returns:
                tensor: the sampling importance recommendation
            """            
            if "uni" in mode:
                return (torch.ones([1, 720, 720]).cuda(0) * spp).type(torch.int)
            else:
                b, _ = unets[mode](x, None, None)
                b = b.reshape(1, 720, 720).cuda(0)
                return b

        def g(x, e):
            """helper function to convert data

            Args:
                x (Tensor): the observation state
                e (int): On which gpu to transfer the data  

            Returns:
                _type_: a dictionary where the observation has the good type
            """            
            return {"obs": torch.Tensor(x).unsqueeze(0).cuda(e)}

        for i in range(2500):
            for e, mode in enumerate(modes):
                unets[mode] = unets[mode].cuda(e)
                a = sims[mode].reset()
                b = f(g(a, e), mode)
                for _ in range(20):
                    a, b, _, _ = sims[mode].step(b)
                    b = f(g(a, e), mode)
                unets[mode] = unets[mode].cpu()


"""
mode guide:
vanilla: our main method
grad: oursA1
uni: oursA2
notp1: oursB1
notp: oursB2
D: oursC
dasr: DASR
ntas: NTAS
imcduni: IMCD
"""
if __name__ == "__main__":
    te = sys.argv[-5:]
    spp, mode, conf, i1, i2 = ("4.0", "vanilla", "111", "35", "40")
    #spp, mode, conf, i1, i2 = te
    if conf[0] == "0":
        config["model"]["custom_model"] = "UNsmall"
    if conf[0] == "2":
        config["model"]["custom_model"] = "UNbig"

    train_ppo_model(spp, mode, conf, [int(i1), int(i2)])
