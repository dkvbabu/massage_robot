
import os
import random
import time
from dataclasses import dataclass

from env import MassageEnv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro

from ppo import Agent

def main():

    env = MassageEnv(render=True,auto_reset=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(env).to(device)
    model_name = 'runs/MassageEnv__ppo__1__1749045480/ppo.cleanrl_model'
    agent.load_state_dict(torch.load(model_name,map_location=device))

    state = env.reset()
    tests_n = 2
    for i in range(env.episode_length*tests_n):

        with torch.no_grad():

            vec_state = torch.as_tensor(state,device=device)[None,:]
            change = agent.actor_mean(vec_state.float())[0].cpu().numpy()
        

        state,reward,done,info = env.step(change=change)
        if done:
            if i<((tests_n*env.episode_length)-1):
                print(i)
                env.reset()
            print(f'Episodic rewards: {info["episode"]['r']}')

        time.sleep(0.01)
    env.close()



if __name__=='__main__':

    main()