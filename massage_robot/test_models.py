
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
from ddpg import ActorDDPG
from TD3_train import Actor

def main():

    NoRL = True
    PPOModel = False
    EnvNoise = False
    env = MassageEnv(render=True,auto_reset=False,train=EnvNoise)
    #state = env.get_state()
    #state_dim = len(state)  # This will now include the 5 new features
    #action_dim = 3
    #max_action = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if PPOModel:
        agent = Agent(env).to(device)
        #model_name = 'runs/MassageEnv__ppo__1__1749212709/ppo.cleanrl_model'
        #model_name = 'runs/MassageEnv__ppo__1__1749561468/ppo.cleanrl_model' # with noise
        model_name = 'runs/MassageEnv__ppo__1__1749568040/ppo.cleanrl_model' # no noise
        agent.load_state_dict(torch.load(model_name,map_location=device))
    else:
        #DDPG
        agent = ActorDDPG(env).to(device)
        #model_name = 'runs/MassageEnv_DDPG__ddpg__1__1749564959/ddpg.cleanrl_model' # with noise
        model_name = 'runs/MassageEnv_DDPG__ddpg__1__1749725147/ddpg.cleanrl_model' # no noise
        agent.load_state_dict(torch.load(model_name,map_location=device)[0])
    #agent = Actor(state_dim,action_dim,max_action).to(device)
    #model_name = 'runs/MassageEnv__ppo__1__1749045480/ppo.cleanrl_model'



    state = env.reset()
    all_returns = []
    tests_n = 1
    for i in range(env.episode_length*tests_n):

        with torch.no_grad():
            vec_state = torch.as_tensor(state,device=device)[None,:]

            if NoRL:
                change = np.zeros(7)
            else:
                if PPOModel:
                    change = agent.actor_mean(vec_state.float())[0].cpu().numpy()
                else:
                    change = agent(vec_state.float())[0].cpu().numpy()


        state,reward,done,info = env.step(change)
        if ((i+1)%(env.episode_length))==0:#done:
            print(f'Episodic rewards: {env.epReturn}')
            all_returns.append(env.epReturn)
            if i<((tests_n*env.episode_length)-1):
                print(i)
                state = env.reset()
            #print(f'Episodic rewards: {info["episode"]['r']}')
            #all_returns.append(info["episode"]['r'])

        time.sleep(0.03)
    
    print(f'Case {['DDPG','PPO'][PPOModel]} with{['out',''][EnvNoise]} noise:')
    print(f'Mean Return: {np.array(all_returns).mean()}')
    print(f'STD Return: {np.array(all_returns).std()}')
    env.close()



if __name__=='__main__':

    main()