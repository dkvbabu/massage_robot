
import os
import random
import time
from dataclasses import dataclass

from env_ import MassageEnv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro

from ppo import Agent
from ddpg import ActorDDPG
from TD3_train import Actor

def process_action(action,env,smooth_target_prev = None, alpha=0.01,t=1,prev_action=[0,0,0]):

    action = action + np.random.normal(0, 0.1, size=3)

    min_bounds = np.min(env.pntsAndReturn, axis=0) 
    max_bounds = np.max(env.pntsAndReturn, axis=0) 

    workspace_range = max_bounds - min_bounds
    action = min_bounds + (action + 1) / 2 * workspace_range

    # Oscillate x target back and forth
    x_min = np.min(env.pntsAndReturn[:, 0])
    x_max = np.max(env.pntsAndReturn[:, 0])
    x_range = x_max - x_min
    oscillation_period = env.episode_length / 2
    x_oscillate = x_min + (x_range / 2) * (1 + np.sin(2 * np.pi * t / oscillation_period))

    y_fixed = 0.3
    z_fixed = 0.95

    action[0] = x_oscillate
    action[1] = y_fixed  # fixed y since min and max are equal
    action[2] = np.clip(action[2], z_fixed - 0.05, z_fixed + 0.05)

    # Smooth action blending
    action_smooth = 0.8 * prev_action + 0.2 * action
    prev_action = action_smooth

    # Smooth target interpolation
    if smooth_target_prev is None:
        smooth_target = action_smooth
    else:
        smooth_target = alpha * action_smooth + (1 - alpha) * smooth_target_prev
    smooth_target_prev = smooth_target
    smooth_target = np.clip(smooth_target, min_bounds, max_bounds)

    return smooth_target_prev,smooth_target,prev_action

def main():

    NoRL = False
    PPOModel = True
    EnvNoise = False
    env = MassageEnv(render=False,auto_reset=False)#,train=EnvNoise)
    #state = env.get_state()
    #state_dim = len(state)  # This will now include the 5 new features
    #action_dim = 3
    #max_action = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if PPOModel:
        agent = Actor(27,3,1.0).to(device)
        #model_name = 'runs/MassageEnv__ppo__1__1749212709/ppo.cleanrl_model'
        #model_name = 'runs/MassageEnv__ppo__1__1749561468/ppo.cleanrl_model' # with noise
        model_name = 'actor_episode_800.pth' # no noise
        agent.load_state_dict(torch.load(model_name,map_location=device))
    else:
        #DDPG
        agent = ActorDDPG(env).to(device)
        model_name = 'runs/MassageEnv_DDPG__ddpg__1__1749564959/ddpg.cleanrl_model' # with noise
        #model_name = 'runs/MassageEnv_DDPG__ddpg__1__1749725147/ddpg.cleanrl_model' # no noise
        agent.load_state_dict(torch.load(model_name,map_location=device)[0])
    #agent = Actor(state_dim,action_dim,max_action).to(device)
    #model_name = 'runs/MassageEnv__ppo__1__1749045480/ppo.cleanrl_model'



    state = env.reset()
    all_returns = []
    tests_n = 100
    smooth_target_prev = None
    prev_action = np.zeros(3)
    for i in range(env.episode_length*tests_n):

        state = state / np.linalg.norm(state) if np.linalg.norm(state) > 0 else state

        with torch.no_grad():
            vec_state = torch.as_tensor(state,device=device)[None,:]

            if NoRL:
                change = np.zeros(7)
            else:
                if PPOModel:
                    action = agent(vec_state.float()).cpu().numpy()[0]
                    smooth_target_prev,action,prev_action = process_action(action,env,smooth_target_prev = smooth_target_prev,
                                                                alpha=0.01, t= i, prev_action = prev_action )
                    base_action = env.get_action()
                    change = action - base_action
                    #change = np.hstack((change,np.zeros(4)[None,:]))[0]
                else:
                    change = agent(vec_state.float())[0].cpu().numpy()

        #print(change)

        state,reward,done,info = env.step(change)
        if ((i+1)%(env.episode_length))==0:#done:
            print(f'Episodic rewards: {env.epReturn}')
            all_returns.append(env.epReturn)
            if i<((tests_n*env.episode_length)-1):
                print(i)
                state = env.reset()
                smooth_target_prev = None
            #print(f'Episodic rewards: {info["episode"]['r']}')
            #all_returns.append(info["episode"]['r'])

        #time.sleep(0.02)
    
    print(f'Case {['DDPG','PPO'][PPOModel]} with{['out',''][EnvNoise]} noise:')
    print(f'Mean Return: {np.array(all_returns).mean()}')
    print(f'STD Return: {np.array(all_returns).std()}')
    env.close()



if __name__=='__main__':

    main()