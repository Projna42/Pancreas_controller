#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import gym
import numpy as np

from stable_baselines.sac.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

import gym
import simglucose

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# Register gym environment. By specifying kwargs,
# you are able to choose which patient to simulate.
# patient_name must be 'adolescent#001' to 'adolescent#010',
# or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
from gym.envs.registration import register

register(
    id='simglucose-adolescent2-v0',
    entry_point='simglucose.envs:T1DSimEnv',
    kwargs={'patient_name': 'adolescent#002'}
)
env = gym.make('simglucose-adolescent2-v0')



# Custom MLP policy of three layers of size 128 each
class CustomSACPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomSACPolicy, self).__init__(*args, **kwargs,
                                           layers=[128, 128, 128],
                                           layer_norm=False,
                                           feature_extraction="mlp")
        
# Create and wrap the environment
env = DummyVecEnv([lambda: env])

model = SAC(CustomSACPolicy, env, verbose=1)
# Train the agent
# model.learn(total_timesteps=100000)
    
# model = SAC.load("sac_simglucose")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)


# del model # remove to demonstrate saving and loading
person_options = (['child#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
                  ['adolescent#0{}'.format(str(i).zfill(2)) for i in range(1, 11)] +
                  ['adult#0{}'.format(str(i).zfill(2)) for i in range(1, 11)])
for i,p in enumerate(person_options):
    patient_id = p.split('#')[0] + str(i + 1)
    
    register(
        id='simglucose-' + patient_id + '-v0',
        entry_point='simglucose.envs:T1DSimEnv',
        kwargs={'patient_name': p}
    )
    
    env = gym.make('simglucose-' + patient_id + '-v0')
    print(p, patient_id)
    env = DummyVecEnv([lambda: env])
    model.learn(total_timesteps=10)
    
model.save("sac_simglucose")

obs = env.reset()
for i in range(10):
    print(obs)
    action, _states = model.predict(obs)
    print(action, _states)
    obs, rewards, dones, info = env.step(action)
    print(obs, rewards, dones, info)
#     env.render()


# In[ ]:




