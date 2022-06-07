'''
Illustrates reinforcement learning using the beam selection problem.
Use tensorboard to visualize training convergence.
ICC Tutorial - June 14, 2021
Tutorial 14: Machine Learning for MIMO Systems with Large Arrays
Aldebaro Klautau (UFPA),
Nuria Gonzalez-Prelcic (NCSU) and
Robert W. Heath Jr. (NCSU)
From: https://github.com/lasseufpa/ml4comm-icc21/
'''
from mushroom_rl.policy import EpsGreedy
from mushroom_rl.utils.parameters import Parameter
from mushroom_rl.algorithms.value import  SARSA , QLearning
from mushroom_rl.core import Core
from env_mimo_rl_simple import Mimo_RL_Simple_Env
import matplotlib.pyplot as plt
import numpy as np
import time
import resource

num_antenna_elements=32 #number of antennas at base station
grid_size=6 #world is a grid_size x grid_size grid
total_timesteps=30 #training total steps

# Defines the environment
mdp = Mimo_RL_Simple_Env(num_antenna_elements=num_antenna_elements, 
        grid_size=grid_size)

# Defines Policy and Learning rate  
epsilon = Parameter(value=1.)
policy = EpsGreedy(epsilon=epsilon)
learning_rate = Parameter(value=.6)

#---------------SARSA------------------------

#Define agent 
agent = SARSA(mdp, policy, learning_rate)

#Core modified
core = Core(agent, mdp)

#Starting counting 
time_start = time.perf_counter()

#Train
core.learn(n_episodes=64, n_steps_per_fit=1)


# -----------QLearning ----------------------

#Define agent 
# agent = QLearning(mdp, policy, learning_rate)

# #Core modified 
# core = Core(agent, mdp)

# #Starting counting 
# time_start = time.perf_counter()

# #Train
# core.learn(n_episodes=64, n_steps_per_fit=1)


#Plot the Computational time x Memory usage 

# time_elapsed = (time.perf_counter() - time_start)
# memMb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0/1024.0
# print("%5.1f secs %5.1f MBytes" % (time_elapsed, memMb))

# ct = time_elapsed
# m = memMb

# labels = ['QLearning']
# men_means = ct
# women_means = m

# x = np.arange(len(labels))  # the label locations
# width = 0.1  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, men_means, width, label='Computational time (sec)')
# rects2 = ax.bar(x + width/2, women_means, width, label='Memory usage (MBytes)')

# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_title('MushroomRL Beam Selection environment')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()

# ax.bar_label(rects1, padding=2)
# ax.bar_label(rects2, padding=2)

# fig.tight_layout()

# plt.show()
