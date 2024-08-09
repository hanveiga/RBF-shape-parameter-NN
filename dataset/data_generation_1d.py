import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle

import sys
sys.path.append('../common/')

from rbf_tools import *
from optimisation import *
from one_dim import *
from two_dim import *

p_ref=np.array([0])

data_per_sample=700


storage = []
nunsuccessful = 0
dataset = []


dimensions = [1]

a = 0 # lower limit of hypercube

for lr_init, b in zip([0.1,0.1,0.1],[0.01,0.1,1.0]):
    freq=2/b
    for u_exact in [lambda x: np.exp(np.sin(np.pi*x)), lambda x: 1/(1+16*(x**2)), lambda x: np.cos(freq*np.pi*x) ]:
    
        for n in dimensions:
            print(f"Dimension: {n}")
            for i in range(data_per_sample):
                x_axis = sample_from_hypercube(n,low=a, high=b, n=10)
                x_axis = sorted( x_axis, key = lambda x: np.linalg.norm(x - p_ref ) )
                x_axis = np.reshape(x_axis, (10,n)) 

                eps_init = initial_guess(x_axis)
                A =  get_int_matrix(x_axis,eps_init)
                cond_init = np.linalg.cond(A,'fro')
            
                try:
                    initial_error, curves_initial = compute_error(x_axis,a,b,eps_init,u_exact)
                except:
                    print("couldn't start")
                    break
                epss = []
                loss = []
                eps = eps_init
                j = 0
                closs = 10.
                trials = 0
                total_iter = 0
            
                lr_init_inside = lr_init
                eps_init_inside = eps_init
                unsuccessful = 0
                while np.abs(closs) > 10**(-3):
                    eps, closs, lr_init_inside, iters = optimization_loop_ADAM(x_axis, eps_init_inside, lr_init_inside)
                    total_iter += iters
                    if np.abs(closs) < 10**(-3):
                        break
                    
                    # check if the condition is too large or too small:
                    # we can improve on this, to make it converge
                    A =  get_int_matrix(x_axis,eps_init)
                    cond = np.linalg.cond(A,'fro')
                    if np.log10(cond) > 12:
                        eps_init_inside = eps_init_inside*2.0
                    if np.log10(cond) < 10:
                        eps_init_inside = eps_init_inside*0.5
                    print(np.log10(cond))

                    trials = trials + 1
                    if trials > 40:
                        print("unsucessful")
                        unsuccessful = 1
                        nunsuccessful +=1
                        break
            
                if unsuccessful==0:
                    A =  get_int_matrix(x_axis,eps)
                    cond = np.linalg.cond(A)
                    final_error, curves_final = compute_error(x_axis,a,b,eps,u_exact)
                    #plot_curves(curves_initial, curves_final,name=f"{b}_{i}")

                    x_axis_flatten=x_axis.flatten()

                    X_data=np.zeros(2*x_axis_flatten.shape[0])
                    X_data[::2]=x_axis_flatten
                    
                    dataset.append([X_data, eps])
                    print(f"niter: {total_iter}, initial eps: {eps_init}, end eps: {eps}, end loss: {closs}, \
                          initial cond: {np.log10(cond_init)}, end cond: {np.log10(cond)},\
                          \n initial error:{initial_error}, final error: {final_error}")
                    
pickle.dump(dataset, open("sort_1d_11_11.5.pkl",'wb'))
print(dataset)


