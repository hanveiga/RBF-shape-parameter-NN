import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
from rbf_tools import *
from optimisation import *
from one_dim import *
from two_dim import *

def calculate_shape(x_axis,lr_init):
    
    eps_init = initial_guess(x_axis)
    AA =  get_int_matrix(x_axis,eps_init)
    cond_init = np.linalg.cond(AA,'fro')
    
    eps = eps_init
    closs = 10.
    trials = 0
    total_iter = 0            
    lr_init_inside = lr_init
    eps_init_inside = eps_init
    nunsuccessful = 0
    while np.abs(closs) > 10**(-3):
        eps, closs, lr_init_inside, iters = optimization_loop_ADAM(x_axis, eps_init_inside, lr_init_inside)
        total_iter += iters

        
        if np.abs(closs) < 10**(-3):
            break                    
        # check if the condition is too large or too small:
        A =  get_int_matrix(x_axis,eps_init)
        cond = np.linalg.cond(A,'fro')
        
        if np.log10(cond) > 12:
            eps_init_inside = eps_init_inside*2.0
        if np.log10(cond) <10:
            eps_init_inside = eps_init_inside*0.5
                
        trials = trials + 1
        if trials > 40:
            print("unsucessful")
          
            nunsuccessful +=1
            break
        
        
        
    return eps,nunsuccessful 
            
