import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from datetime import datetime
import math
from math import exp
from random import seed
from random import random
from scipy.spatial import distance
import pickle

import sys
sys.path.append('../common/')

from rbf_tools import *
from optimisation import *
from one_dim import *
from two_dim import *
from final import calculate_shape

def phi(f, x, y):
    z = (1 + (f * np.linalg.norm(x-y)) ** 2) ** (-0.5)
    return z


data_per_sample=50
dataset = []
nunsuccessful = 0
dimensions = [2]
a=0
N=20 # number of points in domain
n=10 # length of stencil
p_ref=np.array([0,0])

for lr_init, b in zip([0.1,0.1,0.1,0.1],[0.001,0.01,0.1,1.0]):
    for u_exact in [lambda x,y,alpha=0.1: (1+np.exp(-1.0/alpha)-np.exp(-x/alpha)-np.exp((x-1.0)/alpha))*(1+np.exp(-1.0/alpha)-np.exp(-y/alpha)-np.exp((y-1.0)/alpha)),\
                    lambda x,y,alpha=1.0: (1+np.exp(-1.0/alpha)-np.exp(-x/alpha)-np.exp((x-1.0)/alpha))*(1+np.exp(-1.0/alpha)-np.exp(-y/alpha)-np.exp((y-1.0)/alpha)),\
                    lambda x,y: 0.75*np.exp(-(9.0*x-2.0)**2/4.0-(9.0*y-2.0)**2/4.0)+0.75*np.exp(-(9.0*x+1.0)**2/49.0-(9.0*y+1.0)**2/10.0)+0.5*np.exp(-(9.0*x-7.0)**2/4.0-(9.0*y-3.0)**2/4.0)-0.2*np.exp(-(9.0*x-4.0)**2-(9.0*y-7.0)**2)]:
        for n_dim in dimensions:
            print(f"Dimension: {n_dim}")

            for item in range(data_per_sample):
                error1=0
                error2=0
                index_tot=[]
                
                x=sample_from_hypercube(n_dim,low=a, high=b, n=N) 
                x = sorted( x, key = lambda x: np.linalg.norm(x - p_ref ) )
                x = np.reshape(x, (N,n_dim)) 

                number_points= x.shape[0]

                distance = get_distance_matrix(x)

                closest = np.argsort(distance, axis=1)
                for i in range(number_points):
                    x_local=np.zeros((n,2))
                    for j in range(n):
                        x_local[j]=x[closest[i][j]]
                        index_sorted=sorted(closest[i][:n])
                    if index_sorted not in index_tot:
                        index_tot.append(index_sorted)
                        
                        eps_init = initial_guess(x_local)
                        L =  get_int_matrix(x_local,eps_init)
                        cond_init = np.linalg.cond(L,'fro')
                        eps_final,nunsuccessful=calculate_shape(x_local,lr_init)

                        if nunsuccessful==0:
                            L =  get_int_matrix(x_local,eps_final)
                            cond = np.linalg.cond(L,'fro')
                            print(f"initial condition: {np.log10(cond_init)} \
                                  final condition: {np.log10(cond)} x-local: {x_local}\
                                  eps init: {eps_init} eps final: {eps_final}")
                            #final_error=compute_error_2d(x_local,u_exact,phi,L,eps_final,n)
                            #error2=error2+final_error
                        
                        #if index_sorted not in index_tot:
                        #    index_tot.append(index_sorted)
                            X_data=x_local.flatten()
                            dataset.append([X_data, eps_final])
                    
pickle.dump(dataset, open("sort_2d_11_11.5.pkl",'wb'))

print(dataset)
                    

                    



