import numpy as np
from datetime import datetime
from random import seed, random
import numpy as np
from rbf_tools import *

def gradient(epsilon,x_axis, upper_cond = 11.5, lower_cond = 11): 
    try:
        A =  get_int_matrix(x_axis,epsilon)
        Ainv = np.linalg.inv(A)
        cond = np.linalg.cond(A,'fro')
        logcond = np.log10(cond)

        dXde = get_deriv_matrix(x_axis,epsilon)

        deriv = (np.linalg.norm(Ainv, ord='fro') * np.trace(np.matmul(np.transpose(A), dXde))) / np.linalg.norm(A, ord='fro') +\
                + np.linalg.norm(A,ord='fro') * np.trace(np.matmul(np.transpose(Ainv),-np.matmul(np.matmul(Ainv, dXde), Ainv))) / np.linalg.norm(Ainv, ord='fro')
        
        if logcond < lower_cond:
            grad = -deriv/(cond*np.log(10.))
        elif logcond > upper_cond:
            grad = deriv/(cond*np.log(10.))
        else:
            grad = 0
    except:
        grad = 1. # diminish eps 
        
    return grad

def compute_loss(eps,x_axis,upper_cond = 11.5, lower_cond = 11):
    A = get_int_matrix(x_axis,eps)
    cond = np.linalg.cond(A,'fro')
    logcond = np.log10(cond)
    
    if logcond > upper_cond:
        loss = logcond -upper_cond
    elif logcond < lower_cond:
        loss = lower_cond-logcond
    else:
        loss = 0
    
    return loss


def optimization_loop(x_axis, eps_init, lr_init):
        eps = eps_init
        updates = []
        update = 100.
        current_loss = 100.
        
        j = 0
        while np.abs(current_loss) > 10**(-4):
            lr = lr_init/(1+j)
            update = gradient(eps,x_axis)
            eps = eps - lr*update
            current_loss = compute_loss(eps,x_axis)
            updates.append(update)
            j = j+1
            if j > 2000:
                # it is unsuccessful
                # check last updates
                print('update size:',np.abs(update))
                print('loss size:',np.abs(current_loss))
                variation = np.sum(np.array(updates[:-1])-np.array(updates[1:]))
                #print('variation',np.abs(variation))
                small = 10**(-4)
                large = 0.0001
                new_lr = lr_init
                #if np.abs(variation) < small:
                #    new_lr = 2*lr_init
                #elif np.abs(variation) > large:
                #    new_lr = lr_init/2.
                #print('new_lr',new_lr)
                return eps, current_loss,new_lr,j
                
        return eps, current_loss, lr_init,j

def optimization_loop_ADAM(x_axis, eps_init, lr_init):
        eps = eps_init
        updates = []
        update = 100.
        current_loss = 100.
        j = 0

        # use tabulated values
        alpha = lr_init
        b1 = 0.9
        b2 = 0.999

        t = 0
        m0 = 0
        v0 = 0

        while np.abs(current_loss) > 10**(-4):
            eps, update, m0, v0 = compute_parameter_update_ADAM(eps, x_axis, alpha, b1, b2, m0, v0, t)
            current_loss = compute_loss(eps,x_axis)
            updates.append(update)
            t = t+1
            if t > 2000:
                # it is unsuccessful
                # check last updates
                print('update size:',np.abs(update))
                print('loss size:',np.abs(current_loss))
                variation = np.mean(np.array(updates[-100:-1])-np.array(updates[-101:-2]))
                #print('variation',np.abs(variation))
                #small = 10**(-8)
                #large = 0.0001
                #if np.abs(variation) < small:
                #    new_lr = 2*lr_init
                #elif np.abs(variation) > large:
                #    new_lr = lr_init/2.
                #else:
                    # restart the eps
                eps = eps_init*2. # start from safer point
                #    new_lr = 1.0
                new_lr = lr_init
                #print('new_lr',new_lr)
                #print(eps)
                return eps, current_loss,new_lr,t

        return eps, current_loss, lr_init,t

def compute_parameter_update_ADAM(eps, x_axis, alpha, b1, b2, m0, v0, t):
    # compute gradient
    update = gradient(eps,x_axis)
    m1 = b1*m0 + (1-b1)*update
    v1 = b2*v0 + (1-b2)*update**2
    m_hat = m1/(1-np.power(b1,t+1))
    v_hat = v1/(1-np.power(b2,t+1))
    new_eps = eps - alpha*m_hat/(np.sqrt(v_hat)+10**(-8))

    return new_eps, update, m1, v1
