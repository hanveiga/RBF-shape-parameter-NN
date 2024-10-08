import numpy as np
from rbf_tools import *

def rippa_cv(ep,x_axis,rhs,mode='full'):
    # Takes an eps and rhs to compute the error vector
    # as per: http://www.math.iit.edu/~fass/Dolomites.pdf
    error_vector = np.zeros(x_axis.shape)
    n,_ = x_axis.shape
    rhs2 = np.append(rhs,[0.0])
    M = get_int_matrix(x_axis,ep)
    Mcond = np.linalg.cond(M,'fro')
    
    if Mcond > 10**(16):
        return np.inf #if the matrix is too ill-conditioned, disregard the eps -- for stability of results
    try:
        w = np.linalg.solve(M,rhs2)
    except:
        return np.inf
    invM = np.linalg.inv(M)
    error_vector = np.divide(w,np.diagonal(invM))
    return np.linalg.norm(error_vector)

def loo_cv(ep,x_axis,rhs,mode='full'):
    # Takes an eps and rhs to compute the error vector
    # as per: http://www.math.iit.edu/~fass/Dolomites.pdf
    error_vector = np.zeros(x_axis.shape)
    n,_ = x_axis.shape
    
    for i in range(n):
        x_partial = np.delete(x_axis,i)
        x_partial = x_partial.reshape((n-1,1))
        rhs2 = np.delete(rhs,i)
        
        M = get_int_matrix(x_partial,ep)
        rhs2 = np.append(rhs2,[0.0])
        w = np.linalg.solve(M,rhs2)
        
        # make interpolator
        s = Interpolator(w,x_partial,ep)
        error_vector[i] = (rhs[i] - s.evaluate(x_axis[i]))
        
    return np.linalg.norm(error_vector)

def loo_cv_grad_descent(ep_init, x_axis, rhs, decay_rate = 0.9, lr=1.0):
    # Takes an eps and rhs to compute the error vector
    # as per: http://www.math.iit.edu/~fass/Dolomites.pdf
    error_vector = np.zeros(x_axis.shape)
    derivative_vector = np.zeros(x_axis.shape)
    
    n,_ = x_axis.shape
    
    ep_k_p_1 = 0
    ep_k = ep_init
    
    b1 = 0.9
    b2 = 0.999
    m0 = 0
    v0 = 0
    t = 0
    
    for k in range(2000):
        for i in range(n):
            x_partial = np.delete(x_axis,i)
            x_partial = x_partial.reshape((n-1,1))
            rhs2 = np.delete(rhs,i)

            M = get_int_matrix(x_partial,ep_k)
            rhs2 = np.append(rhs2,[0.0])
            w = np.linalg.solve(M,rhs2)

            # make interpolator
            s = Interpolator(w,x_partial,ep_k)
            error_vector[i] = (rhs[i] - s.evaluate(x_axis[i]))
            derivative_vector[i] = s.evaluate_derivative(x_axis[i])

        gradient = np.sum(np.multiply(error_vector,derivative_vector))
        ep_k_p_1, m0, v0 = compute_parameter_update_ADAM_2(ep_k, gradient, lr, b1, b2, m0, v0, t)
        
        if np.abs(ep_k-ep_k_p_1) < 10**(-8):
            break
        lr = lr*decay_rate
            
        ep_k = ep_k_p_1
    
    return np.abs(ep_k_p_1)


def loo_cv_grad_descent_surrogate_error(ep_init, x_axis, rhs, decay_rate = 0.99, lr=1.0):
    # Takes an eps and rhs to compute the error vector
    # as per: http://www.math.iit.edu/~fass/Dolomites.pdf
    error_vector = np.zeros(x_axis.shape)
    derivative_vector = np.zeros(x_axis.shape)
    
    n,_ = x_axis.shape
    
    ep_k_p_1 = 0
    ep_k = ep_init
    
    
    b1 = 0.9
    b2 = 0.999
    m0 = 0
    v0 = 0
    t = 0
    
    
    for k in range(10000):
        M = get_int_matrix(x_axis,ep_k)
        rhs2 = np.append(rhs,[0.0])
        w = np.linalg.solve(M,rhs2)

        # make interpolator
        invM = np.linalg.inv(M)

        error_vector = np.divide(w,np.diagonal(invM))
        s = Interpolator(w,x_axis,ep_k)
        
        for i in range(n): 
            derivative_vector[i] = s.evaluate_derivative(x_axis[i])

        gradient = np.sum(np.multiply(error_vector,derivative_vector))
        #ep_k_p_1 = ep_k - lr*np.sum(gradient)
        
        ep_k_p_1, m0, v0 = compute_parameter_update_ADAM_2(ep_k, gradient, lr, b1, b2, m0, v0, t)
        
        if np.abs(ep_k-ep_k_p_1) < 10**(-8):
            break
        lr = lr*decay_rate
            
        ep_k = ep_k_p_1
    
    return np.abs(ep_k_p_1)

def compute_parameter_update_ADAM_2(eps_k, gradient, lr, b1, b2, m0, v0, t):
    # compute gradient
    
    m1 = b1*m0 + (1-b1)*gradient
    v1 = b2*v0 + (1-b2)*gradient**2
    m_hat = m1/(1-np.power(b1,t+1))
    v_hat = v1/(1-np.power(b2,t+1))
    new_eps = eps_k - lr*m_hat/(np.sqrt(v_hat)+10**(-8))
    
    return new_eps, m1, v1
