import numpy as np

def initial_guess(x_axis):
    distances = get_distance_matrix(x_axis)
    d = 0
    N = distances.shape[0]
    for k in range(N):
        d+= np.partition(distances[:,k], 2)[1] # closest neighbour 
    eps = 1./(0.815*d*N)
    return eps

def sample_from_hypercube(dim,n=10,low=0.0,high=0.01):
    x_coordinates = np.zeros((n,dim))
    #np.random.seed(42)
    for i in range(n):
        x_coordinates[i,:] = np.random.uniform(low=low, high=high, size=dim)
    
    return x_coordinates

def get_int_matrix(x_axis,epsilon):
    B = np.zeros((x_axis.shape[0]+1,x_axis.shape[0]+1)) # including constant function
    for i in range(x_axis.shape[0]):
        for j in range(x_axis.shape[0]):
            B[i,j] = (1.0 + (epsilon*np.linalg.norm(x_axis[i]-x_axis[j]))**2)**(-0.5)
            
    B[-1,:]=1.0
    B[:,-1]=1.0
    B[-1,-1]=0.0
    
    return B

def get_distance_matrix(x_axis):
    B = np.zeros((x_axis.shape[0],x_axis.shape[0]))
    
    for i in range(x_axis.shape[0]):
        for j in range(x_axis.shape[0]):
            B[i,j] = np.linalg.norm((x_axis[i]-x_axis[j])) #there was a sqrt here that didn't make sense!!!
            
    return B

def get_deriv_matrix(x_axis,eps):
    B = np.zeros((x_axis.shape[0]+1,x_axis.shape[0]+1))
    
    for i in range(x_axis.shape[0]):
        for j in range(x_axis.shape[0]): 
            B[i,j] = -eps*(np.linalg.norm(x_axis[i]-x_axis[j]))**2/(1+(eps*np.linalg.norm(x_axis[i]-x_axis[j]))**2)**(3./2.) 
                
    B[-1,:]=0.0
    B[:,-1]=0.0
    B[-1,-1]=0.0
    
    return B

class Interpolator:
    # computes the interpolating function for f:R^n->R
    def __init__(self,w,x_axis,epsilon):
        self.w = w
        self.x_axis = x_axis
        self.epsilon = epsilon

    def evaluate(self,x_eval): 
        fct_eval = np.zeros(x_eval.shape[0])
    
        for i in range(fct_eval.shape[0]):
            for j in range(self.x_axis.shape[0]):
                fct_eval[i] += self.w[j]*(1.0 + (self.epsilon*np.linalg.norm(x_eval[i]-self.x_axis[j]))**2)**(-0.5)
            fct_eval[i] += self.w[-1]

        return fct_eval
    
    def evaluate_derivative(self,x_eval):
        fct_eval = np.zeros(x_eval.shape[0])
    
        for i in range(fct_eval.shape[0]):
            for j in range(self.x_axis.shape[0]):
                fct_eval[i] += -self.w[j]*(self.epsilon*np.linalg.norm(x_eval[i]-self.x_axis[j])**2.)\
                                *(1.0 + (self.epsilon*np.linalg.norm(x_eval[i]-self.x_axis[j]))**2)**(-1.5)
            fct_eval[i] += 0. # constant state is zero

        return fct_eval
    
    
