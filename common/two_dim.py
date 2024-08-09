import numpy as np
from matplotlib import pyplot as plt
from rbf_tools import *

# compute errors
def get_structured_points_modified(a,b,c,d,n):
    x = np.linspace(a, b, n)
    y = np.linspace(c, d, n)
    xv, yv = np.meshgrid(x, y)

    x_axis = np.zeros((xv.shape[0]*xv.shape[1],2))
    x_axis[:,0] = xv.flatten()
    x_axis[:,1] = yv.flatten()

    return x_axis

def compute_error_2d(x_local,u_exact,phi,L,eps,n):
    f=np.zeros((n+1,1))
    for i in range(n):
        f[i,0]=u_exact(x_local[i][0],x_local[i][1])
    f[-1,0]=0.0 

    w=np.linalg.solve(L, f) 
    oversampled=get_structured_points_modified(np.min(x_local, axis=0)[0],np.max(x_local, axis=0)[0],np.min(x_local, axis=0)[1],np.max(x_local, axis=0)[1],n)
    number_oversampled=oversampled.shape[0]
    true_values =np.zeros((number_oversampled,1))


    for i in range(number_oversampled):
        true_values[i,0]=u_exact(oversampled[i][0],oversampled[i][1])

    s=np.zeros((number_oversampled,n+1))
    for i in range(n):      
        for j in range(number_oversampled):            
            s[j,i] = phi(eps,oversampled[j] , x_local[i])
    s[:,-1]=1.0        
   
    approximator=np.matmul(s,w)
    for i in range(number_oversampled):
        error=abs(approximator[i,0]-true_values[i,0])**2
    
    return error

def compute_evaluation_error(x_local,evall,u_exact,phi,L,eps,n):
    f=np.zeros((n+1,1))
    for i in range(n):
        f[i,0]=u_exact(x_local[i][0],x_local[i][1])
    f[-1,0]=0.0 

    w=np.linalg.solve(L, f) 
    
    true_value=u_exact(evall[0],evall[1])

    s=np.zeros((1,n+1))
    for i in range(n):      
        s[0,i] = phi(eps,evall , x_local[i])
    s[0,-1]=1.0        
   
    approximator=np.matmul(s,w)
    error=abs(approximator[0,0]-true_value)**2
    
    return error
def plot_figures_2D(true_surface, approximated_surface, x_axis, inter_points, x_axis_orig, name='test'):
    
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    X = x_axis[:,0].reshape(20,20)
    Y = x_axis[:,0].reshape(20,20)
    approximated_surface_reshaped = approximated_surface.reshape(X.shape)
    true_surface_reshaped = true_surface.reshape(X.shape)

    ax.plot_surface(X, Y, approximated_surface_reshaped, label = 'Approximated surface',alpha=0.4)
    ax = fig.add_subplot(121, projection='3d')
    ax.plot_surface(X, Y, true_surface_reshaped, label = 'True surface')
    #ax.legend()
    
    plt.savefig(name+'.png')
    plt.close()
    
    return None
