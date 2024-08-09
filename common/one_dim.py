import numpy as np 
from matplotlib import pyplot as plt
from rbf_tools import *

def compute_error(x_axis,a,b,eps,u_exact,vis=True):
    # define function to approximate
    # to_approximate = lambda x: (x-0.2)**2. + x**3. - 0.5 + np.cos(16*np.pi*x)
    to_approximate = u_exact
    A =  get_int_matrix(x_axis,eps)
    f = np.array([to_approximate(x) for x in x_axis])
    f=np.append(f,[0.0])
    w = np.linalg.solve(A,f)

    #approximator = get_interpolator(w,x_axis,eps,x_axis)

    approximator = Interpolator(w,x_axis,eps)

    # checking approximation outside of interpolation points
    x_axis_oversampled = np.linspace(a,b,100)
    approximator_oversampled = approximator.evaluate(x_axis_oversampled)
    inter_values = approximator.evaluate(x_axis)
    true_vals = [to_approximate(x) for x in x_axis_oversampled]
    error = np.mean(true_vals-approximator_oversampled)**2
    
    return error, [x_axis, inter_values, x_axis_oversampled, true_vals, approximator_oversampled]

def plot_curves(curves_initial, curves_final, name='test'):
    
    x_axis, inter_values_initial, x_axis_oversampled, true_vals, approximator_initial = curves_initial
    _, _, _, _, approximator_final = curves_final
    
    fig,axs=plt.subplots(ncols=1,nrows=1,figsize=(12,5))
    axs.plot(x_axis_oversampled,true_vals, label = 'True function', color='gray')
    axs.plot(x_axis_oversampled,approximator_initial, label = 'Initial approximation')
    axs.plot(x_axis_oversampled,approximator_final, label = 'Final approximation')
    axs.plot(x_axis,inter_values_initial, linestyle='None', marker='.',label='Interpolation Points',color='green')
    axs.legend()
    
    plt.savefig("epsilon_new/"+name+'.png')
    plt.close()
    
    return None

 
