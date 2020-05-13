#=============================================================================
#Author         John Bowersox
#Date           5/10/2020
#Project Name   SIR parameter estimation using gradient descent
#=============================================================================

import autograd
from autograd.builtins import tuple
import autograd.numpy as np
#Automatic ODE integrater <This will be used for the approximation of the gradient of the forwardstep
from scipy.integrate import odeint
#MATLAB library graph plotter
import matplotlib.pyplot as plot
import numpy as nump

#=============================================================================
#Block 1 Model Definition
#Function is the derivatives of the SIR model, for calculating a value at time t
#F can be passed to Autograd odeint functions to generate plots or to generate a Jacobian
#Inputs:        Y a np.array of starting conditions
#               t a np.linespace(start, end)
#               beta a float
#               gama a float
#Returns        np.array of floats [3,t]
#               di, ds, dr are the solved system over t
#=============================================================================
def f(y, t, beta, gama):

    S, I, R = y

    ds = -beta*S*I

    di = beta*S*I - gama*I 

    dr = gama*I 

    return np.array([ds, di, dr])

#=============================================================================
#Block 2 Jacobian Matrix Instantiation and packaging for odeint
#Function Name: SYS
#Inputs:        Y a np.array of starting conditions
#               t a np.linespace(start, end)
#               beta a float
#               gama a float
#Returns        np.array of floats
#               dy_dt is the solved system over t
#               grad_y_beta is the gradient of y in respect to beta for later multiplication
#               grad_y_gama is the gradient of y in respect to gama for later multiplication
#=============================================================================
J = autograd.jacobian(f, argnum = 0)
grad_f_beta = autograd.jacobian(f, argnum = 2)
grad_f_gama = autograd.jacobian(f, argnum = 3)

def SYS(Y, t, beta, gama):

    dy_dt = f(Y[0:3], t, beta, gama)

    grad_y_beta = J(Y[:3], t, beta, gama)@Y[-3::] + grad_f_beta(Y[:3], t, beta, gama)
    grad_y_gama = J(Y[:3], t, beta, gama)@Y[-3::] + grad_f_gama(Y[:3], t, beta, gama)

    system = np.concatenate([dy_dt, grad_y_beta]) 
    return np.concatenate([system, grad_y_gama])

#=============================================================================
#Block 3 
#Generating a training data set
#=============================================================================

#These are our known parameters we are attempting to estimate
BETA = 0.00006
GAMA = 0.2

#The inital conditions for the model
start = np.array([50000, 1, 0, 0, 0, 0, 0, 0, 0])
t = np.linspace(0,29)

#The model ran using odeint
data = odeint(SYS, y0 = start, t = t, args = tuple([BETA, GAMA]))

#plotting the results
plot.scatter(t,data[:,0], marker = '.', alpha = 0.5, label = 'S')
plot.scatter(t,data[:,1], marker = '.', alpha = 0.5, label = 'I')
plot.scatter(t,data[:,2], marker = '.', alpha = 0.5, label = 'R')
plot.show()

#=============================================================================
#Block 4 Cost function declaration
#Object Name: Cost
#Function name: error
#Inputs:        Y a np.array of starting conditions
#               t a np.linespace(start, end)
#               beta a float
#               gama a float
#Returns        np.array of floats
#               dy_dt is the solved system over t
#               grad_y_beta is the gradient of y in respect to beta for later multiplication
#               grad_y_gama is the gradient of y in respect to gama for later multiplication
#=============================================================================
def Cost(observed_data):

    def error(predicted_data):

        n = observed_data.shape[0]

        err = np.linalg.norm(observed_data - predicted_data, 2, axis = 1)

        return np.sum(err)/n

    #This is to get a gradient of the loss function for cross multiplication
    return error

#=============================================================================
#Block 5 Gradient Descent
#=============================================================================

#Initialization points
learning_beta = 0.0001
learning_gama = 0.25

#Instantiating Cost object
cost = Cost(data[1:,:3])
#generating the gradient of the cost function
grad_C = autograd.grad(cost)

#hyper parameter settings
epoch = 200
learning_rate = 0.000000000000001

for e in range(0,epoch):

    #Solving the system with the learning parameters
    y = odeint(SYS, y0 = start, t = t, args = tuple([learning_beta, learning_gama]))

    #Pulling out the predicted values from the model for cost function
    perd = y[:,:3]

    #finding the gradient of the loss
    gradient_of_Loss_wrt_solution = grad_C(perd[1:,:])

    #pulling out the parameter gradients wrt data
    beta_gradient_data = y[1:, 3:6]
    gama_gradient_data = y[1:, 6:9]

    #updating the parameters, cross multiplying the gradients to condence into a scalar value
    #and to relate to the change in loss wrt the unkown parameter
    learning_beta += learning_rate*(gradient_of_Loss_wrt_solution*beta_gradient_data).sum()
    #Comment out Gama to see only beta tune, behavior generaly improves
    learning_gama += learning_rate*(gradient_of_Loss_wrt_solution*gama_gradient_data).sum()

    print(learning_beta,learning_gama)

#=============================================================================
#Block 6
#Plotting data from final itteration of gradient descent for visual comparison
#=============================================================================
print(learning_beta,learning_gama)
plot.scatter(t,perd[:,0], marker = '.', alpha = 0.5, label = 'S')
plot.scatter(t,perd[:,1], marker = '.', alpha = 0.5, label = 'I')
plot.scatter(t,perd[:,2], marker = '.', alpha = 0.5, label = 'R')
plot.show()


