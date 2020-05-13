# ODE-Parameter-Estimation-with-Gradient-Descent
Attempt to use gradient descent to learn two or more unknown parameters in a SIR model

Author: John Bowersox
Date 5/10/2020
Project Name: ODE Parameter Estimation using Gradient Descent

The goal project was to experiment with gradient decent as a method to find unknown parameters in a known math model. Since many parametric methods for calculating parameters from datasets for ODE’s resemble gradient decent but are computationally expensive it seemed worthwhile to investigate. After reading a blog by [Demetri Pananos](https://dpananos.github.io/posts/2019/05/blog-post-14/), I wanted to investigate weather upscaling could be achieved, such as finding a second unknown parameter, or if a better gradient decent method could be used to speed up learning. 

The following Libraries were used

Matplotlib

Autograd: To generate a jacobian of an input ODE

Scipy.integrate: Used to run the system over a set timespan

Numpy: Used to store data and for linear algebra operations

Overall performance was poor and directly upscaling the process does not seem possible for most models. If one parameter accidentally crosses a steady state threshold then the system will not converge. As well the system is highly susceptible to initialization, hyperparameter tuning, and accumulating gradient for more advanced gradient descent may cause issues when condensed to update the scalar parameter. 

