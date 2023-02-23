#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.io import loadmat
import pandas as pd
import datetime


# ## Problem 3

# In[4]:


#Part 1. 


# Define the mean vector and covariance matrix
mu = np.array([1, 1])
cov = np.array([[1, 0], [0, 2]])

# Create a grid of points in the x-y plane
x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

# Evaluate the probability density function at each point on the grid
z = multivariate_normal.pdf(pos, mean=mu, cov=cov)

# Create a plot with contours and colorbar legend
fig, ax = plt.subplots()
cp = ax.contour(x, y, z, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(cp, label='Density')
 
plt.show()


# In[5]:


#Part 2


# Define the mean vector and covariance matrix
mu = np.array([-1, 2])
cov = np.array([[2, 1], [1, 4]])

# Create a grid of points in the x-y plane
x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

# Evaluate the probability density function at each point on the grid
z = multivariate_normal.pdf(pos, mean=mu, cov=cov)

# Create a plot with contours and colorbar legend
fig, ax = plt.subplots()
cp = ax.contour(x, y, z, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(cp, label='Density')
plt.title('Isocontours of f(μ,Σ) with μ = [-1, 2] and Σ = [[2, 1], [1, 4]]')
 
plt.show()




# In[6]:


# Part 3

# Define the mean vectors and covariance matrices
mu1 = np.array([0, 2])
mu2 = np.array([2, 0])
cov = np.array([[2, 1], [1, 1]])

# Create a grid of points in the x-y plane
x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

# Evaluate the difference of probability density functions at each point on the grid
z = multivariate_normal.pdf(pos, mean=mu1, cov=cov) - multivariate_normal.pdf(pos, mean=mu2, cov=cov)

# Create a plot with contours and colorbar legend
fig, ax = plt.subplots()
cp = ax.contour(x, y, z, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(cp, label='f(µ1, Σ1) − f(µ2, Σ2)')
plt.title('Isocontours of f(µ1, Σ1) − f(µ2, Σ2)')
 
plt.show()






# In[7]:


# Part 4

# Define the mean vectors and covariance matrices
mu1 = np.array([0, 2])
mu2 = np.array([2, 0])
cov1 = np.array([[2, 1], [1, 4]])
cov2 = np.array([[2, 1], [1, 4]])

# Create a grid of points in the x-y plane
x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

# Evaluate the difference in probability density function at each point on the grid
z = multivariate_normal.pdf(pos, mean=mu1, cov=cov1) - multivariate_normal.pdf(pos, mean=mu2, cov=cov2)

# Create a plot with contours and colorbar legend
fig, ax = plt.subplots()
cp = ax.contour(x, y, z, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(cp, label='Density')

plt.show()



# In[8]:


# Part 5 


# Define the mean vectors and covariance matrices
mu1 = np.array([1, 1])
mu2 = np.array([-1, -1])
cov1 = np.array([[2, 0], [0, 1]])
cov2 = np.array([[2, 1], [1, 2]])

# Create a grid of points in the x-y plane
x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

# Evaluate the probability density function at each point on the grid for both means and covariances
z1 = multivariate_normal.pdf(pos, mean=mu1, cov=cov1)
z2 = multivariate_normal.pdf(pos, mean=mu2, cov=cov2)

# Subtract the two functions
z = z1 - z2

# Create a plot with contours and colorbar legend
fig, ax = plt.subplots()
cp = ax.contour(x, y, z, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar(cp, label='Density')

plt.show()





# ## Problem 4

# In[9]:


# Set random seed for reproducibility
np.random.seed(12345)

# Define the mean and covariance matrix
mean = np.array([3, 3.5])
cov = np.array([[9, 4.5], [4.5, 4]])

# Draw n random samples from the multivariate normal distribution
n = 100
samples = np.random.multivariate_normal(mean, cov, size=n)

# Compute the mean of the samples
sample_mean = np.mean(samples, axis=0)

# Compute the covariance matrix of the samples
sample_cov = np.cov(samples.T)

# Compute the eigenvectors and eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(sample_cov)

# Sort the eigenvectors and eigenvalues in descending order of eigenvalues
sort_indices = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sort_indices]
eigenvalues = eigenvalues[sort_indices]

# Plot the samples and covariance eigenvectors
plt.scatter(samples[:, 0], samples[:, 1])
plt.arrow(sample_mean[0], sample_mean[1], eigenvalues[0] * eigenvectors[0, 0], eigenvalues[0] * eigenvectors[1, 0], width=0.1, color='r')
plt.arrow(sample_mean[0], sample_mean[1], eigenvalues[1] * eigenvectors[0, 1], eigenvalues[1] * eigenvectors[1, 1], width=0.1, color='r')
plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter plot of samples and covariance eigenvectors')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

# Rotate the samples by the eigenvectors and plot the rotated samples
rotated_samples = np.dot(eigenvectors.T, (samples - sample_mean).T).T
plt.scatter(rotated_samples[:, 0], rotated_samples[:, 1])
plt.xlim([-15, 15])
plt.ylim([-15, 15])
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Scatter plot of rotated samples')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()


# ## Problem 8

# In[12]:


mnist = np.load('/mnist-data-hw3.npz')


# In[11]:





# In[ ]:





# In[ ]:





# In[ ]:




