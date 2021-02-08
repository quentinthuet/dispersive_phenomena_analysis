#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:21:34 2021

@author: n7student
"""

import sys
import numpy as np
import numpy.linalg as npl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#  Discrétisation en espace


xmin = 0.0; xmax = 1.0; nptx = 5; nx = nptx-2  
a = xmax-xmin
hx = a/(nptx -1)
xx = np.linspace(xmin,xmax,nptx) 
xx = xx.transpose()
xxint = xx[1:nx+1]
ymin = 0.0; ymax = 1.0; npty = 5; ny = npty-2 
b = ymax-ymin
hy = b/(npty -1)
yy = np.linspace(ymin,ymax,npty)
yy=yy.transpose() 
yyint = yy[1:ny+1]

#  Matrix system
# On Ox
Kx = np.diag(np.concatenate((np.concatenate((np.array([1]),np.repeat(2/hx**2,nx))),np.array([1])))) +\
    np.diag(np.concatenate(((np.repeat(-1/hx**2,nx)),np.array([0]))),k=-1) +\
    np.diag(np.concatenate(((np.array([0]),np.repeat(-1/hx**2,nx)))),k=1) # Local matrix of size Nx+2 relative to Ox discretization

K2Dx = np.eye(nptx*npty) # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2
K2Dx[nptx:-nptx,nptx:-nptx] = np.kron(np.eye(ny),Kx) 

# # On Oy
Ky = np.diag(np.concatenate((np.concatenate((np.array([0]),np.repeat(2/hy**2,ny))),np.array([0])))) +\
    np.diag(np.concatenate(((np.repeat(-1/hy**2,ny)),np.array([0]))),k=-1) +\
    np.diag(np.concatenate(((np.array([0]),np.repeat(-1/hy**2,ny)))),k=1) # Local matrix of size Nx+2 relative to Oy discretization


matrix_temp= np.eye(nptx)
matrix_temp[0]=0
matrix_temp[-1]=0
K2Dy = np.kron(Ky,matrix_temp) # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2
#K2Dy[::nptx,::nptx]=0
#K2Dy[nptx-1::nptx,nptx-1::nptx]=0
# #
# #
K2 = K2Dx + K2Dy # Final matrix of Laplacien operator with Dirichlet Boundary conditions

for ind in range(2,npty):
    K2[ind*nptx-1,ind*nptx-1] = -3/2/hx
    K2[ind*nptx-1,ind*nptx-2] = 2/hx
    K2[ind*nptx-1,ind*nptx-3] = -1/2/hx
##  Solution and source terms
u = np.zeros((nx+2)*(ny+2)) #Numerical solution
u_ex = np.zeros((nx+2)*(ny+2)) #Exact solution
F = np.zeros((nx+2)*(ny+2)) #Source term

def creer_probleme(nptx,npty,nx,ny,hx,hy,a,b,n=1,k=1):
    # Source term
    def Source_int(x):
        return 2*np.pi**2*(np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
    def Source_bnd(x):
        return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
    def Sol_sin(x):
        return (np.cos(np.pi*x[0])-1)*np.sin(np.pi*x[1])    
    
    for i in range(nptx):
        for j in range(npty):
            coord = np.array([i*hx,j*hy])
            u_ex[j*(nx+2) + i] = Sol_sin(coord)
        if i==0 or i==nptx-1: # Boundary x=0 ou x=xmax
            for j in range(npty):
                coord = np.array([i*hx,j*hy])
                F[j*(nx+2) + i]=Source_bnd(coord)
        else:
            for j in range(npty):
                coord = np.array([i*hx,j*hy])
                if j==0 or j==npty-1: # Boundary y=0 ou y=ymax
                    F[j*(nx+2) + i]=Source_bnd(coord)
                else:
                    F[j*(nx+2) + i]=Source_int(coord)
                    
    return u_ex, F

def creer_probleme_neumann(nptx,npty,nx,ny,hx,hy,a,b,n=1,k=1):
    # Source term
    def Source_int(x):
        return np.pi**2*np.sin(np.pi*x[1]/b)*(np.cos(np.pi*x[0])/b**2+np.cos(np.pi*x[0])/a**2-1)
    def Source_bnd(x):
        return 0
    def Sol_sin(x):
        return (np.cos(np.pi*x[0]/a)-1)*np.sin(np.pi*x[1]/b)    
    
    for i in range(nptx):
        for j in range(npty):
            coord = np.array([i*hx,j*hy])
            u_ex[j*(nx+2) + i] = Sol_sin(coord)
        if i==0 or i==nptx-1: # Boundary x=0 ou x=xmax
            for j in range(npty):
                coord = np.array([i*hx,j*hy])
                F[j*(nx+2) + i]=Source_bnd(coord)
        else:
            for j in range(npty):
                coord = np.array([i*hx,j*hy])
                if j==0 or j==npty-1: # Boundary y=0 ou y=ymax
                    F[j*(nx+2) + i]=Source_bnd(coord)
                else:
                    F[j*(nx+2) + i]=Source_int(coord)
                    
    return u_ex, F

def creer_probleme_neumann_v2(nptx,npty,nx,ny,hx,hy,a,b,vx,n=1,k=1):
    # Source term
    def Source_int(x):
        return np.pi**2*np.sin(np.pi*x[1]/b)*(np.cos(np.pi*x[0])/b**2+np.cos(np.pi*x[0])/a**2-1)+vx*(-np.pi/a)*np.sin(np.pi*x[1]/b)*np.sin(np.pi*x[0]/a)
    def Source_bnd(x):
        return 0
    def Sol_sin(x):
        return (np.cos(np.pi*x[0]/a)-1)*np.sin(np.pi*x[1]/b)    
    
    for i in range(nptx):
        for j in range(npty):
            coord = np.array([i*hx,j*hy])
            u_ex[j*(nx+2) + i] = Sol_sin(coord)
        if i==0 or i==nptx-1: # Boundary x=0 ou x=xmax
            for j in range(npty):
                coord = np.array([i*hx,j*hy])
                F[j*(nx+2) + i]=Source_bnd(coord)
        else:
            for j in range(npty):
                coord = np.array([i*hx,j*hy])
                if j==0 or j==npty-1: # Boundary y=0 ou y=ymax
                    F[j*(nx+2) + i]=Source_bnd(coord)
                else:
                    F[j*(nx+2) + i]=Source_int(coord)
                    
    return u_ex, F

vx = 0

# # On Oy
VKx = np.diag(np.concatenate(((np.repeat(1/hx,nx)),np.array([0]))),k=-1) +\
    np.diag(np.concatenate(((np.array([0]),np.repeat(1/hx,nx)))),k=1) # Local matrix of size Nx+2 relative to Ox discretization

V2Dx = np.zeros((nptx*npty,nptx*npty)) # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2
V2Dx[nptx:-nptx,nptx:-nptx] = vx*np.kron(np.eye(ny),VKx) 

u_ex, F = creer_probleme_neumann_v2(nptx,npty,nx,ny,hx,hy,a,b,vx)

A2 = K2+V2Dx

u = npl.solve(A2,F)

#Post-traintement u_ex+Visualization of the exct solution
uu_ex = np.reshape(u_ex,(nx+2 ,ny+2),order = 'F');
uu = np.reshape(u,(nx+2 ,ny+2),order = 'F');
X,Y = np.meshgrid(xx,yy)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,uu_ex.T,rstride = 1, cstride = 1);
ax.plot_surface(X,Y,uu.T,rstride = 1, cstride = 1);
plt.show()


print('Norme de l\'erreur : ',np.linalg.norm(uu_ex-uu))





