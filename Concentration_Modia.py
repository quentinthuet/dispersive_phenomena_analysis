#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:34:48 2021

@author: cantin
"""

import numpy as np
import numpy.linalg as npl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#  Discrétisation en espace


xmin = 0.0; xmax = 2; nptx = 61; nx = nptx-2  
hx = (xmax-xmin)/(nptx -1)
xx = np.linspace(xmin,xmax,nptx) 
xx = xx.transpose()
xxint = xx[1:nx+1]
ymin = 0.0; ymax = 1.0; npty = 31; ny = npty-2 
hy = (ymax-ymin)/(npty -1)
yy = np.linspace(ymin,ymax,npty)
yy=yy.transpose() 
yyint = yy[1:ny+1]


# =============================================================================
### Parameters
mu = 0.01 # Diffusion parameter
vx = 1 # Vitesse along x
# =============================================================================

cfl =0.5 # cfl =mu*dt/hx^2+mu*dt/hy^2 ou v*dt/h
dt = (hx**2)*(hy**2)*cfl/(mu*(hx**2 + hy**2)) # dt = pas de temps
#dt = cfl*hx/vx
Tfinal = 5   # Temps final souhaitÃ©



###### Matrice de Diffusion Dir/Neumann
#  Matrix system
# On Ox
Kx = np.diag(np.concatenate((np.concatenate((np.array([1]),mu*np.repeat(2/hx**2,nx))),np.array([1])))) +\
    np.diag(np.concatenate(((mu*np.repeat(-1/hx**2,nx)),np.array([0]))),k=-1) +\
    np.diag(np.concatenate(((np.array([0]),mu*np.repeat(-1/hx**2,nx)))),k=1) # Local matrix of size Nx+2 relative to Ox discretization

K2Dx = np.eye(nptx*npty) # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2
K2Dx[nptx:-nptx,nptx:-nptx] = np.kron(np.eye(ny),Kx) 

# # On Oy
Ky = np.diag(np.concatenate((np.concatenate((np.array([0]),np.repeat(2/hy**2,ny))),np.array([0])))) +\
    np.diag(np.concatenate(((np.repeat(-1/hy**2,ny)),np.array([0]))),k=-1) +\
    np.diag(np.concatenate(((np.array([0]),np.repeat(-1/hy**2,ny)))),k=1) # Local matrix of size Nx+2 relative to Oy discretization


matrix_temp= np.eye(nptx)
matrix_temp[0]=0
matrix_temp[-1]=0
K2Dy = np.kron(mu*Ky,matrix_temp) # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2
#K2Dy[::nptx,::nptx]=0
#K2Dy[nptx-1::nptx,nptx-1::nptx]=0
# #
# #
K2 = (K2Dx + K2Dy) # Final matrix of Laplacien operator with Dirichlet Boundary conditions
for ind in range(2,npty):
    K2[ind*nptx-1,ind*nptx-1] = 3/2/hx
    K2[ind*nptx-1,ind*nptx-2] = -1/2/hx
    K2[ind*nptx-1,ind*nptx-3] = 2/hx


#### Matrice de Convection  (Centré)
# # On Oy
VKx = np.diag(np.concatenate(((-np.repeat(1/2/hx,nx)),np.array([0]))),k=-1) +\
    np.diag(np.concatenate(((np.array([0]),np.repeat(1/2/hx,nx)))),k=1) # Local matrix of size Nx+2 relative to Ox discretization

V2Dx = np.zeros((nptx*npty,nptx*npty)) # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2
V2Dx[nptx:-nptx,nptx:-nptx] = vx*np.kron(np.eye(ny),VKx) 


#### Global matrix : diffusion + convection
A2D = -(K2 + V2Dx) #-mu*Delta + V.grad
#
#
##  Cas explicite
u = np.zeros((nx+2)*(ny+2))
u_ex = np.zeros((nx+2)*(ny+2))
err = np.zeros((nx+2)*(ny+2))
F = np.zeros((nx+2)*(ny+2))
#
#
# =============================================================================
# Time stepping
# =============================================================================
s0 = 0.1
x0 = 0.25
y0=0.5

def Sol_init(x):
    return np.exp( -((x[0]-x0)/s0)**2 -((x[1]-y0)/s0)**2   )

def Sol_init_2(x):
    return (s0**2-(x[0]-x0)**2-(x[1]-y0))*(s0**2-(x[0]-x0)**2-(x[1]-y0) > 0)



u_init = np.zeros((nx+2)*(ny+2))
for i in range(nptx):
     for j in range(npty):
             coord = np.array([xmin+i*hx,ymin+j*hy])
             u_init[j*(nx+2) + i] = Sol_init(coord)
             
             
uu_init = np.reshape(u_init,(nx+2 ,ny+2),order = 'F');
fig = plt.figure(figsize=(10, 7))
X,Y = np.meshgrid(xx,yy)
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, uu_init.T, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
ax.view_init(60, 35)
ax.set_zlim(0,1)
             
## Initialize u by the initial data u0
u = u_init.copy()

# Nombre de pas de temps effectues
nt = int(Tfinal/dt)
Tfinal = nt*dt # on corrige le temps final (si Tfinal/dt n'est pas entier)

# Time loop
for n in range(1,nt+1):
    
    # Schéma explicite en temps
    #u = u + dt*A2D.dot(u)
    
    
    u = np.linalg.solve((np.eye(A2D.shape[0])-dt/2*A2D),(np.eye(A2D.shape[0])+dt/2*A2D).dot(u)) 
  
 # Print solution
    if n%5 == 0:
      plt.figure(1)
      plt.clf()
      fig = plt.figure(figsize=(10, 7))
      ax = plt.axes(projection='3d')
      uu = np.reshape(u,(nx+2 ,ny+2),order = 'F');
      surf = ax.plot_surface(X, Y, uu.T, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
      ax.view_init(60, 35)
      plt.title(['Schema explicite avec CFL=%s' %(cfl), '$t=$%s' %(n*dt)])
      ax.set_zlim(-4,1)

####################################################################
# comparaison solution exacte avec solution numerique au temps final
j0 = int((npty-1)/2)
i0 = int((nptx-1)/2)


plt.figure(2)
plt.clf()
x = np.linspace(xmin,xmax,nptx)
y = np.linspace(ymin,ymax,npty)
plt.plot(y,uu_init[-1,:],y,uu[-1,:],'k') #,x,uexacte,'or')
plt.legend(['Solution initiale','Schema explicite =%s' %(cfl)]) #,'solution exacte'],loc='best')
plt.show()

