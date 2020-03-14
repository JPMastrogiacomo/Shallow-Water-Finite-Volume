from FV_functions import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


m=61 #Grid dimensions
xy=np.linspace(-1,1,m)
dxy=xy[2]-xy[1] #Grid spacing

xx,yy=np.meshgrid(xy,xy)


#Height, x-momentum, y-momentum
huv=np.zeros((3,m+2,m+2))


#Initial conditions. Square displacement.
def ic(x,y):
    if -0.5<x and x<0.5:
        if -0.5<y and y<0.5:
            return 2
        else:
            return 1
    else:
        return 1

#Fill out initial ghost cells
for i in range(m):
    huv[0,0,i+1]=ic(xx[0,i],yy[0,i]) #Top
    huv[0,-1,i+1]=ic(xx[-1,i],yy[-1,i]) #Bottom
    huv[0,i+1,0]=ic(xx[i,0],yy[i,0]) #Left
    huv[0,i+1,-1]=ic(xx[i,-1],yy[i,-1]) #Right
    
#Fill out initial conditions throughout the domain
for i in range(m):
    for j in range(m):
        huv[0,i+1,j+1]=ic(xx[i,j],yy[i,j])


t=0
mass=np.zeros(0) #Vector of mass to ensure conservation

#Runs until hit time limit
while(t<2):
    
    #Dynamically chosen time step according to CFL condition
    lam_x=np.min(lam_max_x(huv[:,1:-1,1:-1],huv[:,2:,1:-1]))
    lam_y=np.min(lam_max_y(huv[:,1:-1,1:-1],huv[:,1:-1,2:]))
    dt=(0.5/2)*np.minimum(dxy/lam_x,dxy/lam_y)
        
    #Copy ghost cells from adjacent cells
    huv[:,0,:]=huv[:,1,:]
    huv[:,-1,:]=huv[:,-2,:]
    huv[:,:,0]=huv[:,:,1]
    huv[:,:,-1]=huv[:,:,-2]
    
    #Update normal velocities in ghost cells to ensure wave refleccion
    huv[1,0,:]=-huv[1,1,:]
    huv[1,-1,:]=-huv[1,-2,:]
    huv[2,:,0]=-huv[2,:,1]
    huv[2,:,-1]=-huv[2,:,-2]
    
    #Update grid using finite volume scheme
    huv[:,1:-1,1:-1]=huv[:,1:-1,1:-1]-(dt/dxy)*(f_star(huv[:,:,:])+g_star(huv[:,:,:]))
    
    #Plot each time step
    plt.clf()
    plt.pcolormesh(xx,yy,huv[0,1:-1,1:-1],cmap=cm.coolwarm,vmin=1,vmax=2)
    plt.colorbar()
    plt.draw()
    plt.pause(0.01)
    
    #Check current mass
    mass=np.append(mass, [np.sum(huv[0,1:-1,1:-1])])
    
    #Update current time
    t+=dt

#Final plotting
#plt.plot(np.arange(len(mass)),mass)
#plt.xlabel("Iterations")
#plt.ylabel("Integral of height over square")
#plt.title("Demonstrates Conservation of Mass")

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(xx,yy,huv[0,1:-1,1:-1])
#plt.pcolormesh(xx,yy,huv[0,1:-1,1:-1],cmap=cm.coolwarm)
#plt.colorbar()
#plt.title("Height at t=3")
#plt.show()




