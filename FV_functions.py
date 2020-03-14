import numpy as np

def f_star(huv):
    result=f_flux(huv[:,1:-1,1:-1],huv[:,2:,1:-1])-f_flux(huv[:,:-2,1:-1],huv[:,1:-1,1:-1])
    return result

#Lax-Friedrichs f]ux function
def f_flux(v1,v2):
    lam=lam_max_x(v1,v2)
    return 0.5*(f(v1)+f(v2))-0.5*lam*(v2-v1)

def f(x):
    x1=x[1,:,:]
    x2=np.power(x[1,:,:],2)/x[0,:,:]+np.power(x[0,:,:],2)/2
    x3=x[1,:,:]*x[2,:,:]/x[0,:,:]
    return np.stack((x1,x2,x3))

def g_star(huv):
    result=g_flux(huv[:,1:-1,1:-1],huv[:,1:-1,2:])-g_flux(huv[:,1:-1,:-2],huv[:,1:-1,1:-1])
    return result

#Lax-Friedrichs f]ux function
def g_flux(v1,v2):
    lam=lam_max_y(v1,v2)
    return 0.5*(g(v1)+g(v2))-0.5*lam*(v2-v1)

def g(x):
    x1=x[2,:,:]
    x2=x[1,:,:]*x[2,:,:]/x[0,:,:]
    x3=np.power(x[2,:,:],2)/x[0,:,:]+np.power(x[0,:,:],2)/2
    return np.stack((x1,x2,x3))
    

def lam_max_x(huv_1,huv_2):
    lam_1=np.abs(huv_1[1,:,:]/huv_1[0,:,:])+np.sqrt(huv_1[0,:,:])
    lam_2=np.abs(huv_2[1,:,:]/huv_2[0,:,:])+np.sqrt(huv_2[0,:,:])
    return np.maximum(lam_1,lam_2)

def lam_max_y(huv_1,huv_2):
    lam_1=np.abs(huv_1[2,:,:]/huv_1[0,:,:])+np.sqrt(huv_1[0,:,:])
    lam_2=np.abs(huv_2[2,:,:]/huv_2[0,:,:])+np.sqrt(huv_2[0,:,:])
    return np.maximum(lam_1,lam_2)