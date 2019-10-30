# Created by: Elias Muñoz
# Created: October 24th 2019
# Description: Implementation of the Global Kinematical System Index to the Karpiel Mechanism
# email: jemontenegro@uninorte.edu.co

"""
    Import Libraries
    Numpy as principal library, due to the matrix computations
"""
import numpy as np
import scipy.stats as sp

"""
    Inverse Kinematics
    this function transform the position of the end-effector in the mechanism to
    the actuators' positions

    Inputs:
        L: represents all the bar's legnth of the mechanism. Base's radius, Revolute A's lenght,
            Revolute D's length and the diference between the Revolute B and C.
            Is a list with 4 float values
        P: represent the position of the end-effector.
            Is a numpy matrix of 3x1 shape or 1x3 shape
    
    Outputs:
        Theta: represents the angular position of the three revolute A
            Is a 1x3 list
        Q: represents the positions of the actuators
            Is a 1x3 list
        Beta: represents the angular position of the three revolute B
            Is a 1x3 list
"""

def InverseKinematics(L,P):
    #R_b = L[0], L_A = L[1], L_D = L[2], e = L[3]
    if(P.shape[1] == 3):
	    P=P.T
    P = P*np.matrix([1, 1, 1])
    alpha = np.matrix([0, 120, 240])*np.pi/180
    OOi = L[0]*np.concatenate((np.cos(alpha),np.sin(alpha),0*alpha))
    OiP = P - OOi
    Theta = np.arctan2(OiP[1,:],OiP[0,:]) - alpha + np.pi*np.matrix([-1,1,1])
    OiAi = (-L[1]+L[2])*np.concatenate((np.cos(alpha+Theta),np.sin(alpha+Theta),0*alpha))
    AiCi = -OOi + P + OiAi
    M = np.sum(np.power(AiCi,2),axis=0)
    Q  = np.sqrt(M - L[3]**2)
    CTx2 = np.concatenate((P[2,:] - Q,-2*L[3]*np.matrix([1,1,1]),P[2,:] + Q)).T
    CTx2 = CTx2.tolist()
    Tx2 = np.matrix([np.roots(CTx2[0]),np.roots(CTx2[1]),np.roots(CTx2[2])]).T
    A = Tx2.copy()
    Beta = 2*np.arctan(A[0,:]) + np.pi/2
    return Theta, Q, Beta

"""
    JacobianQ
    this function create the Jacobian matrix, which is the connectivity matrix of the output - input velocities of the mechanism

    Inputs:
        L: represents all the bar's legnth of the mechanism. Base's radius, Revolute A's lenght,
            Revolute D's length and the diference between the Revolute B and C.
            Is a list with 4 float values
        Theta: represents the angular position of the three revolute A
            Is a 1x3 list
        Q: represents the positions of the actuators
            Is a 1x3 list
        Beta: represents the angular position of the three revolute B
            Is a 1x3 list
    Outputs:
        Jq: represents the matrix of connectivity between the end-effector's velocity and the actuator's velocity
            Is a 3x3 matrix
"""

def JacobianQ(L, Theta, Q, Beta):
    #R_b = L[0], L_A = L[1], L_D = L[2], e = L[3]
    Alpha = np.matrix([0, 120, 240])*np.pi/180
    if(Q.shape[1] == 3):
	    Q=Q.T
    Q = Q*np.matrix([1, 1, 1])
    S = -np.concatenate((np.multiply(np.cos(Alpha + Theta),np.cos(Beta)),np.multiply(np.sin(Alpha + Theta),np.cos(Beta)),np.sin(Beta)))
    Sb = np.concatenate((np.multiply(np.cos(Alpha + Theta),np.sin(Beta)),np.multiply(np.sin(Alpha + Theta),np.sin(Beta)),-np.cos(Beta)))
    Sb= np.divide(Sb,Q)
    Jq =  (S + L[3]*Sb).T
    return Jq

"""
    LocalIndexes
    this function compute the four local indexes implement in Global Performance Index System for Kinematic Optimization of
    Robotic Mechanism.

    Inputs:
        L: represents all the bar's legnth of the mechanism. Base's radius, Revolute A's lenght,
            Revolute D's length and the diference between the Revolute B and C.
            Is a list with 4 float values
        Jq: represents the matrix of connectivity between the end-effector's velocity and the actuator's velocity
            Is a 3x3 matrix
    Outputs:
        I: are the result or the score obtained in each index
            Is a 1x4 list
"""

def LocalIndexes(L, J):
    k = np.linalg.eig(J*J.T)
    k = k[0]
    #print(k)
    lt= np.sum(L)
    Mr = np.power(np.product(k),1/3)/lt**2
    Vm = np.amin(k)/lt
    Vi = np.power(np.product(k),1/3)/np.average(k)
    #Kj = 1/(np.linalg.det(J)*np.linalg.det(np.linalg.pinv(J)))
    Kj = np.sqrt(np.amin(k)/np.amax(k))
    return [Mr, Vm, Vi, Kj]

def WorkspaceDesired(Len = 1, dZ = 0, stepsize = 0.1):
    Pi = np.matrix(np.arange(0.0,Len+stepsize,stepsize))
    m = Pi.shape[1]
    M1 = np.matrix(np.ones_like(Pi)).T
    Pi = M1*Pi
    Pj = Pi.T
    Pi = np.reshape(Pi,(-1,1),order='F').T
    Pj = np.reshape(Pj,(-1,1),order='F').T
    M1 = np.matrix(np.ones((1,m))).T
    Pi = M1*Pi
    Pj = M1*Pj
    Pk = np.reshape(Pi,(-1,1),order='F').T - dZ
    Pi = np.reshape(Pi,(-1,1),order='C').T - Len/2
    Pj = np.reshape(Pj,(-1,1),order='C').T - Len/2
    P  = np.concatenate((Pi,Pj,Pk)).T
    #P = Pi
    return P

def AllIndex(L):
    #P = WorkspaceDesired(500.0,650.0,5.0)
    #P = WorkspaceDesired(500.0,650.0,10.0)
    #P = WorkspaceDesired(500.0,650.0,15.0)
    #P = WorkspaceDesired(500.0,650.0,20.0)
    P = WorkspaceDesired(500.0,650.0,50.0)
    P = P.copy()
    I = []
    for ii in range(P.shape[0]):
        #print(f'ii = {ii}',end='\n')
        Theta, Q, Beta = InverseKinematics(L,P[ii])
        J = JacobianQ(L, Theta, Q, Beta)
        I.append(LocalIndexes(L, J))
    I = np.matrix(I)
    return I

def IntegratedIndex(I):
    I_ave = np.average(I,axis = 0)
    I_std = np.std(I,axis=0)
    I_vol = I_std/I_ave
    I_skw = sp.skew(I,axis = 0)
    I_krt = sp.kurtosis(I,axis = 0)
    I_max = np.amax(I,axis = 0)
    M1 = np.concatenate((np.ones((1,4)),-1*I_ave,-0.1*I_ave,-0.1*I_ave))
    I_int = np.concatenate((I_ave,I_vol,[I_skw],[I_krt]))
    I_int = np.multiply(M1,I_int)
    I_int = np.sum(I_int,axis = 0)
    I_int = np.concatenate((I_int,I_max))
    return I_int

def GlobalIndex(I):
    W = np.matrix([9, 5, 7, 3])
    W = W/np.linalg.norm(W)
    B  = np.divide(W,I[1]).T
    GI = I[0]*B
    GI.tolist()
    return GI

"""
    Study Case
"""

if __name__ == "__main__":
    pass
    import time
    seconds = time.time()
    R_b = 683.01
    L_A = 70.0
    L_D = 200.0
    e   = 69.4-12.4
    L = [R_b, L_A, L_D, e]
    I = AllIndex(L)
    #print('All Index \n',I)
    n = I.shape[0]
    I = IntegratedIndex(I)
    I = GlobalIndex(I)
    seconds = time.time() - seconds
    print(f'Ejemplo fue ejecutado en {seconds} s, probando en {n} puntos y dio como resultado {I[0,0]}')
    #print()

