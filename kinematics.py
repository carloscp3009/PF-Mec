import numpy as np 

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

def G():

    return True