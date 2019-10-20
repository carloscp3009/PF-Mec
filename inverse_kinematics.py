import numpy as np

R_b = 683.01
L_A = 70
L_D = 200
e   = 69.4-12.4

L = [R_b, L_A, L_D, e]

P = np.matrix([0, 0, -500])

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
    CTx2 = np.concatenate((P[2,:] - Q,-2*e*np.matrix([1,1,1]),P[2,:] + Q)).T
    CTx2 = CTx2.tolist()
    Tx2 = np.matrix([np.roots(CTx2[0]),np.roots(CTx2[1]),np.roots(CTx2[2])]).T
    A = Tx2.copy()
    Beta = 2*np.arctan(A[0,:]) + np.pi/2
    return Theta, Q, Beta

Theta, Q, Beta = InverseKinematics(L,P)
print('Theta',Theta*180/np.pi,sep = ':\t')
print('Q',Q,sep = ':\t')
print('Beta',Beta*180/np.pi,sep = ':\t')