# -*- coding: utf-8 -*-
"""

#the following code modules can be used to:
(1) generate a HD categorical dataset based on the true PPTD model parameters
(2) estimate the PTD model parameters via EM algorithm
(3) estimate the PPTD model parameters via EM algorithm
(4) estimate the PTD-HT model parameters by hard-threholding the PTD model parameters
(5) verify the effectiveness of AIC in the selection of h
(6) compute test likihood when h deviates from the true value
(7) compute MSE of PTD, PTD-HT, PPTD in model parameter estimation
Note: Module 7 can be repeatedly used to compute MSEs under different p's and n's

"""


import numpy as np
from scipy.stats import multinomial
import copy


#setup
P = 50
d = 5
N = 300


######
#Module 1: generate a HD categorical dataset based on the true PPTD model parameters
#true proportion vector of PPTD
G = np.array([0.4, 0.3, 0.2, 0.1])
#import true probability vectors of PPTD that are stored in U_true.npy
U = np.load('U_true.npy')

def multi_to_one(u):
    return np.argmax(multinomial.rvs(n=1, p=u, size=1))+1

def x_generator(G, U, N, P):
    Z = np.argmax(multinomial.rvs(n=1, p=G, size=N), axis=1)
    X = np.zeros((N,P))
    for i in np.arange(N):
        for j in np.arange(P):
            X[i,j] = multi_to_one(U[j,Z[i],:])
    return X.astype(int)
            
np.random.seed(0)
X = x_generator(G, U, N, P) #X is a 300*50 categorical dataset


######
#Module 2: PTD model parameter estimation
######
def cp_e_step(X, G, U, N, P, h):
    Z = np.zeros((N,h))
    for i in np.arange(N):
        for k in np.arange(h):
            prodt = G[k]
            for j in np.arange(P):
                idx_level = X[i,j]-1
                prodt = prodt * U[j,k,idx_level]
            Z[i,k] = prodt
    return Z / np.sum(Z, axis=1, keepdims=True)
    
def cp_m_step_g(Z, N):
    G = np.sum(Z, axis=0) / N
    return G

def cp_m_step_u(X, Z, N, P, d, h):
    U = np.zeros((P, h, d))
    for j in np.arange(P):
        for k in np.arange(h):
            for c in np.arange(d):
                U[j,k,c] = np.sum(Z[:,k] * (X[:,j]==(c+1)))
    return U / np.sum(U, axis=2, keepdims=True)

def log_likelihood(X, G, U, N, P, h):
    l = 0
    for i in np.arange(N):
        l_ind = 0
        for k in np.arange(h):
            prodt = G[k]
            for j in np.arange(P):
                idx_level = X[i,j]-1
                prodt = prodt * U[j,k,idx_level]
            l_ind = l_ind + prodt
        l = l + np.log(l_ind)
    return l

def cp(X, G_ini, U_ini, N, P, d, h):
    G_old = G_ini
    U_old = U_ini
    LK = []
    LK.append(log_likelihood(X, G_old, U_old, N, P, h))
    
    ite = 0
    err = 10
    while err>1e-6 and ite<500:
        #e step
        Z = cp_e_step(X, G_old, U_old, N, P, h)
        #m step
        G_new = cp_m_step_g(Z, N)
        U_new = cp_m_step_u(X, Z, N, P, d, h)

        err = np.mean(np.abs(U_new - U_old))
        ite = ite + 1

        G_old = G_new
        U_old = U_new

        LK.append(log_likelihood(X, G_old, U_old, N, P, h))
        
    return G_old, U_old, LK

def cp_simple(X, G_ini, U_ini, N, P, d, h):
    G_old = G_ini
    U_old = U_ini
    
    ite = 0
    err = 10
    while err>1e-6 and ite<500:
        #e step
        Z = cp_e_step(X, G_old, U_old, N, P, h)
        #m step
        G_new = cp_m_step_g(Z, N)
        U_new = cp_m_step_u(X, Z, N, P, d, h)

        err = np.mean(np.abs(U_new - U_old))
        ite = ite + 1

        G_old = G_new
        U_old = U_new
        
    return G_old, U_old

def post_u(U):
    U_new = U
    if np.any(U_new<1e-10):
        small_idx = U_new<1e-10
        U_new[small_idx] = 1e-10
        U_new = U_new / np.sum(U_new, axis=2, keepdims=True)
    return U_new
    

######
#Module 3: PPTD model parameter estimation
######
def scp_e_step(X, G, U, N, P, h):
    Z = np.zeros((N,h))
    for i in np.arange(N):
        for k in np.arange(h):
            prodt = G[k]
            for j in np.arange(P):
                idx_level = X[i,j]-1
                prodt = prodt * U[j,k,idx_level]
            Z[i,k] = prodt
    return Z / np.sum(Z, axis=1, keepdims=True)

def scp_m_step_g(Z, N):
    G = np.sum(Z, axis=0) / N
    return G

def fw(Alpha, Beta, lambd, FW_ini, d):
    Basis = np.identity(d)
    FW_old = FW_ini
    
    err_fw = 10
    ite_fw = 0
    
    while err_fw>1e-5 and ite_fw<200:
        Gradient = lambd*Beta
        Gradient[FW_old!=0] = Gradient[FW_old!=0] + Alpha[FW_old!=0]/FW_old[FW_old!=0]
        loc = np.argmax(Gradient)
        Dirt = Basis[loc,:]
        
        stp = 2 / (ite_fw + 3)
        
        FW_new = (1-stp)*FW_old + stp*Dirt
        
        err_fw = np.mean(np.abs(FW_new - FW_old))
        ite_fw = ite_fw + 1
        
        FW_old = FW_new   
    return FW_new

def scp_m_step_u(X, Z, U, N, P, d, h, lambd):
    U1 = np.zeros_like(U)
    for j in np.arange(P):
        for k in np.arange(h):
            Alpha = np.zeros(d)
            for c in np.arange(d):
                Alpha[c] = np.sum(Z[:,k] * (X[:,j]==(c+1)))
            Beta = U[j,k,:]
            U1[j,k,:] = fw(Alpha, Beta, lambd, Beta, d)
    return U1

def scp_simple(X, G_ini, U_ini, N, P, d, h, lambd):
    G_old = G_ini
    U_old = U_ini
    
    ite = 0
    err = 10
    while err>1e-5 and ite<15:
        #print(ite)
        #e step
        Z = scp_e_step(X, G_old, U_old, N, P, h)
        #m step
        G_new = scp_m_step_g(Z, N)
        U_new = scp_m_step_u(X, Z, U_old, N, P, d, h, lambd)

        err = np.mean(np.abs(U_new - U_old))
        ite = ite + 1

        G_old = G_new
        U_old = U_new
        
    return G_old, U_old


######
#Module 4: PTD-HT model parameter estimation
######
def cp_ht(X, G_ini, U_ini, N, P, d, h, th_val):
    U_new = copy.deepcopy(U_ini)
    small_idx = U_new<(th_val)
    U_new[small_idx] = 0
    U_new = U_new / np.sum(U_new, axis=2, keepdims=True)
    
    small_idx = U_new<1e-10
    U_new[small_idx] = 1e-10
    U_new = U_new / np.sum(U_new, axis=2, keepdims=True)
    
    Z = cp_e_step(X, G_ini, U_new, N, P, h)
    G_new = cp_m_step_g(Z, N)
    return G_new, U_new


######
#Module 5: the effectiveness of AIC for selecting h
######
#200 daasets are generated and each selects h by AIC
rep_num = 200
AIC_h = np.zeros(rep_num)

h_num = 10
Parnum_over_h = np.array([i-1+P*i*(d-1) for i in np.arange(1,h_num+1)])

for rep in np.arange(rep_num):
    print(rep)
    np.random.seed(rep)
    X = x_generator(G, U, N, P)
    
    LK_over_h = np.zeros(h_num)
    for h in np.arange(1,h_num+1):
        np.random.seed(0)
        G_ini = np.random.dirichlet(np.ones(h), size=1)[0,:]
        U_ini = np.random.dirichlet(np.ones(d), size=(P,h))
        
        G_cp, U_cp, LK_cp = cp(X, G_ini, U_ini, N, P, d, h)
        LK_over_h[h-1] = LK_cp[-1]
    
    AIC = Parnum_over_h * 2 - 2 * LK_over_h
    AIC_h[rep] = np.argmin(AIC)


######
#Module 6: test likelihood when h deviates from 4
######
N_inf = int(1e5)
np.random.seed(0)
X_inf = x_generator(G, U, N_inf, P)

#test likelihood over h
h_num = 10
rep_num = 200
TLK_over_h = np.zeros((rep_num, h_num))

for rep in np.arange(rep_num):
    print(rep)
    np.random.seed(rep)
    X = x_generator(G, U, N, P)
    
    for h in np.arange(1,h_num+1):
        np.random.seed(0)
        G_ini = np.random.dirichlet(np.ones(h), size=1)[0,:]
        U_ini = np.random.dirichlet(np.ones(d), size=(P,h))
        
        G_cp, U_cp = cp_simple(X, G_ini, U_ini, N, P, d, h)
        U_cp = post_u(U_cp)
        
        G_scp, U_scp = scp_simple(X, G_cp, U_cp, N, P, d, h, 9)
        U_scp = post_u(U_scp)
        
        TLK_over_h[rep,h-1] = log_likelihood(X_inf, G_scp, U_scp, N_inf, P, h)


######
#Module 7: model parameter estimation comparison in terms of MSE
######
#estimation error
def mse(G, U, G_hat, U_hat):
    match_idx = np.argsort(-G_hat)
    #mse_g = np.mean((G_hat[match_idx]-G)**2)
    mse_u = np.mean((U_hat[:,match_idx,:]-U)**2)
    return mse_u

rep_num = 200
MSE = np.zeros((rep_num,3))

h = 4
for rep in np.arange(rep_num):
    print(rep)
    np.random.seed(rep)
    X = x_generator(G, U, N, P)
    
    G_cp, U_cp = cp_simple(X, G_ini, U_ini, N, P, d, h)
    U_cp = post_u(U_cp)
    
    G_ht, U_ht = cp_ht(X, G_cp, U_cp, N, P, d, h, 0.02)
    
    G_scp0, U_scp0 = scp_simple(X, G_cp, U_cp, N, P, d, h, 0)
    U_scp0 = post_u(U_scp0)
    
    G_scp, U_scp = scp_simple(X, G_cp, U_cp, N, P, d, h, 9)
    U_scp = post_u(U_scp)
    
    MSE[rep,0] = mse(G, U, G_ht, U_ht)
    MSE[rep,1] = mse(G, U, G_scp0, U_scp0)
    MSE[rep,2] = mse(G, U, G_scp, U_scp)



