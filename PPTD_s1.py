# -*- coding: utf-8 -*-
"""

the following code modules can be used to:
(1) generate a HD categorical dataset by categorizing a HD continuous dataset
(2) estimate the PTD model parameters via EM algorithm
(3) estimate the PPTD model parameters via EM algorithm
(4) select the number of latent classes h by AIC
(5) determine the tuning parameter lambda by 5-fold cross validation 
(6) compare the running time of PTD and PPTD
(7) compare the estimated model parameters of PTD and PPTD
(8) compare the test likelihood of PTD and PPTD 
Note: the above modules can be repeatedly applied to different rho's
Note: the modules 2-7 can be applied to the real dataset Beijing_air.csv

"""

import numpy as np
from scipy.stats import multivariate_normal, norm, entropy
import time
import matplotlib.pyplot as plt

#setup
N = 300
P = 50
d = 5


######
#Module 1: generate a HD categorical dataset
######
Mu = np.zeros(P)
rho = 0.98
Sigma = np.zeros((P, P))
for i in np.arange(P):
    for j in np.arange(P):
        Sigma[i,j] = rho**np.abs(i-j)

np.random.seed(0)
Interval_size = (np.random.dirichlet(np.ones(d), size=P).T - 1/d) * 0.75 + 1/d
Cum_size = np.cumsum(Interval_size, axis=0)

Th_prob = np.zeros((d+1, P))
Th_prob[1:d,:] = Cum_size[0:(d-1),:]
Th_prob[d,:] = 1

#continuous data to categorical data
def cont_to_cate(Data, Th_prob, P, d):
    Data_cat = np.zeros_like(Data)
    for j in np.arange(P):
        Th = norm.ppf(Th_prob[:,j], loc=0, scale=1)
        data_cat = Data_cat[:,j]
        for i in np.arange(d):
            data_cat = data_cat + ((Data[:,j]>=Th[i]) & (Data[:,j]<Th[i+1]))*(i+1)
        Data_cat[:,j] = data_cat
    return Data_cat.astype(int)

np.random.seed(0)
Data_raw = multivariate_normal.rvs(mean=Mu, cov=Sigma, size=N)
X = cont_to_cate(Data_raw, Th_prob, P, d) #X is a 300*50 categorical matrix or dataset


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

def reg_log_likelihood(X, G, U, N, P, h, lambd):
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
    reg_l = l + 0.5 * lambd * np.sum(U**2)
    return reg_l

def scp(X, G_ini, U_ini, N, P, d, h, lambd):
    G_old = G_ini
    U_old = U_ini
    Reg_lk = []
    Reg_lk.append(reg_log_likelihood(X, G_old, U_old, N, P, h, lambd))
    
    ite = 0
    err = 10
    while err>1e-5 and ite<20:
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

        Reg_lk.append(reg_log_likelihood(X, G_old, U_old, N, P, h, lambd))
        
    return G_old, U_old, Reg_lk


######
#Module 4: selection of h by AIC 
######

h_num = 20 #try h from 1 to 20
rep_num = 150 #try 150 different initial values in the EM algorithm

#number of PTD model parameters over h
Parnum_over_h = np.array([i-1+P*i*(d-1) for i in np.arange(1,h_num+1)])

#log-likelihood over h
LK_over_h = np.zeros(h_num)
Seed_over_h = np.zeros(h_num)

for h in np.arange(1,h_num+1):
    print(h)
    LK_over_rep = np.zeros(rep_num)
    for rep in np.arange(rep_num):
        np.random.seed(rep)
        G_ini = np.random.dirichlet(np.ones(h), size=1)[0,:]
        U_ini = np.random.dirichlet(np.ones(d), size=(P,h))
        
        G_cp, U_cp, LK_cp = cp(X, G_ini, U_ini, N, P, d, h)
        LK_over_rep[rep] = LK_cp[-1]
    LK_over_h[h-1] = np.max(LK_over_rep)
    Seed_over_h[h-1] = np.argmax(LK_over_rep)

AIC = Parnum_over_h * 2 - 2 * LK_over_h
plt.plot(np.arange(1,h_num+1), AIC)

#optimal value of h
h = 8 
#a good initial value of the EM algorithm when h=8
rad_seed = 127
np.random.seed(rad_seed)
G_ini = np.random.dirichlet(np.ones(h), size=1)[0,:]
U_ini = np.random.dirichlet(np.ones(d), size=(P,h))


######
#Module 5: selection of lambda by cross validation
######
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

f_num = 5 #5-fold cross validation
f_size = int(N/f_num)
N_test = f_size
N_train = N - N_test

rep_cv_num = 10
lamda_num = 51
Lambd = np.linspace(0, 250, num=lamda_num)

CV_lk = np.zeros((rep_cv_num, lamda_num))
for cv in np.arange(rep_cv_num):
    print(cv)
    np.random.seed(cv)
    Re_idx = np.random.permutation(np.arange(N))

    for lam in np.arange(lamda_num):
        print(lam)
        lambd = Lambd[lam]
    
        lk_cv = 0
        for f in np.arange(f_num):
            Select_idx = Re_idx[(f*f_size):(f*f_size+f_size)]
            Select_out = np.array([i in Select_idx for i in np.arange(N)])
            Select_in = ~Select_out
    
            X_train = X[Select_in, :]
            X_test = X[Select_out,:]
    
            G_scp, U_scp = scp_simple(X_train, G_cp, U_cp, N_train, P, d, h, lambd)
            U_scp = post_u(U_scp)
            lk_cv = lk_cv + log_likelihood(X_test, G_scp, U_scp, N_test, P, h)
        
        CV_lk[cv, lam] = lk_cv

plt.plot(np.mean(CV_lk, axis=0))  

     
######
#Module 6: running time comparison between PTD and PPTD 
######
#PTD model
start_time = time.time()
G_cp, U_cp, LK_cp = cp(X, G_ini, U_ini, N, P, d, h)
U_cp = post_u(U_cp)
end_time = time.time()
print(end_time - start_time) 

#PPTD model
start_time = time.time()
G_scp, U_scp, LK_scp = scp(X, G_ini, U_ini, N, P, d, h, 2)
end_time = time.time()
print(end_time - start_time)
    

###### 
#Module 7: model parameter comparion between PPTD and PTD
######
#PPTD model
lambd_opt = 95 #9.5*2*d
G_scp, U_scp, LK_scp = scp(X, G_cp, U_cp, N, P, d, h, lambd_opt)
U_scp = post_u(U_scp)

#PTD model: a special case of PPTD with lambda=0
lambd = 0
G_scp0, U_scp0, LK_scp0 = scp(X, G_cp, U_cp, N, P, d, h, lambd)
U_scp0 = post_u(U_scp0)

#the probability vectors of the first categorical variable U^(1)
j = 0
plt.imshow(U_scp0[j,:,:], vmin=0, vmax=1)
plt.imshow(U_scp[j,:,:], vmin=0, vmax=1)

Entropy = np.zeros((h,2))
Entropy[:,0] = entropy(U_scp0[j,:,:], axis=1)
Entropy[:,1] = entropy(U_scp[j,:,:], axis=1)

#probability mass function of the first two categorical variables
k = 0
Base_scp0 = np.outer(U_scp0[0,k,:], U_scp0[1,k,:])
Base_scp = np.outer(U_scp[0,k,:], U_scp[1,k,:])

col_bound = 0.7
plt.imshow(Base_scp0, vmin=0, vmax=col_bound)
plt.imshow(Base_scp, vmin=0, vmax=col_bound)

