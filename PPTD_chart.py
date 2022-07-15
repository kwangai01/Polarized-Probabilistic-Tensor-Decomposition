# -*- coding: utf-8 -*-
"""

the following code modules can be used to:
(1) build the ewma chart based on the proportion vectors
(2) compute IC ARLs of PPTD: the true model parameter is not changed 
(3) compute OC ARLs of PPTD: the true model parameter is changed
Note: the categorical dataset can be generated based on continuous datasets (scenario 1)
or based on the true mode parameters (scenario 2)
Note: Module 3 can be repeatedly used for different OC cases
Note: Module 1 can be applied to the real dataset Beijing_air.csv
"""


import numpy as np
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt


#setup
N = 300
P = 50
d = 5
h = 8

#generate categorical dataset
Mu = np.zeros(P)
rho = 0.98
Sigma = np.zeros((P, P))
for i in np.arange(P):
    for j in np.arange(P):
        lag_rho = np.abs(i-j)
        Sigma[i,j] = rho**lag_rho
np.all(np.sort(np.linalg.eigvals(Sigma)) > 0)
plt.imshow(Sigma)

#threshold
np.random.seed(0)
Interval_size = (np.random.dirichlet(np.ones(d), size=P).T - 1/d) * 0.75 + 1/d
np.min(Interval_size)
np.max(Interval_size)
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


######PTD model parameter estimation
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
    
def cp(X, G_ini, U_ini, N, P, d, h):
    G_old = G_ini
    U_old = U_ini
    
    ite = 0
    err = 10
    while err>1e-6 and ite<100:
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


######PPTD model parameter estimation
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

def scp(X, G_ini, U_ini, N, P, d, h, lambd):
    G_old = G_ini
    U_old = U_ini
    
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
        
    return G_old, U_old


######
#Module 1: build an EWMA chart for the proportion vector
######
def eta_onedata(x, G, U, P, h):
    z = np.zeros(h)
    for k in np.arange(h):
        prodt = G[k]
        for j in np.arange(P):
            idx_level = x[j]-1
            prodt = prodt * U[j,k,idx_level]
        z[k] = prodt
    return z / np.sum(z)

def eta_alldata(X, G, U, P, h):
    return np.apply_along_axis(eta_onedata, 1, X, G, U, P, h)

def eta_mean_cov_train(X, G, U, P, h):
    Z = eta_alldata(X, G, U, P, h)
    Mean_z = np.mean(Z, axis=0)
    Cov_z = np.cov(Z, rowvar=False)
    return Mean_z, Cov_z
    
def eta_mean_cov_appx(G, h):
    Cov_appx = np.zeros((h,h))
    for k1 in np.arange(h):
        for k2 in np.arange(h):
            Cov_appx[k1,k2] = -1 * G[k1] * G[k2]
            if k1==k2:
                Cov_appx[k1,k2] = G[k1] - G[k1]**2
    return G, Cov_appx

def eta_ewma(Z, z_ini, N, w):
    Z_ewma = np.zeros_like(Z)
    z_old = z_ini
    for i in np.arange(N):
        z = Z[i,:]
        z_new = (1-w)*z_old + w*z
        Z_ewma[i,:] = z_new
        z_old = z_new
    return Z_ewma

def X_generator(X, N, N_inf):
    np.random.seed(666)
    Idx = np.random.randint(low=0, high=N, size=N_inf)
    X_inf = X[Idx,:]
    return X_inf

def pear_chi_sq(Z_ewma, Mean_z, Cov_z_inv):
    Pear_chi2 = np.diag((Z_ewma-Mean_z).dot(Cov_z_inv).dot((Z_ewma-Mean_z).T))
    return Pear_chi2

def cl_est(X, X_inf, G, U, Mean_z, Cov_z_inv, N, N_inf, P, h, w, lag):
    X_all = np.concatenate((X, X_inf), axis=0)
    Z_all = eta_alldata(X_all, G, U, P, h) 
    Z_ewma = eta_ewma(Z_all, Mean_z, N+N_inf, w)[lag:,]
    Pear_chi2 = pear_chi_sq(Z_ewma, Mean_z, Cov_z_inv)
    return np.quantile(Pear_chi2, 0.995)

def rl_rep(X, Mu, Sigma, Th_prob, G, U, Mean_z, Cov_z_inv, N, P, d, h, w, lag, cl, n_rl):
    RL = np.zeros(n_rl)
    for rep_rl in np.arange(n_rl):
        np.random.seed(rep_rl)
        Data_raw_rep = multivariate_normal.rvs(mean=Mu, cov=Sigma, size=1000)
        X_rep = cont_to_cate(Data_raw_rep, Th_prob, P, d)
        
        X_all_rep = np.concatenate((X, X_rep), axis=0)
        Z_all_rep = eta_alldata(X_all_rep, G, U, P, h) 
        Z_ewma_rep = eta_ewma(Z_all_rep, Mean_z, N+1000, w)[lag:,]
        Pear_chi2_rep = pear_chi_sq(Z_ewma_rep, Mean_z, Cov_z_inv)
        
        if np.all(Pear_chi2_rep<cl):
            RL[rep_rl] = 1000
        else:
            RL[rep_rl] = np.argmax(Pear_chi2_rep>=cl)
    return RL


lambd = 2
N_inf = int(1e4)
w = 0.10
lag = 300

ini_seed = 127
np.random.seed(ini_seed)
G_ini = np.random.dirichlet(np.ones(h), size=1)[0,:]
U_ini = np.random.dirichlet(np.ones(d), size=(P,h))


######
#Module 2: compute IC ARLs of PPTD: OC parameters = IC parameters
######
Mu_oc = np.zeros(P)
Sigma_oc = np.zeros((P, P))
for i in np.arange(P):
    for j in np.arange(P):
        lag_rho = np.abs(i-j)
        Sigma_oc[i,j] = (rho-0.00)**lag_rho
        
        
######
#Module 3: compute OC ARLs of PPTD: OC parameters /= IC parameters
######
Mu_oc = np.zeros(P)
Mu_oc[np.arange(0,P)] = 0.8
Sigma_oc = np.zeros((P, P))
for i in np.arange(P):
    for j in np.arange(P):
        lag_rho = np.abs(i-j)
        Sigma_oc[i,j] = (rho-0.00)**lag_rho

n_dataset = 1000
n_rl = 10000
RL_dataset = []


for rep_dataset in np.arange(n_dataset):
    print(rep_dataset)
    
    np.random.seed(rep_dataset)
    Data_raw = multivariate_normal.rvs(mean=Mu, cov=Sigma, size=N)
    X = cont_to_cate(Data_raw, Th_prob, P, d)
    
    G_cp, U_cp = cp(X, G_ini, U_ini, N, P, d, h)
    U_cp = post_u(U_cp)
    G_hat, U_hat = scp(X, G_cp, U_cp, N, P, d, h, lambd)
    U_hat = post_u(U_hat)
    
    Mean_z, Cov_z = eta_mean_cov_appx(G_hat, h)
    try:
        Cov_z_inv = np.linalg.inv(Cov_z)
    except:
        continue
    
    X_inf = X_generator(X, N, N_inf)
    cl_hat = cl_est(X, X_inf, G_hat, U_hat, Mean_z, Cov_z_inv, 
                    N, N_inf, P, h, w, lag)
    
    RL_hat = rl_rep(X, Mu_oc, Sigma_oc, Th_prob, G_hat, U_hat, Mean_z, Cov_z_inv, 
                    N, P, d, h, w, lag, cl_hat, n_rl)
    
    RL_dataset.append(RL_hat)
    

RL_dataset = np.array(RL_dataset)
ARL = np.nanmean(RL_dataset, axis=1)
np.mean(ARL)
np.std(ARL)