#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 12:50:05 2025

@author: feliceschyns
"""

import numpy as np


class CosMethodForCall:
    def __init__(self, K, r, sigma, Delta_T, N_terms, lambda_, mu_j, sigma_j, model):
        self.K = K
        self.r = r
        self.sigma = sigma
        self.Delta_T =Delta_T
        self.N_terms = N_terms
        self.lambda_ = lambda_
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.model = model
        
     
    
        #changed c1 in both cumuGBM and cumuMerton to be centered around log(S0) because when centered around 0 log(S0) was not in the domain. 
    def cumuGBM(self, S):
        sigma2 = self.sigma**2
        c1 = np.log(S)+self.Delta_T*(self.r - 0.5 * sigma2)
        c2 = self.Delta_T * sigma2
        L = 12 #scaling parameter
        a = c1 - L*np.sqrt(c2)
        b = c1 + L*np.sqrt(c2)
        return a, b
    
    def chFunction(self, u, S):
        
        return np.exp(1j * u * (np.log(S) + self.Delta_T * (self.r - 0.5 * self.sigma**2)) - 0.5 * self.Delta_T * self.sigma**2 * u**2)
        
    def cumuMerton(self, S):
        sigma2 = self.sigma**2
        omegaline = self.lambda_ * (np.exp(0.5 * self.sigma_j**2 + self.mu_j) - 1)
        c1 = np.log(S) +self.Delta_T * (self.r - 0.5 - omegaline -0.5 * sigma2 + self.lambda_ * self.mu_j)
        c2 = self.Delta_T * (sigma2 + self.lambda_ * self.mu_j**2 + self.sigma_j**2 * self.lambda_)
        c4 = self.Delta_T * self.lambda_ * (self.mu_j**4 + 6 * self.sigma_j**2 * self.mu_j**2 + 3 * self.sigma_j**4 * self.lambda_)
        L = 10 #scaling parameter
        a = c1 - L * np.sqrt(c2 + np.sqrt(c4))
        b = c1 + L * np.sqrt( c2 + np.sqrt(c4))
        return a, b
    
    def chFunctionMerton(self, u, S):
        
        return (np.exp(1j * u * (np.log(S) + self.Delta_T * (self.r - 0.5 * self.sigma**2))
               - 0.5 * self.sigma**2 * self.Delta_T * u**2)
        * np.exp(self.lambda_ * self.Delta_T * (
            np.exp(1j * self.mu_j * u - 0.5 * self.sigma_j**2 * u**2) - 1)))
        
    def chi_k(self, k, a, b):
        omega_k = k * np.pi / (b - a)
        chi_b = np.exp(b) * (omega_k * np.sin(omega_k * (b - a)) + np.cos(omega_k * (b - a)))
        chi_a = self.K * (omega_k * np.sin(omega_k * (np.log(self.K) - a)) + np.cos(omega_k * (np.log(self.K) - a)))
        value_chi = (chi_b - chi_a)/(1 + omega_k ** 2)
        return value_chi
    
    def psi_k (self, k, a, b):
        omega_k = k * np.pi / (b - a)
        if k==0:
            return b-np.log(self.K)
        psi_b = np.sin(omega_k * (b - a))# set d = b
        psi_a = np.sin(omega_k * (np.log(self.K) - a))# set c = log(X2)= log(self.K)
        value_psi = (psi_b - psi_a) / omega_k 
        return value_psi
        
    
    def H_inner_k (self, k, a, b):
        value_H_inner = (self.chi_k(k, a, b)- self.K * self.psi_k(k, a, b)) * (2 / (b - a))
        return value_H_inner
    
    def F_inner_k (self, k, a, b, S):
        omega_k = k * np.pi / (b - a)
        if self.model=="GBM":
            chf = self.chFunction(omega_k, S)
        elif self.model =="Merton":
            chf = self.chFunctionMerton(omega_k, S)
        else:
            chf = self.chFunction(omega_k, S)
        
        value_F_inner = np.real(np.exp(1j * omega_k * (-a)) * chf) 
        return value_F_inner
    
    
    def C_inner_call(self, S): #If statement so the code knows which model to use
       if self.model == 'GBM':
           a, b = self.cumuGBM(S)
       elif self.model == 'Merton':
           a, b = self.cumuMerton(S)
       else:
           a, b = self.cumuGBM(S)
       result = 0 
        
       for k in range(self.N_terms):
            if k==0:#first term summation gets weighed by a half 
                result += 0.5 * self.F_inner_k(k, a, b, S) * self.H_inner_k(k, a, b)
            else:
                result += self.F_inner_k(k, a, b, S) * self.H_inner_k(k, a, b)
       return np.exp(-self.r * self.Delta_T)*result 

#We now continu to write the code for C_CoC. Need to approximate using midpoint rule. 
#This is the class for European Call on Call part of the option
#uses numerical integration for the payoff coefficients

class CosMethodForCoC:
    def __init__(self, S0, r, sigma, X1, X2, T1, T2, n_out, N_out, N_in, lambda_, mu_j, sigma_j, model):
        self.S0 = S0      # stock price at time t=0
        self.X_1 = X_1      # set as X1
        self.r = r
        self.sigma = sigma
        self.n_out = n_out #midpoint rule numerical 
        self.N_out = N_out #COS terms
        self.Delta_T = T_1     
        self.lambda_ = lambda_
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.model = model
        self.RegularCall = CosMethodForCall(K = X_2, r=self.r, sigma=self.sigma, Delta_T= T_2 - T_1, N_terms = N_in , lambda_ =self.lambda_, mu_j = self.mu_j, sigma_j = self.sigma_j, model=self.model)

  

    def a_b_outer(self):
        
        if self.model=="GBM":
            c1 = np.log(self.S0) + (self.r - 0.5 * self.sigma**2) * self.Delta_T
            c2 = self.sigma**2 * self.Delta_T
            L = 30 #L can be changed to desired value for GBM and Merton
            a_out = c1 - L * np.sqrt(c2)
            b_out = c1 + L * np.sqrt(c2)
            
        elif self.model=="Merton":
            sigma2 = sigma**2
            omegaline = self.lambda_*(np.exp(0.5*self.sigma_j**2+self.mu_j)-1)
            c1 = np.log(self.S0)+ self.Delta_T * (self.r - 0.5 - omegaline - 0.5 * sigma2 + self.lambda_ * self.mu_j)
            c2 = self.Delta_T * (sigma2 + self.lambda_ * self.mu_j**2 + self.sigma_j**2 * self.lambda_)
            c4 = self.Delta_T * self.lambda_ * (self.mu_j**4 + 6 * self.sigma_j**2 * self.mu_j**2 + 3 * self.sigma_j**4 * self.lambda_)
            L = 10 #scaling parameter
            a_out = c1 - L * np.sqrt(c2 + np.sqrt(c4))
            b_out = c1 + L * np.sqrt(c2 + np.sqrt(c4))
            
        else:
            raise ValueError(f"Model '{self.model}' not recognized. Use 'GBM' or 'Merton'.")
        return a_out, b_out
        
    def chFunction(self, u):
        S = self.S0
        return np.exp(1j * u * (np.log(S) + self.Delta_T * (self.r - 0.5 * self.sigma**2)) - 0.5 * self.Delta_T * self.sigma**2 * u**2)
    
    def chFunctionMerton(self, u):
        S = self.S0
        return (np.exp(1j * u * (np.log(S) + self.Delta_T * (self.r - 0.5 * self.sigma**2))
               - 0.5 * self.sigma**2 * self.Delta_T * u**2)
        * np.exp(self.lambda_ * self.Delta_T * (
            np.exp(1j * self.mu_j * u - 0.5 * self.sigma_j**2 * u**2) - 1)))
        
    def F_outer_n(self, n):
        a_out, b_out = self.a_b_outer( )
        omega_n = n * np.pi / (b_out - a_out)
        if self.model=="GBM":
            chfX = self.chFunction(omega_n)   
        elif self.model =="Merton":
            chfX = self.chFunctionMerton(omega_n)
        else:
            chfX = self.chFunction(omega_n)
        value_F_outer = np.real(np.exp(1j * omega_n * (-a_out)) * chfX)
        return value_F_outer
    

   #H_outer_n payoff coefficients are now part of CoC to keep the numerical integration (midpoint rule) from repeating for every n. Code is faster this way
   
    def C_CoC(self):
        a_out, b_out = self.a_b_outer( )
        dx = (b_out - a_out) / self.n_out
        value_compoundoption=0.0
        
       #midpoints calculation
        x_j = a_out + (np.arange(self.n_out)+0.5) * dx 
        
        #C_inners for every midpoint in vector
        C_inner = np.array([self.RegularCall.C_inner_call(np.exp(x_j)) for x_j in x_j])
        payoffs = np.maximum(C_inner-self.X_1,0)
        # now the cosine terms.  
        n_vector = np.arange(self.N_out) 
        n_column = n_vector.reshape(-1, 1) 
        omega_n = n_column * np.pi/(b_out-a_out) 
        cosine = np.cos(omega_n * (x_j -a_out)) #(N_out,nout) matrix with broadcast
        
        #combine all of these for H_outer calculation within C_CoC
        H = (2 / (b_out - a_out)) * (np.matmul(cosine, payoffs )) * dx
        for n in range(self.N_out):
            F = self.F_outer_n(n)
            #H = self.H_outer_n(n, X_1, T_1, T_2, n_out, X_2, N)
            factor = 0.5 if n == 0 else 1.0  # COS: first term of summation gets weighed by 1/2 -> might be better approach then for loop as above
            value_compoundoption += factor * np.exp(-self.r * self.Delta_T) * F * H[n]
            
        return value_compoundoption
    
     
S0 = 150
X_1= 58.37
X_2 = 197.22
r = 0.0484 #risk-free-rate 
sigma = 0.25
T_1 = 5 #exercise time of compound part
T_2 = 9 #exercise time of regular call part (C_inner) -> T_1 < T_2
lambda_ = 0 #jump intensity
mu_j = 0 #jump sizes
sigma_j = 0 # jump volatility
model ='Merton'   

N_in = 500  #cosine series cut for inner
N_out = 500 #cosine series outer
n_out = 500 #midpoint rule terms
    
method = CosMethodForCoC(S0, r, sigma, X_1, X_2, T_1, T_2, n_out, N_out, N_in, lambda_, mu_j, sigma_j, model)
print('COS Compound value',method.C_CoC())

Inititial Upload of the COS code, valuing compound CoC options using COS method. European options.
