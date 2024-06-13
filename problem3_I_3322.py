import cvxpy as cp
import numpy as np
import random as rd
import math
import cmath

Id = np.identity(2)

lower_bound = []
result_opt = 0

for k in range(10**1):
    phi_A0 = 2*math.pi*rd.random()
    theta_A0 = math.pi*rd.random()    
    ket_v0 = np.zeros(shape=(2,1), dtype='complex')
    ket_v0[0,0] = math.cos(theta_A0/2)
    ket_v0[1,0] = (cmath.exp(1j*phi_A0))*math.sin(theta_A0/2)
    trans_ket_v0 = np.transpose(ket_v0)
    dagket_v0 = np.conjugate(trans_ket_v0)
    A0_opt = 2*np.matmul(ket_v0,dagket_v0) - Id
    
    phi_A1 = 2*math.pi*rd.random()
    theta_A1 = math.pi*rd.random()
    ket_v1 = np.zeros(shape=(2,1), dtype='complex')
    ket_v1[0,0] = math.cos(theta_A1/2)
    ket_v1[1,0] = (cmath.exp(1j*phi_A1))*math.sin(theta_A1/2)
    trans_ket_v1 = np.transpose(ket_v1)
    dagket_v1 = np.conjugate(trans_ket_v1)
    A1_opt = 2*np.matmul(ket_v1,dagket_v1) - Id
    
    phi_A2 = 2*math.pi*rd.random()
    theta_A2 = math.pi*rd.random()    
    ket_v4 = np.zeros(shape=(2,1), dtype='complex')
    ket_v4[0,0] = math.cos(theta_A2/2)
    ket_v4[1,0] = (cmath.exp(1j*phi_A2))*math.sin(theta_A2/2)
    trans_ket_v4 = np.transpose(ket_v4)
    dagket_v4 = np.conjugate(trans_ket_v4)
    A2_opt = 2*np.matmul(ket_v4,dagket_v4) - Id
    
    phi_B0 = 2*math.pi*rd.random()
    theta_B0 = math.pi*rd.random()
    ket_v2 = np.zeros(shape=(2,1), dtype='complex')
    ket_v2[0,0] = math.cos(theta_B0/2)
    ket_v2[1,0] = (cmath.exp(1j*phi_B0))*math.sin(theta_B0/2)
    trans_ket_v2 = np.transpose(ket_v2)
    dagket_v2 = np.conjugate(trans_ket_v2)
    B0_opt = 2*np.matmul(ket_v2,dagket_v2) - Id
    
    phi_B1 = 2*math.pi*rd.random()
    theta_B1 = math.pi*rd.random()
    ket_v3 = np.zeros(shape=(2,1), dtype='complex')
    ket_v3[0,0] = math.cos(theta_B1/2)
    ket_v3[1,0] = (cmath.exp(1j*phi_B1))*math.sin(theta_B1/2)
    trans_ket_v3 = np.transpose(ket_v3)
    dagket_v3 = np.conjugate(trans_ket_v3)
    B1_opt = 2*np.matmul(ket_v3,dagket_v3) - Id
    
    phi_B2 = 2*math.pi*rd.random()
    theta_B2 = math.pi*rd.random()
    ket_v5 = np.zeros(shape=(2,1), dtype='complex')
    ket_v5[0,0] = math.cos(theta_B2/2)
    ket_v5[1,0] = (cmath.exp(1j*phi_B2))*math.sin(theta_B2/2)
    trans_ket_v5 = np.transpose(ket_v5)
    dagket_v5 = np.conjugate(trans_ket_v5)
    B2_opt = 2*np.matmul(ket_v5,dagket_v5) - Id
    
    for p in range(15):
        
        G_I_3322 = (-np.kron(A0_opt, Id) -np.kron(A1_opt, Id) -np.kron(Id, B0_opt) -np.kron(Id, B1_opt) 
        -np.kron(A0_opt, B0_opt) -np.kron(A1_opt, B0_opt) -np.kron(A2_opt,B0_opt) -np.kron(A0_opt, B1_opt) 
        -np.kron(A1_opt, B1_opt) +np.kron(A2_opt, B1_opt) -np.kron(A0_opt, B2_opt) +np.kron(A1_opt, B2_opt))
        
        eigenvalues = np.linalg.eigh(G_I_3322).eigenvalues
        eigenvectors = np.linalg.eigh(G_I_3322).eigenvectors
        largest = eigenvalues[0]
        for q in range(4):
            if eigenvalues[q] > largest:
                largest = eigenvalues[q]
                index = q
        psi = np.zeros(shape=(4,1), dtype='complex')
        for m in range(4):
            psi[m,0] = eigenvectors[m][index]
        trans_psi = np.transpose(psi)
        dagpsi = np.conjugate(trans_psi)
        rho_opt = np.matmul(psi,dagpsi)
        
        if p == 0:
        
            print('Valor inicial de Tr(rho G_I_3322) (sem otimização):', np.real(np.trace(np.matmul(rho_opt,G_I_3322))))
        
        A0_var = cp.Variable(shape=(2,2), hermitian='true')
        A1_var = cp.Variable(shape=(2,2), hermitian='true')
        A2_var = cp.Variable(shape=(2,2), hermitian='true')
        
        G_I_3322_var = (-cp.kron(A0_var, Id) -cp.kron(A1_var, Id) -cp.kron(Id, B0_opt) -cp.kron(Id, B1_opt) 
        -cp.kron(A0_var, B0_opt) -cp.kron(A1_var, B0_opt) -cp.kron(A2_var,B0_opt) -cp.kron(A0_var, B1_opt) 
        -cp.kron(A1_var, B1_opt) +cp.kron(A2_var, B1_opt) -cp.kron(A0_var, B2_opt) +cp.kron(A1_var, B2_opt))
        
        objective = cp.Maximize(cp.real(cp.trace(cp.matmul(rho_opt, G_I_3322_var))))
        constraints = [A0_var + Id >> 0, Id - A0_var >> 0, A1_var + Id >> 0, Id - A1_var >> 0, 
                       A2_var + Id >> 0, Id - A2_var >> 0]
        
        prob = cp.Problem(objective, constraints)
        result = prob.solve()
        
        A0_opt = A0_var.value
        A1_opt = A1_var.value
        A2_opt = A2_var.value
        
        B0_var = cp.Variable(shape=(2,2), hermitian='true')
        B1_var = cp.Variable(shape=(2,2), hermitian='true')
        B2_var = cp.Variable(shape=(2,2), hermitian='true')
        
        G_I_3322_var_1 = (-cp.kron(A0_opt, Id) -cp.kron(A1_opt, Id) -cp.kron(Id, B0_var) -cp.kron(Id, B1_var) 
        -cp.kron(A0_opt, B0_var) -cp.kron(A1_opt, B0_var) -cp.kron(A2_opt,B0_var) -cp.kron(A0_opt, B1_var) 
        -cp.kron(A1_opt, B1_var) +cp.kron(A2_opt, B1_var) -cp.kron(A0_opt, B2_var) +cp.kron(A1_opt, B2_var))
        
        objective_1 = cp.Maximize(cp.real(cp.trace(cp.matmul(rho_opt,G_I_3322_var_1))))
        constraints_1 = [B0_var + Id >> 0, Id - B0_var >> 0, B1_var + Id >> 0, Id - B1_var >> 0, 
                         B2_var + Id >> 0, Id - B2_var >> 0]
        
        prob_1 = cp.Problem(objective_1, constraints_1)
        result_1 = prob_1.solve()
        
        B0_opt = B0_var.value
        B1_opt = B1_var.value
        B2_opt = B2_var.value
        print(k, p, result_1)
        
        if result_1 > result_opt:
            A0_opt_final = A0_opt
            A1_opt_final = A1_opt
            A2_opt_final = A2_opt
            B0_opt_final = B0_opt
            B1_opt_final = B1_opt
            B2_opt_final = B2_opt
            rho_opt_final = rho_opt
        lower_bound.append(result_1)

print('Um limite inferior para a cota de Tsirelson da desigualdade I_3322 <= 4 é: ', max(lower_bound))
print()
print('Os operadores ótimos são: A_0 = ', np.around(A0_opt_final,decimals=2))
print()
print('A_1 = ', np.around(A1_opt_final,decimals=2))
print()
print('A_2 = ', np.around(A2_opt_final,decimals=2))
print()
print('B_0 = ', np.around(B0_opt_final, decimals=2))
print()
print('B_1: ', np.around(B1_opt_final,decimals=2))
print()
print('B_2: ', np.around(B2_opt_final,decimals=2))
print()
print('O estado ótimo é: rho = :', np.around(rho_opt_final,decimals=2))