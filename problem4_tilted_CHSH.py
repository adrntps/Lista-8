import ncpol2sdpa as nc
import numpy as np
import matplotlib.pyplot as plt

alpha_val = np.linspace(0, 1, num=20)

x=[]
y=[]
y_prime=[]

n_A = 2 # Number of dichotomic observables of party A
n_B = 2 # Number of dichotomic observables of party B

for val in alpha_val:
    alpha = val
    x.append(alpha)
    
    A = nc.generate_operators('A', n_A, hermitian=True)
    B = nc.generate_operators('B', n_B, hermitian=True)
    
    subs = {A[i] ** 2 :1 for i in range(n_A)}
    subs.update({B[i] ** 2 :1 for i in range(n_B)})
    subs.update({A[i]*B[j]:B[j]*A[i] for i in range(n_A) for j in range(n_B)})
    
    objective = alpha*A[0] + A[0]*B[0] + A[0]*B[1] + A[1]*B[0] - A[1]*B[1]
    
    sdp = nc.SdpRelaxation(A+B)
    sdp.get_relaxation(level=2, objective=objective, substitutions=subs)
    sdp.solve(solver='mosek')
    print(-sdp.primal, -sdp.dual, sdp.status)
    y.append(-sdp.primal)
    y_prime.append((8 + 2*(alpha**2))**(1/2))
plt.xlabel('alpha')
plt.ylabel('Máxima violação')
plt.grid()
plt.plot(x,y, marker = 'o', color = 'red')
plt.plot(x,y_prime,'black')
