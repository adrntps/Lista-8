import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

Prbox = np.zeros(shape=(16,1))
Prbox[0] = 1/2
Prbox[3] = 1/2
Prbox[4] = 1/2
Prbox[7] = 1/2
Prbox[8] = 1/2
Prbox[11] = 1/2
Prbox[13] = 1/2
Prbox[14] = 1/2
# Definindo o comportamento referente à caixa PR.

PL = np.zeros(shape=(16,1))
PL[0] = 1
PL[4] = 1
PL[8] = 1
PL[12] = 1
# Definindo o comportamento local.

PI = np.full((16,1),1/4)
# Definindo o comportamento isotrópico.

alpha = cp.Variable(nonneg=True)
# Definindo o parâmetro \alpha como uma das variáveis do programa linear.

beta = cp.Parameter(nonneg=True)
# Definindo o parâmetro \beta como um parâmetro a ser variado em um loop.

beta_val = np.linspace(0, 1, num=50)
# Retorna números uniformemente espaçados em um determinado intervalo. Em
# particular, escolheu-se o intervalo de validade do parâmetro \beta; e 
# são criados 50 números, i.e., o loop seguinte correrá por 50 valores
# de \beta no intervalo de 0 a 1.

x=[]
y=[]
# Criando duas listas para receber os valores de \beta e \alpha, respectivamente,
# de forma a se plotar um gráfico do \alpha máximo para o qual o comportamento
# ainda é local em função de \beta.

for val in beta_val:
# O loop corre para cada um dos 50 números gerados anteriormente.
    beta.value = val
    # \beta recebe o valor de "val". Esse passo corresponde ao "dado um valor de
    # \beta, no programa linear.
    x.append(beta.value)
    # Salvando o valor de \beta na lista.
    P_abxy = alpha*Prbox + (1-alpha)*(beta*PL + (1-beta)*PI)
    # Definindo a família de comportamentos para um dado \beta, e em função de
    # \alpha.
    gamma = cp.Variable(shape=(5,5), symmetric = True)
    # Definindo a matriz gamma 5x5 e simétrica.
    objective = cp.Maximize(alpha)
    # Definindo a função objetivo: maximizar o valor de \alpha.
    constraints = [alpha <= 1, gamma >> 0, gamma[0][0] == 1, gamma[0][1] == P_abxy[0]+P_abxy[1], gamma[0][2] == P_abxy[8]+P_abxy[9], 
                   gamma[0][3] == P_abxy[0]+P_abxy[2], gamma[0][4] == P_abxy[4]+P_abxy[6], gamma[1][1] == P_abxy[0]+P_abxy[1],
                   gamma[1][3] == P_abxy[0], gamma[1][4] == P_abxy[4], gamma[2][2] == P_abxy[8]+P_abxy[9],  gamma[2][3] == P_abxy[8],
                   gamma[2][4] == P_abxy[12],  gamma[3][3] == P_abxy[0]+P_abxy[2], gamma[4][4] == P_abxy[4]+P_abxy[6]]
    # Restrições da SDP (nível 1 da hierarquia NPA): \alpha <= 1; \gamma >> 0 (\gamma deve ser positivo semi-definida para que o comportamento
    #pertença ao conjunto quântico do nível 1; aqui, para todos os efeitos, Q_1 é o conjunto quântico); as demais restrições estão relacionadas 
    #às componentes da matriz \gamma escritas em termo das componentes do comportamento P_abxy previamente definido.
    prob = cp.Problem(objective, constraints)
    prob.solve()
    y.append(alpha.value)
    print(alpha.value)    
plt.plot(x,y, 'black')