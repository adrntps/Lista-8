import ncpol2sdpa as nc

n_A = 2 # Number of dichotomic observables of party A
n_B = 2 # Number of dichotomic observables of party B

A = nc.generate_operators('A', n_A, hermitian=True)
B = nc.generate_operators('B', n_B, hermitian=True)

subs = {A[i] ** 2 :1 for i in range(n_A)}
subs.update({B[i] ** 2 :1 for i in range(n_B)})
subs.update({A[i]*B[j]:B[j]*A[i] for i in range(n_A) for j in range(n_B)})

objective = A[0]*B[0] + A[0]*B[1] + A[1]*B[0] - A[1]*B[1]

sdp = nc.SdpRelaxation(A+B)
sdp.get_relaxation(level=1, objective=objective, substitutions=subs)
sdp.solve(solver='mosek')
print(sdp.primal, sdp.dual, sdp.status)