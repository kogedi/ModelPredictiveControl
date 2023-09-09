import casadi as ca
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

#Variablen Definition
N=10
m=4
k=1000
gc=9.81
x1= [[-2],[1]]
xN= [[2],[1]]

#Definiton of variables
x = ca.DM.zeros(1,2*N)
H = ca.DM.zeros(2*N,2*N)
g = ca.DM.zeros(2*N,1)
x_lb = -1*ca.DM.inf(2*N,1)
x_ub = ca.DM.inf(2*N,1)
#Contraints
x_lb[0] = x_ub[0] = -2
x_lb[1] = x_ub[1] = 1
x_lb[2*N-2] = x_ub[2*N-2] = 2
x_lb[2*N-1] = x_ub[2*N-1] = 1
A= ca.DM.zeros(1,2*N)
a_lb = 0
a_ub = 0

# Objective function
H = (2*ca.DM.eye(2*N) -np.eye(2*N,k=2)-np.eye(2*N,k=-2))
H[0,0] = H[1,1] = H[2*N-2,2*N-2] = H[2*N-1,2*N-1] = 1
H = 0.5*k*H
print(H)
g[1::2] = m*gc
print(g)

#Solver
qp = {'h' : H. sparsity () , 'a' : A. sparsity () }
S = ca.conic('S' , 'osqp' ,qp) # ’ qpoases ’ −> s o l v e r
r = S(h=H, g=g, a=A, lbx=x_lb, ubx=x_ub, lba=a_lb , uba=a_ub)

x_opt = r['x']

Y0 = x_opt[0::2]
Z0 = x_opt[1::2]
print("Y0",Y0)
print("Z0",Z0)

plt.plot(Y0, Z0, 'o-')
plt.show()

#plt.plot(r['x'][0],r['x'][1])
# plt.show()
# #print("Optimal sol is selling", int(round(float(r['x'][0]))), "first class and", int(round(float(r['x'][1]))), "second class tickets")