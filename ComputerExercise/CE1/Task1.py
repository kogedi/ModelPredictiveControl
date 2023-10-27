import casadi as ca
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt


#Maximize cost fuction
#       J = 2000 *x_1 + x_2*1500
#           x_1 = first class tickets
#       x =
#           x_2 = second class tickets
#Contraints:
#       x_1 > 20
#       x_2 > 35

## DEFINITION OF CONSTRAINTS
lb_x1 = 35
lb_x2 = 35
profit_x1 = 1500
profit_x2 = 1500
max_passengers = 130

#casadi
x_ub=ca.DM.zeros(2,1)
x_lb=ca.DM.zeros(2,1)
A = ca.DM.zeros(1, 2)
A[0] = 1
A[1] = 1

#Init of our bounds
x_ub = ca.inf * np.ones(2)
x_lb[0] = lb_x1
x_lb[1] = lb_x2
a_ub = max_passengers
a_lb = lb_x1 +lb_x2

# objective function
H = ca.DM.zeros(2,2)
g = ca.DM.ones(2,1)
g[0] = [profit_x1]
g[1] = [profit_x2]
#Transform to minimization problem
g = -1*g

g = ca.DM.zeros((2, 1))
g[0, 0] = profit_x1
g[1, 0] = profit_x2
g = -1 * g

qp = {'h' : H. sparsity () , 'a' : A. sparsity () }
S = ca.conic('S' , 'qpoases' ,qp) # ’ qpoases ’ −> s o l v e r
r = S(h=H, g=g, a=A, lbx=x_lb, ubx=x_ub, lba=a_lb , uba=a_ub)
x_opt = r [ 'x' ]
print("r[x]", r['x'])
print("Profit: ", -r['cost'])
print("Optimal sol is selling", int(round(float(r['x'][0]))), "first class and", int(round(float(r['x'][1]))), "second class tickets")