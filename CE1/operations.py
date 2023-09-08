import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Numpy operations

# Q1: take
A = np.zeros((9, 9))
# Knowing that in Python the iterator format is:
# start:stop:step , so 1::2 means start at 1, stop at the end
#                   and use a step of 2 -> 1, 3, 5, ...
# Change the matrix A so that the border of A is 0 but the core
# of A follows the pattern in:
#                       [0, 0, 0, 0, 0]
#                       [0, 1, 0, 1, 0]
#                       [0, 0, 0, 0, 0]
#                       [0, 1, 0, 1, 0]
#                       [0, 0, 0, 0, 0]

# Q2: Take now
A = np.eye(3) * 10
b = np.linspace(1, 3, 3)
# Calculate and print
# i) A b
# ii) b^T A b
# iii) A^{-1} b
# iv) || b || (norm-2)


# Q3: With np.linspace or np.arange and np.sin, plot a sine function
#     using plt.plot(x, y) and plt.show()


# Q4: Take now a random array such as
b_rnd = np.random.uniform(low=0, high=10, size=(1001))
# Get the 2nd largest value in the array by creating a for loop
# that looks for this item and stores its index too.
# Useful functions: range(), len()

# Q5: Minimize x**2 + y**2 using the ca.conic solver, where the bounds on x and y
#     should be 0 < x < inf, 0 < y < inf
#     Note: check CasADi's documentation to help and guide you in this question.
# Fill the variables: g, H, A, lbx, ubx, lba, uba. Then, create a dictionary with
# {'h': H.sparsity(), 'a': A.sparsity()} and solve the problem with
# S = ca.conic('S', 'osqp', qp)
# r = S(h=H, g=g, a=A, lbx=lbx, ubx=ubx, lba=lba, uba=uba)
#
# Then r['x'][0] and r['x'][1] will contain the values for x and y, respectively
