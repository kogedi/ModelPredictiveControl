#from msilib.schema import Control
import numpy as np
import control as ct

from astrobee_1d import Astrobee
from controller import Controller
from simulation import EmbeddedSimEnvironment

# Create pendulum and controller objects
abee = Astrobee(h=0.1)
ctl = Controller()

# Get the system discrete-time dynamics
A, B = abee.one_axis_ground_dynamics()
C = np.array([[0, 0],[0, 0]])
D = np.array([[0],[0]])
# TODO: Get the discrete time system with casadi_c2d
Ad, Bd, Cd, Dd = abee.casadi_c2d(A, B, C, D)
abee.set_discrete_dynamics(Ad, Bd)

abee.poles_zeros(A, B, C, D)
# Plot poles and zeros
abee.poles_zeros(Ad, Bd, Cd, Dd)
#StateSpace Formulation
sys1 = ct.StateSpace(A,B,C,D,0)
poles = ct.pzmap(sys1,False)
print("The poles are ",poles[0])

# Get control gains
ctl.set_system(Ad, Bd,Cd,Dd)
desired_poles = [0.974, 0.984]
K = ctl.get_closed_loop_gain(desired_poles)

# Set the desired reference based on the dock position and zero velocity on docked position
dock_target = np.array([[0.0, 0.0]])
ctl.set_reference(dock_target)

# Starting position
x0 = [1.0, 0.0]
#print ("Was kommt hier raus: ", 0.01*(x0[0]-0))

# Initialize simulation environment
sim_env = EmbeddedSimEnvironment(model=abee,
                                 dynamics=abee.linearized_discrete_dynamics,
                                 controller=ctl.control_law,
                                 time=40.0)
t, y, u = sim_env.run(x0)
sim_env.visualize()

# Disturbance effect
abee.set_disturbance()
time1=40
time2=100
sim_env = EmbeddedSimEnvironment(model=abee,
                                 dynamics=abee.linearized_discrete_dynamics,
                                 controller=ctl.control_law,
                                 time=time1)
t, y, u = sim_env.run(x0)
sim_env.visualize()

# Activate feed-forward gain
ctl.activate_integral_action(dt=0.1, ki=0.028)
t, y, u = sim_env.run(x0)
sim_env.visualize()
