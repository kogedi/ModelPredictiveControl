import numpy as np

from astrobee import Astrobee
from mpc import MPC
from simulation import EmbeddedSimEnvironment

import getpass

# Get the username of the currently logged-in user
username = getpass.getuser()

#print("Username:", username)
#Set the path to the trajectory file:
#complete the 'tuning_file_path' variable to the path of your tuning.yaml

if username == 'Konrad Dittrich':
    trajectory_quat = 'C:/Users/Konrad Dittrich/git/repos/Model_Predictive_Control/Assignments/Project Assignment/Dataset/trajectory_quat.txt'
    tuning_file_path = 'C:/Users/Konrad Dittrich/git/repos/Model_Predictive_Control/Assignments/Project Assignment/tuning.yaml'
elif username == 'catherine':
    trajectory_quat = '/Users/catherine/git/Model_Predictive_Control/Assignments/Project Assignment/Dataset/trajectory_quat.txt'
    tuning_file_path = '/Users/catherine/git/Model_Predictive_Control/Assignments/Project Assignment/tuning.yaml'
else:
    trajectory_quat = '/home/el2700/Documents/Project Assignment/Dataset/trajectory_quat.txt'

    tuning_file_path = '/home/el2700/Documents/Project Assignment/tuning.yaml'
# Q1
# TODO: Set the Astrobee dynamics in Astrobee->astrobee_dynamics_quat
abee = Astrobee(trajectory_file=trajectory_quat)

# If successful, test-dynamics should not complain
abee.test_dynamics()

# Instantiate controller
u_lim, x_lim = abee.get_limits()

# Create MPC Solver
# TODO: Select the parameter type with the argument param='P1'  - or 'P2', 'P3'
MPC_HORIZON = 8
ctl = MPC(model=abee,
          dynamics=abee.model,
          param='P2',
          N=MPC_HORIZON,
          ulb=-u_lim, uub=u_lim,
          xlb=-x_lim, xub=x_lim,
          tuning_file=tuning_file_path)

# Q2: Reference tracking
# TODO: adjust the tuning.yaml parameters for better performance
x_d = abee.get_static_setpoint()
ctl.set_reference(x_d)
# Set initial state
x0 = abee.get_initial_pose()
sim_env = EmbeddedSimEnvironment(model=abee,
                                 dynamics=abee.model,
                                 controller=ctl.mpc_controller,
                                 time=80)
t, y, u = sim_env.run(x0)
sim_env.visualize()  # Visualize state propagation

# Q3: Activate Tracking
# TODO: complete the MPC class for reference tracking

#'''''''''''''''''
# Chose different parameter options
paramChoice  = 'P4'
#'''''''''''''''''
mpc_solverops = {} #dictionary
mpc_solverops['ipopt.tol'] = 1e-3
tracking_ctl = MPC(model=abee,
                   dynamics=abee.model,
                   param=paramChoice,
                   N=MPC_HORIZON,
                   trajectory_tracking=True,
                   ulb=-u_lim, uub=u_lim,
                   xlb=-x_lim, xub=x_lim,
                   tuning_file=tuning_file_path,solver_opts=mpc_solverops)
sim_env_tracking = EmbeddedSimEnvironment(model=abee,
                                          dynamics=abee.model,
                                          controller=tracking_ctl.mpc_controller,
                                          time=80)
t, y, u = sim_env_tracking.run(x0)


sim_env_tracking.visualize()  # Visualize state propagation
sim_env_tracking.visualize_error()

# Test 3: Activate forward propagation
# TODO: complete the MPC Astrobee class to be ready for forward propagation
abee.test_forward_propagation()
tracking_ctl.set_forward_propagation()
t, y, u = sim_env_tracking.run(x0)
# sim_env_tracking.visualize()  # Visualize state propagation
sim_env_tracking.visualize_error()

#Print maximum and average solvetime
max_ct = tracking_ctl.get_max_solvetime()
avg_ct = tracking_ctl.get_avg_solvetime()
cvg_t = sim_env_tracking.get_cvg_t(traj_dev_pos=0.05, traj_dev_att=10)
convergence_attitude_error = sim_env_tracking.get_convergence_attitude_error()
convergenc_pose_error = sim_env_tracking.get_convergenc_pose_error()
perscore = sim_env_tracking.perf_score(avg_ct, max_ct, cvg_t, convergenc_pose_error, convergence_attitude_error)

print("********************************************************")
print("** Parameters",paramChoice,"MPC_HORIZON:",MPC_HORIZON,"mpc_solverops:", mpc_solverops['ipopt.tol'],"**")
print("Average cpu time to solve avg_ct = " ,round(avg_ct,4))
print("Maximum cpu time to solve max_ct = ", round(max_ct,4))
print("Time to converge to solve cvg_t = ", round(cvg_t,4))
print("The convergence_attitude_error = ", round(convergence_attitude_error,4))
print("The convergenc_pose_error = ", round(convergenc_pose_error,4))
#print("********************************************************")
#print("Performance Score for", paramChoice,"is: ", round(perscore,4))
print("********************************************************")
