import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time


class EmbeddedSimEnvironment(object):

    def __init__(self, model, dynamics, controller, time=100.0):
        """
        Embedded simulation environment. Simulates the syste given dynamics
        and a control law, plots in matplotlib.

        :param model: model object
        :type model: object
        :param dynamics: system dynamics function (x, u)
        :type dynamics: casadi.DM
        :param controller: controller function (x, r)
        :type controller: casadi.DM
        :param time: total simulation time, defaults to 100 seconds
        :type time: float, optional
        """
        self.model = model
        self.dynamics = dynamics
        self.controller = controller
        self.total_sim_time = time  # seconds
        self.dt = self.model.dt
        self.estimation_in_the_loop = False

    def run(self, x0):
        """
        Run simulator with specified system dynamics and control function.
        """

        print("Running simulation....")
        sim_loop_length = int(self.total_sim_time / self.dt) + 1  # account for 0th
        t = np.array([0])
        x_vec = np.array([x0]).reshape(self.model.n, 1)
        u_vec = np.empty((6, 0))
        e_vec = np.empty((12, 0))

        for i in range(sim_loop_length):

            # Get control input and obtain next state
            x = x_vec[:, -1].reshape(self.model.n, 1)
            u, error = self.controller(x, i * self.dt)
            x_next = self.dynamics(x, u)
            x_next[6:10] = x_next[6:10] / ca.norm_2(x_next[6:10])

            # Store data
            t = np.append(t, t[-1] + self.dt)
            x_vec = np.append(x_vec, np.array(x_next).reshape(self.model.n, 1), axis=1)
            u_vec = np.append(u_vec, np.array(u).reshape(self.model.m, 1), axis=1)
            e_vec = np.append(e_vec, error.reshape(12, 1), axis=1)

        _, error = self.controller(x_next, i * self.dt)
        e_vec = np.append(e_vec, error.reshape(12, 1), axis=1)

        self.t = t
        self.x_vec = x_vec
        self.u_vec = u_vec
        self.e_vec = e_vec
        self.sim_loop_length = sim_loop_length
        return t, x_vec, u_vec

    def visualize(self):
        """
        Offline plotting of simulation data
        """
        variables = list([self.t, self.x_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_vec = self.x_vec
        u_vec = self.u_vec

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig2, (ax5, ax6) = plt.subplots(2)
        ax1.clear()
        ax1.set_title("Astrobee States")
        ax1.plot(t, x_vec[0, :], 'r--',
                 t, x_vec[1, :], 'g--',
                 t, x_vec[2, :], 'b--')
        ax1.legend(["x1", "x2", "x3"])
        ax1.set_ylabel("Position [m]")
        ax1.grid()

        ax2.clear()
        ax2.plot(t, x_vec[3, :], 'r--',
                 t, x_vec[4, :], 'g--',
                 t, x_vec[5, :], 'b--')
        ax2.legend(["x3", "x4", "x5"])
        ax2.set_ylabel("Velocity [m/s]")
        ax2.grid()

        ax3.clear()
        ax3.plot(t, x_vec[6, :], 'r--',
                 t, x_vec[7, :], 'g--',
                 t, x_vec[8, :], 'b--')
        ax3.legend(["x6", "x7", "x8"])
        ax3.set_ylabel("Attitude [rad]")
        ax3.grid()

        ax4.clear()
        ax4.plot(t, x_vec[10, :], 'r--',
                 t, x_vec[11, :], 'g--',
                 t, x_vec[12, :], 'b--')
        ax4.legend(["x9", "x10", "x11"])
        ax4.set_ylabel("Ang. velocity [rad/s]")
        ax4.grid()

        # Plot control input
        ax5.clear()
        ax5.set_title("Astrobee Control inputs")
        ax5.plot(t[:-1], u_vec[0, :], 'r--',
                 t[:-1], u_vec[1, :], 'g--',
                 t[:-1], u_vec[2, :], 'b--')
        ax5.legend(["u0", "u1", "u2"])
        ax5.set_ylabel("Force input [N]")
        ax5.grid()

        ax6.clear()
        ax6.plot(t[:-1], u_vec[3, :], 'r--',
                 t[:-1], u_vec[4, :], 'g--',
                 t[:-1], u_vec[5, :], 'b--')
        ax6.legend(["u3", "u4", "u5"])
        ax6.set_ylabel("Torque input [Nm]")
        ax6.grid()

        plt.show()

    def visualize_error(self):
        """
        Offline plotting of simulation data
        """
        variables = list([self.t, self.e_vec, self.u_vec, self.sim_loop_length])
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")

        t = self.t
        x_vec = self.e_vec
        u_vec = self.u_vec

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig2, (ax5, ax6) = plt.subplots(2)
        ax1.clear()
        ax1.set_title("Trajectory Error")
        ax1.plot(t, x_vec[0, :], 'r--',
                 t, x_vec[1, :], 'g--',
                 t, x_vec[2, :], 'b--')
        ax1.legend(["x1", "x2", "x3"])
        ax1.set_ylabel("Position Error [m]")
        ax1.grid()

        ax2.clear()
        ax2.plot(t, x_vec[3, :], 'r--',
                 t, x_vec[4, :], 'g--',
                 t, x_vec[5, :], 'b--')
        ax2.legend(["x3", "x4", "x5"])
        ax2.set_ylabel("Velocity Error [m/s]")
        ax2.grid()

        ax3.clear()
        ax3.plot(t, x_vec[6, :], 'r--',
                 t, x_vec[7, :], 'g--',
                 t, x_vec[8, :], 'b--')
        ax3.legend(["ex", "ey", "ez"])
        ax3.set_ylabel("Attitude Error [rad]")
        ax3.grid()

        ax4.clear()
        ax4.plot(t, x_vec[9, :], 'r--',
                 t, x_vec[10, :], 'g--',
                 t, x_vec[11, :], 'b--')
        ax4.legend(["x9", "x10", "x11"])
        ax4.set_ylabel("Ang. velocity Error [rad/s]")
        ax4.grid()

        # Plot control input
        ax5.clear()
        ax5.set_title("Astrobee Control inputs")
        ax5.plot(t[:-1], u_vec[0, :], 'r--',
                 t[:-1], u_vec[1, :], 'g--',
                 t[:-1], u_vec[2, :], 'b--')
        ax5.legend(["u0", "u1", "u2"])
        ax5.set_ylabel("Force input [N]")
        ax5.grid()

        ax6.clear()
        ax6.plot(t[:-1], u_vec[3, :], 'r--',
                 t[:-1], u_vec[4, :], 'g--',
                 t[:-1], u_vec[5, :], 'b--')
        ax6.legend(["u3", "u4", "u5"])
        ax6.set_ylabel("Torque input [Nm]")
        ax6.grid()

        plt.show()
        
    def get_convergenc_pose_error(self):
        """
        Calculate the position error at the last state as a Euklidian norm
        :param x_ref_Nt: reference vector at last time step
        :type v: ca.MX vector
        """  
        convergence_pose_error = np.linalg.norm(self.x_vec[0:4,-1]) #-1,0:4],2)
    
        return convergence_pose_error

    def get_convergence_attitude_error(self):
        """
        Calculate the attitude error at the last state as a Euklidian norm
        """
        convergence_attitude_error = np.linalg.norm(self.x_vec[5:8,-1]) #-1,5:8],)
        
        return convergence_attitude_error

    def perf_score(self, max_ct, avg_ct, cvg_t, convergence_pose_error, convergence_attitude_error):
        """
        Calculates a performance score depending on the given inputs
        
        Args:
            max_ct (_type_): maximum computational time taken by your solver call;
            avg_ct (_type_): average computational time taken by your solver call;
            cvg_t (_type_): time the controller takes to converge to within 5cm of the target trajectory and 10 degrees of the desired trajectory attitude
            ss_p (_type_): steady state position error # not used
            ss_a (_type_): steady state attitude error # not used
            convergence_pose_error (_type_): _description_
            convergence_attitude_error (_type_): _description_

        Returns:
            _type_: Performance score
        """
        score = 0.0
        score -= max(round((max_ct - 0.1) * 100, 3), 0.0) * 0.1
        
        # Penalize average above
        if avg_ct > 0.1:
            score += (0.1 - avg_ct) * 30
        else:
            score += max((0.1 - avg_ct), 0.0) * 5
            
        # Factor in convergence time
        score += max((35.0 - cvg_t), 0.0) * 0.1
        # Factor in steady-state errors
        score += (convergence_pose_error) * 100
        score += np.rad2deg(convergence_attitude_error) * 1
        
        return score

    def get_cvg_t(self, traj_dev_pos, traj_dev_att):
        """gives back the time to reach position and attitude of interest

        Args:
            traj_dev_pos (_type_): distance in m as a reference give back the time to reach
            traj_dev_att (_type_): distance in degree as a reference give back the time to reach

        Returns:
            _type_: time to reach both, max of both times
        """
        t = self.t
        t_dist = 1000
        t_deg = 1000
        
        for i in range(0, len(self.e_vec[0])):
            distance = np.linalg.norm(self.e_vec[0:4,i],2)
            if distance < traj_dev_pos:
                t_dist = t[i]
                break
            
        for i in range(0, len(self.e_vec[0])):
            degree = np.rad2deg(np.linalg.norm(self.e_vec[6:9,i],2))
            #print("degree",i, " ", degree)
            if degree < traj_dev_att:
                t_deg = t[i]
                break
            
        return max(t_dist , t_deg)
       
        
    