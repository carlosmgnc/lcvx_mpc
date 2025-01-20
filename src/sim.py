import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sy
from sympy import *
#from plot import Plots
from six_dof import sixDof
from lcvx import optProblem
from parameters import Params
from plot import Plots

class simulation:
    def __init__(self, sixdof ,params):
        self.params = params
        self.sixdof = sixdof
        self.opt = optProblem(params)

        self.nt, x1, self.u1 = self.opt.opt_min_time()
        self.nsim = self.nt
        self.nsim = 60

        self.u = np.zeros((3, self.nsim))

        self.trajectory = np.zeros((14, self.nsim))
        self.trajectory[:, [0]] = self.params.xi

    # rk4 single step function
    def rk41(self, func, tk, xk, u, dt):

        k1 = func(tk, xk, u)
        k2 = func(tk + dt / 2, xk + (dt / 2) * k1, u)
        k3 = func(tk + dt / 2, xk + (dt / 2) * k2, u)
        k4 = func(tk + dt, xk + dt * k3, u)
        output = xk + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return output
    
    # integrate nonlinear dynamics using rk4
    def integrate_full_trajectory(self):
        nsub = 15
        dt_sub = self.opt.dt / (nsub + 1)
        P_temp = self.trajectory[:, [0]]

        nt_opt = self.nt
        for i in range(0, self.nsim-6):#self.nt - 1):

            if i != 0:
                self.opt.ri = self.trajectory[1:4, [i]]
                self.opt.vi = self.trajectory[4:7, [i]]
                self.opt.mw = self.trajectory[0, i]
                self.opt.z_init = np.log(self.opt.mw)
                self.opt.y0 = np.vstack([self.opt.ri, self.opt.vi, self.opt.z_init])

                print("step: " + str(i))
                
                # mpc step
                nt_opt = nt_opt - 2
                if nt_opt < 3:
                    nt_opt = 3

                for k in range(0, 10):
                    cost, x_opt, u_opt = self.opt.solve_cvx_problem(nt_opt)
                    if cost != float("inf"):
                        self.u[:, [i]] = u_opt[:-1, [0]] * np.exp(x_opt[-1, [0]])

                        self.u[:, [i]] = self.sixdof.controller(self.trajectory[7:11, i], self.u[:, i], self.trajectory[11:14, [i]])
                        if np.linalg.norm(self.trajectory[1:4, [i]]) <= 100:
                            v = np.linalg.norm(self.u[:, i]) * np.array([[1], [0], [0]] )
                            self.u[:, [i]] = self.sixdof.controller(self.trajectory[7:11, i], v, self.trajectory[11:14, [i]])
                        break
                    else:
                        self.u[:, [i]] = np.zeros((3, 1))
                        nt_opt = nt_opt + 1



            #integrate 
            for j in range(0, nsub + 1):
                sub_time = i * self.opt.dt + j * dt_sub
                print(self.u[:, [i]])
                P_temp = self.rk41(self.sixdof.P_dot, sub_time, P_temp, self.u[:, [i]], dt_sub)

            self.trajectory[:, [i + 1]] = P_temp

plotter = Plots()
params = Params()
sixdof = sixDof(params)
sim = simulation(sixdof, params)

sim.integrate_full_trajectory()

print("final position: " + str(sim.trajectory[1:4, 0]))
plotter.plot(sim, sim.trajectory, sixdof)
