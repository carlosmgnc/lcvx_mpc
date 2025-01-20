import numpy as np
import cvxpy as cvx
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from parameters import Params


class optProblem:
    def __init__(self, params):

        self.params = params

        self.ri = self.params.ri
        self.vi = self.params.vi
        self.mw = self.params.mw

        self.rho1 = self.params.Tmin
        self.rho2 = self.params.Tmax
        self.z_init = np.log(self.mw)

        self.y0 = np.vstack([self.ri, self.vi, self.z_init])

        self.dt = 1
        self.tmin = (self.params.md) * np.linalg.norm(self.vi) / self.rho2
        self.tmax = (self.mw - self.params.md) / (self.params.alpha * self.rho1)

        self.Ntmin = int(self.tmin / self.dt) + 1
        self.Ntmax = int(self.tmax / self.dt)

        self.cost_list = []
        self.Nt_list = []

        # continuouss time dynamics Ac (7x7), Bc (7x4)
        Ac = np.block(
            [[np.zeros((3, 3)), np.eye(3), np.zeros((3, 1))], [np.zeros((4, 7))]]
        )
        Bc = np.block(
            [
                [np.zeros((3, 4))],
                [np.eye(3), np.zeros((3, 1))],
                [np.zeros((1, 3)), -self.params.alpha],
            ]
        )

        # discretize by computing state transition matrix for 0->dt
        block_n = Ac.shape[1] + Bc.shape[1]
        exp_aug = sp.linalg.expm(
            self.dt * np.block([[Ac, Bc], [np.zeros((block_n - Bc.shape[0], block_n))]])
        )

        self.A = exp_aug[: Ac.shape[0], : Ac.shape[1]]
        self.B = exp_aug[: Ac.shape[0], Ac.shape[1] :]


    ############################# cvx problem #############################

    def solve_cvx_problem(self, Nt):
        time_Nt = np.arange(0, Nt * self.dt, self.dt)

        y = cvx.Variable((7, Nt))
        p = cvx.Variable((4, Nt - 1))

        theta_max = np.deg2rad(10)

        cost = 0
        constraints = []
        constraints += [y[:, [0]] == self.y0]

        # stay above ground constraint
        constraints += [y[0, :] >= 0]

        # hard final state constraint
        constraints += [y[:6, [-1]] == np.array([[0], [0], [0], [0], [0], [0]])]

        # gravity input vector
        g_input = np.vstack([self.params.g[0, 0], np.zeros((3, 1))])

        for k in range(Nt - 1):
            # dynamics constraints
            constraints += [
                y[:, [k + 1]] == self.A @ y[:, [k]] + self.B @ (p[:, [k]] + g_input)
            ]

            # relaxed thrust constraints
            constraints += [cvx.norm(p[:3, [k]]) <= p[3, k]]

            z = y[6, k]
            z0 = np.log(self.mw - self.params.alpha * self.rho2 * time_Nt[k])
            mu1 = self.rho1 * cvx.exp(-z0)
            mu2 = self.rho2 * cvx.exp(-z0)

            constraints += [
                mu1 * (1 - (z - z0) + cvx.power(z - z0, 2) * 0.5) <= p[3, k]
            ]
            constraints += [p[3, k] <= mu2 * (1 - (z - z0))]

            constraints += [z0 <= y[6, k]]
            constraints += [
                y[6, k] <= cvx.log(self.mw - self.params.alpha * self.rho1 * time_Nt[k])
            ]

            constraints += [p[0, k] >= p[3, k] * np.cos(theta_max)]

            cost += p[3, k] * self.dt

        # #final state cost (relaxation on hard terminal constraints)
        #R = np.diag([10000, 10000, 10000, 100000, 10000, 10000 ])
        #cost += cvx.quad_form(y[:6, [-1]], R)

        objective = cvx.Minimize(cost)
        prob = cvx.Problem(objective, constraints)
        prob.solve()
        print("solver status: " + prob.status)

        if prob.status != "optimal":
            opt_cost = float("inf")
            return opt_cost, None, None

        else:
            opt_cost = cost.value
            traj_Nt = y.value
            u_Nt = p.value
            return opt_cost, traj_Nt, u_Nt

    # line search for optimal time of flight using golden section search
    def opt_min_time(self):
        Nt_search = np.arange(self.Ntmin, self.Ntmax, 1)
        g = 2 / (1 + np.sqrt(5))
        a = 0
        b = len(Nt_search)

        while abs(b - a) > 3:
            c = b - int(np.floor((b - a) * g))
            d = a + int(np.ceil((b - a) * g))
            fc, trajc, uc = self.solve_cvx_problem(Nt_search[c])
            fd, _, _ = self.solve_cvx_problem(Nt_search[d])

            self.cost_list.extend([fc, fd])
            self.Nt_list.extend([Nt_search[c], Nt_search[d]])

            if fc < fd:
                b = d
            else:
                a = c

        print(Nt_search[c])
        return Nt_search[c], trajc, uc


# params = Params()
# opt = optProblem(params)
# Nt_opt, traj, u = opt.opt_min_time()
# time = np.arange(0, Nt_opt * opt.dt, opt.dt)