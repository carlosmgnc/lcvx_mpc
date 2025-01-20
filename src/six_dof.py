import numpy as np
import cvxpy as cvx
import sympy as sy
from sympy import *

class sixDof:
    def __init__(self, params):
        self.params = params
        self.E_f = self.dynamics()

    # Direction Cosine Matrix Function
    def DCM(self, q): 
        return sy.Matrix(
            [
                [
                    1 - 2 * (q[2] ** 2 + q[3] ** 2),
                    2 * (q[1] * q[2] + q[0] * q[3]),
                    2 * (q[1] * q[3] - q[0] * q[2]),
                ],
                [
                    2 * (q[1] * q[2] - q[0] * q[3]),
                    1 - 2 * (q[1] ** 2 + q[3] ** 2),
                    2 * (q[2] * q[3] + q[0] * q[1]),
                ],
                [
                    2 * (q[1] * q[3] + q[0] * q[2]),
                    2 * (q[2] * q[3] - q[0] * q[1]),
                    1 - 2 * (q[1] ** 2 + q[2] ** 2),
                ],
            ]
        )

    def DCM_np(self, q): 
        return np.array(
            [
                [
                    1 - 2 * (q[2] ** 2 + q[3] ** 2),
                    2 * (q[1] * q[2] + q[0] * q[3]),
                    2 * (q[1] * q[3] - q[0] * q[2]),
                ],
                [
                    2 * (q[1] * q[2] - q[0] * q[3]),
                    1 - 2 * (q[1] ** 2 + q[3] ** 2),
                    2 * (q[2] * q[3] + q[0] * q[1]),
                ],
                [
                    2 * (q[1] * q[3] + q[0] * q[2]),
                    2 * (q[2] * q[3] - q[0] * q[1]),
                    1 - 2 * (q[1] ** 2 + q[2] ** 2),
                ],
            ]
        )
    
    # skew symmetric quaternion matrix
    def omega(self, w):
        return sy.Matrix(
        [
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0],
        ]
    )

    # skew symmetric cross product matrix function
    def cr(self, v):
        return sy.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    
    def cr_np(self, v):
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    

    # returns linearized system matrices as a function
    def dynamics(self):
        f_symb = sy.zeros(14, 1)
        x = sy.Matrix(sy.symbols("m r0 r1 r2 v0 v1 v2 q0 q1 q2 q3 w0 w1 w2", real=True))
        u = sy.Matrix(sy.symbols("u0 u1 u2", real=True))

        g = sy.Matrix(self.params.g)
        rt = sy.Matrix(self.params.rt)
        Jb = sy.Matrix(self.params.Jb)
        Jbinv = sy.Matrix(self.params.Jbinv)

        f_symb[0, 0] = -self.params.alpha * u.norm()
        f_symb[1:4, 0] = x[4:7, 0]

        #f_symb[4:7, 0] =  (1/x[0, 0]) * u + g #(1/x[0, 0]) * self.DCM(x[7:11, 0]).T * u
        f_symb[4:7, 0] =  (1/x[0, 0]) * self.DCM(x[7:11, 0]).T * u +  g
        f_symb[7:11, 0] = (1/2) * self.omega(x[11:14, 0]) * x[7:11, 0]
        f_symb[11:14, 0] = Jbinv * (self.cr(rt) * u - self.cr(x[11:14, 0]) * Jb * x[11:14, 0])

        E_f = sy.lambdify((x, u), f_symb, 'numpy')

        return E_f

    # derivative function for rk4
    def P_dot(self, t, X, u):
        
        test = t
        X_flat = X.flatten()
        u_flat = u.flatten()     

        return self.E_f(X_flat, u_flat)
    
    def controller(self, q, u, w):
        DCM = self.DCM_np(q.flatten())

        print("DCM" + str(DCM))
        print("q" + str(q))
        x = np.array([[1],[0],[0]])
        ub = DCM @ u

        r = self.cr_np(x.flatten()) @ ub
        r = r.reshape((3, 1))

        ub_dir = ub / np.linalg.norm(ub)

        if np.linalg.norm(r) != 0:
            r = r / np.linalg.norm(r)
            theta = np.arccos(x.T @ ub_dir)

            print("theta: " + str(theta))

            kp = 2000
            kd = 10000
            torque = r * kp * theta - kd * w
        else:
            torque = np.zeros((3, 1))

        thrust_y = -torque[2].reshape((1, 1))
        thrust_z = torque[1].reshape((1, 1))
        thrust_x = np.array([[np.sqrt(np.square(np.linalg.norm(ub)) - (np.square(thrust_y[0, 0]) + np.square(thrust_z[0, 0])))]])

        print("normU: " + str(np.linalg.norm(ub)))
        #print(thrust_x)

        thrust = np.vstack([thrust_x,thrust_y, thrust_z])
        # print(u)
        # print(thrust)

        return thrust