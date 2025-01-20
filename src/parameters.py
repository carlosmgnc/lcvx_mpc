import numpy as np


class Params():
    def __init__(self):
        # vehicle properties
        self.mw = 1905
        self.md = 1500
        self.Tmax = 1905 * 3.7114 * 2
        self.Tmin = self.Tmax * 0.3

        self.rt = np.array([[-10], [0], [0]])
        a = (0.25*(self.mw)*100+(1.0/12.0)*(self.mw)*(4*100) )
        self.Jbvec = np.array([[0.3*a], [a], [a]])
        self.Jb = np.diag(self.Jbvec.flatten())
        self.Jbinv = np.diag((1 / self.Jbvec).flatten())
        self.alpha = 0.0007

        # initial trajectory guess
        self.ri = np.array([[4000], [600], [1000]])
        self.vi = np.array([[-60], [-20], [-20]])
        self.vf = np.array([[0], [0], [0]])
        self.qi = np.array([[1],[0],[0],[0]])
        #self.qi = np.array([ [0.9887711], [0], [0.1056687], [0.1056687] ])
        self.wi = np.array([[0],[0],[0]])
        self.g = np.array([[-3.7114], [0], [0]])

        self.tfguess = 5

        self.xi = np.vstack([self.mw, self.ri, self.vi, self.qi, self.wi])