from .physics_model import StabilityModel,BatchStabilityModel
import scipy.sparse
import numpy as np
from .physics_solver import GUROBISolver
class RBEModelDiag(StabilityModel):
    def __init__(self, rho = 1, nt = 4, name = "", qp_solver=None, mu=0.5, Ccp=1E5,glue_strength=0, **args):
        self.glue_strength = glue_strength
        super(RBEModelDiag, self).__init__(rho = rho, nt = nt, name = name, qp_solver=qp_solver, mu=mu, Ccp=Ccp, **args)

    @property
    def nx(self):
        return (1 + self.nt) * self.nλ + self.nf

    @property
    def nA(self):
        return self.nλ

    @property
    def Knt(self):
        if not hasattr(self, '_Knt'):
            Inf = scipy.sparse.eye_array(self.nf, dtype=np.float64)
            self._Knt = scipy.sparse.block_array([[self.Jn.T, self.Jt.T, -Inf]])
        return self._Knt

    def qp_matrices(self):
        if not hasattr(self, '_L'):
            Inλ = scipy.sparse.eye_array(self.nλ, dtype=np.float64)
            Onλ = scipy.sparse.coo_matrix((self.nλ, self.nf), dtype=np.float64)
            Inx = scipy.sparse.eye_array(self.nx, dtype=np.float64)
            self._L = self.invML.T @ self.Knt
            self._A = scipy.sparse.block_array([[self.mu * Inλ, -self.E, Onλ]])
            self._eq = []
        return [self._L, self._A, self._eq]

    def qp_bounds(self, part_states: np.ndarray):
        if part_states.ndim == 1:
            part_states = part_states.reshape(1, -1)
        assert part_states.ndim == 2
        nbatch = part_states.shape[0]

        ps = self.p(part_states)
        cs = self.c(part_states)
        Pg = ps * self.g[:, None]
        q = self.Knt.T @ self.invM @ Pg
        #g = np.tile(self.g[:, None], (1, nbatch))
        #q = self.Knt.T @ self.invM @ g

        Al = np.zeros((self.nλ, nbatch), dtype=np.float64)
        Au = np.ones((self.nλ, nbatch), dtype=np.float64) * self.mu * self.Ccp
        xl = np.vstack([-self.glue_strength*np.ones((self.nλ, nbatch), dtype=np.float64),
                        -self.glue_strength*np.ones((self.nλ * self.nt, nbatch), dtype=np.float64),
                        -self.Ccp * (1 - ps)])
        xu = np.vstack([self.Ccp * cs,
                        np.tile(self.Ccp * cs, (self.nt, 1)),
                        self.Ccp * (1 - ps)])
        return q, xl, xu, Al, Au

class BatchRBEModel(BatchStabilityModel):
    def __init__(self, rho = 1, nt = 4, name = "", qp_solver=None, mu=0.5, Ccp=1E5,glue_strength=0, **args):
        self.glue_strength = glue_strength
        super(BatchRBEModel, self).__init__(rho = rho, nt = nt, name = name, qp_solver=qp_solver, mu=mu, Ccp=Ccp, **args)

    @property
    def Knt(self):
        if not hasattr(self, '_Knt'):
            Inf = scipy.sparse.eye_array(self.nf, dtype=np.float64)
            self._Knt = scipy.sparse.block_array([[self.Jn.T, self.Jt.T, -Inf]])
        return self._Knt

    def qp_matrices(self):
        if not hasattr(self, '_L'):
            Inλ = scipy.sparse.eye_array(self.nλ, dtype=np.float64)
            Onλ = scipy.sparse.coo_matrix((self.nλ, self.nf), dtype=np.float64)
            self._L = self.Knt
            self._A = scipy.sparse.block_array([[self.mu * Inλ, -self.E, Onλ]])
            self._eq = []
            # Invalidate warm start: problem structure changed (new contacts or blocks)
            if hasattr(self, 'qp_solver') and self.qp_solver is not None:
                self.qp_solver.clear()
        return [self._L, self._A, self._eq]

    def qp_bounds(self, part_states: np.ndarray):
        if part_states.ndim == 1:
            part_states = part_states.reshape(1, -1)
        assert part_states.ndim == 2
        nbatch = part_states.shape[0]

        ps = self.p(part_states)
        cs = self.c(part_states)
        Pg = ps * self.g[:, None]
        q = self.Knt.T @ self.invM @ Pg
        #g = np.tile(self.g[:, None], (1, nbatch))
        #q = self.Knt.T @ self.invM @ g

        Al = np.zeros((self.nλ, nbatch), dtype=np.float64)
        Au = np.ones((self.nλ, nbatch), dtype=np.float64) * self.mu * self.Ccp
        xl = np.vstack([-self.glue_strength*np.ones((self.nλ, nbatch), dtype=np.float64),
                        -self.glue_strength*np.ones((self.nλ * self.nt, nbatch), dtype=np.float64),
                        -self.Ccp * (1 - ps)])
        xu = np.vstack([self.Ccp * cs,
                        np.tile(self.Ccp * cs, (self.nt, 1)),
                        self.Ccp * (1 - ps)])
        return q, xl, xu, Al, Au
class RBEModel(RBEModelDiag):
    def __init__(self, rho = 1, nt = 4, name = "", qp_solver=None, mu=0.5, Ccp=1E5, **args):
        super(RBEModel, self).__init__(rho = rho, nt = nt, name = name, qp_solver=qp_solver, mu=mu, Ccp=Ccp, **args)
    def qp_matrices(self):
        if not hasattr(self, '_L'):
            Inλ = scipy.sparse.eye_array(self.nλ, dtype=np.float64)
            Onλ = scipy.sparse.coo_matrix((self.nλ, self.nf), dtype=np.float64)
            Inx = scipy.sparse.eye_array(self.nx, dtype=np.float64)
            self._P = self.M @ self.Knt
            self._A = scipy.sparse.block_array([[self.mu * Inλ, -self.E, Onλ]])
            self._eq = []
        return [self._L, self._A, self._eq]

    def qp_bounds(self, part_states: np.ndarray):
        if part_states.ndim == 1:
            part_states = part_states.reshape(1, -1)
        assert part_states.ndim == 2
        nbatch = part_states.shape[0]

        ps = self.p(part_states)
        cs = self.c(part_states)
        Pg = ps * self.g[:, None]
        q = self.Knt.T @ self.invM @ Pg
        #g = np.tile(self.g[:, None], (1, nbatch))
        #q = self.Knt.T @ self.invM @ g

        Al = np.zeros((self.nλ, nbatch), dtype=np.float64)
        Au = np.ones((self.nλ, nbatch), dtype=np.float64) * self.mu * self.Ccp
        xl = np.vstack([-self.glue_strength*np.ones((self.nλ, nbatch), dtype=np.float64),
                        -self.glue_strength*np.ones((self.nλ * self.nt, nbatch), dtype=np.float64),
                        -self.Ccp * (1 - ps)])
        xu = np.vstack([self.Ccp * cs,
                        np.tile(self.Ccp * cs, (self.nt, 1)),
                        self.Ccp * (1 - ps)])
        return q, xl, xu, Al, Au