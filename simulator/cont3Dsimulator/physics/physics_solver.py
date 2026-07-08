import gurobipy as gp
from gurobipy import GRB
import piqp
import math
import numpy as np
from scipy import sparse
import torch
import scipy as sp
from time import perf_counter

import numpy as np
import torch
from torch import nn
from .helper import *


# solving a batch of qps using GPUs
# min 0.5 (Lx)^TLx + q^Tx + reg x^Tx
# s.t. Al <= Ax <= Au
#      xl <=  x <= xu
# eq: row indices where Al == Au
# reg: regularization coefficient
class QPSolver:
    def __init__(self, eps = 1E-6, reg = 0, verbose = False, **args):
        self.eps_abs = eps
        self.verbose = verbose
        self.regularization = reg
        self.args = args

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def clear(self):
        pass

    """
    set_callback: a callback function to evulate the feasibility of current data points 
    """
    def set_callback(self, callback):
        self.callback = callback

    def update_problem(self,L,A,eq, q, xl, xu, Al, Au):
        self.L = L.copy().todense()
        self.A = A.copy().todense()
        self.eq = eq.copy()

        self.xl, self.xu = xl, xu
        self.Al, self.Au = Al, Au
        self.q = q
        self.zl = np.vstack([self.Al, self.xl])
        self.zu = np.vstack([self.Au, self.xu])

    def update_initial_guess(self, inds):
        pass

    def solve(self, inds):
        pass

    @property
    def nx(self):
        return self.A.shape[1]

    @property
    def nAh(self):
        return self.A.shape[0] + self.nx

    @property
    def nA(self):
        return self.A.shape[0]

    @property
    def nr(self):
        return self.rs.shape[0]

    def inf_norm(self, x):
        return torch.max(torch.abs(x), dim = 0).values

    def start(self):
        torch.cuda.synchronize()
        self.start_timer = perf_counter()

    def stop(self, name="", n_batch=1):
        torch.cuda.synchronize()
        end = perf_counter()
        #print("{} time {:.2e}".format(name, (end - self.start_timer) / n_batch))

class Callback:

    def __init__(self, physics_model, record_flag = False):
        self.physics_model = physics_model
        self.record_flag = record_flag


    def set_part_states(self, part_states):
        self.ps = tensor(self.physics_model.p(part_states))
        if self.record_flag:
            self.nbatch = part_states.shape[0]
            self.npart = part_states.shape[1]
            self.record_vs = []

    @property
    def Jn(self):
        if not hasattr(self, '_Jn'):
            self._Jn = sparse(self.physics_model.Jn)
        return self._Jn

    @property
    def Jt(self):
        if not hasattr(self, '_Jt'):
            self._Jt = sparse(self.physics_model.Jt)
        return self._Jt

    @property
    def g(self):
        if not hasattr(self, '_g'):
            self._g = tensor(self.physics_model.g)
        return self._g

    @property
    def invM(self):
        if not hasattr(self, '_invM'):
            self._invM = sparse(self.physics_model.invM)
        return self._invM

    def forces(self, x):
        return x[:self.physics_model.nλ], x[self.physics_model.nλ:self.physics_model.nλ * (self.physics_model.nt + 1)]

    def eval(self, x, inds):
        λn, λt = self.forces(x)
        r = self.Jn.T @ λn + self.Jt.T @ λt + self.g[:, None]
        r = r * self.ps[:, inds]
        vs = self.invM @ r
        vs_inf = torch.max(torch.abs(vs), dim=0).values#inf norm
        if self.record_flag:
            self.record(vs, inds)
        return vs_inf
class GUROBIStaticSolver():
    def __init__(self, nx, nc, eps = 1E-6, reg = 0, verbose = False, **args):
        super(GUROBIStaticSolver, self).__init__(eps, reg, verbose,n_block_max=100,n_contact_max=100, **args)
        self.env = gp.Env()
        self.env.setParam('OptimalityTol', self.eps_abs)
        self.env.setParam('OutputFlag', self.verbose)
        self.env.setParam('Method', -1)
        self.model = gp.Model(env = self.env)
        self.c = np.ones(nx,dtype=np.float64)
        self.nx = nx
        self.nc = nc
        self.x = self.model.addMVar(nx, lb=0, ub=gp.GRB.INFINITY)
        self.nx_current = 0
        self.model.setObjective(self.c @ self.x, gp.GRB.MINIMIZE)
    def build_model(self, A_eq, b_eq, A,b, xl, xu):
        nx = xl.shape[0]
        self.c = np.ones(nx,dtype=np.float64) #shape: (nx)
        self.A_eq = A_eq.copy() #shape: (neq, nx)
        self.b_eq = b_eq.copy() #shape: (neq)
        self.A = A.copy() #shape: (n, nx)
        self.b = b.copy() #shape: (n)
        self.xl = xl.copy() #shape: (nx)
        self.xu = xu.copy() #shape: (nx)
        self.model = gp.Model(env = self.env)
        nx = self.Q.shape[0]
        self.model.setbounds(self.x[:nx], self.xl, self.xu)
        self.model.addConstr(self.A_eq @ self.x[:nx] == self.b_eq)
        for Ai in range(self.A.shape[0]):
            self.model.addConstr(self.A @ self.x <= self.b)
        self.nx_current = nx
        self.model.update()
    def update_problem(self, Aeq_row, Aeq_col, A_row, A_col, b_eq, b, xl, xu):
        #row contains the corner of the block matrix
        nnx = Aeq_col.shape[1]
        self.model.setbounds(self.x[self.nx_current:self.nx_current+nnx], xl, xu)
        for i in range(nnx):
            col = gp.Column()
            Aeq_col_i = Aeq_col[:, i]
            idx_eq = np.nonzero(Aeq_col_i, axis=0)[0]
            coef_eq = Aeq_col_i[idx_eq]
            col.addTerms(coef_eq, self.x[idx_eq])
        self.model.update()
        
    def solve(self, inds):
        import gurobipy as gp
        import numpy as np
        flags, xs = [], []
        nx, nA = self.q.shape[0], self.Al.shape[0]
        
        for id in inds:
            xli, xui, Ali, Aui, qi = self.xl[:, id], self.xu[:, id], self.Al[:, id], self.Au[:, id], self.q[:, id]
            
            # Update bounds, linear objective, and constraint RHS dynamically
            self.x.lb = xli
            self.x.ub = xui
            self.x.Obj = qi
            self.constr_A_lb.RHS = Ali
            self.constr_A_ub.RHS = Aui
            
            self.model.optimize()

            if self.model.Status == gp.GRB.OPTIMAL:
                flags.append(True)
                xk = np.clip(self.x.X, xli, xui)
                xs.append(xk)
            else:
                print("Fail")
                flags.append(False)
                xs.append(np.zeros(nx))

        xs = np.array(xs).T
        xs = xs.reshape(nx, -1)
        return xs, np.array(flags)
class GUROBISolver(QPSolver):

    def __init__(self, eps=1E-6, reg=0, verbose=False, n_batch=1, **args):
        super(GUROBISolver, self).__init__(eps, reg, verbose, **args)
        self.env = gp.Env()
        self.env.setParam('OptimalityTol', self.eps_abs)
        self.env.setParam('OutputFlag', self.verbose)
        self.env.setParam('Method', 1)  # primal simplex: benefits from x.Start warm start
        self.n_batch = n_batch
        self.x_warm = None
        self._L_blocks = None

    def clear(self):
        self.x_warm = None

    def update_problem(self, L, A, eq, q, xl, xu, Al, Au):
        self.xl, self.xu = xl, xu
        self.Al, self.Au = Al, Au
        self.q = q

        nb = getattr(self, 'n_batch', 1)
        nλ_total = Al.shape[0]
        nf_total = L.shape[0]
        nx_total = q.shape[0]
        self._nλ_b = nλ_total // nb
        self._nf_b = nf_total // nb
        self._nx_b = nx_total // nb

        self._L_blocks, self._A_blocks, self._col_indices = [], [], []

        if nλ_total == 0:
            # No contacts — nothing to decompose; solve() will return trivial results
            self._nt = 0
            return

        # nx_total = (1+nt)*nλ_total + nf_total  →  nt = (nx_total - nf_total)//nλ_total - 1
        self._nt = (nx_total - nf_total) // nλ_total - 1
        nλ_b, nf_b, nx_b, nt = self._nλ_b, self._nf_b, self._nx_b, self._nt

        L_arr = np.asarray(L.todense()) if hasattr(L, 'todense') else np.asarray(L)
        A_arr = np.asarray(A.todense()) if hasattr(A, 'todense') else np.asarray(A)

        for b in range(nb):
            cols = (list(range(b * nλ_b, (b+1) * nλ_b)) +
                    list(range(nλ_total + b * nλ_b * nt, nλ_total + (b+1) * nλ_b * nt)) +
                    list(range(nλ_total * (1+nt) + b * nf_b, nλ_total * (1+nt) + (b+1) * nf_b)))
            self._L_blocks.append(L_arr[b*nf_b:(b+1)*nf_b][:, cols])
            self._A_blocks.append(A_arr[b*nλ_b:(b+1)*nλ_b][:, cols])
            self._col_indices.append(cols)

    def solve(self, inds):
        flags, xs_list = [], []
        nb = getattr(self, 'n_batch', 1)
        nx_b  = self._nx_b
        nλ_b  = self._nλ_b
        nx_total = nx_b * nb

        # No contacts: all envs are trivially stable
        if not self._L_blocks:
            xs_out = np.zeros((nx_total, len(inds)))
            return xs_out, np.ones(len(inds), dtype=bool)

        if self.x_warm is None or self.x_warm.shape != (nx_b, nb):
            self.x_warm = np.zeros((nx_b, nb))

        for b in inds:
            cols = self._col_indices[b]
            L_b  = self._L_blocks[b]
            A_b  = self._A_blocks[b]
            xli  = self.xl[cols, b]
            xui  = self.xu[cols, b]
            Ali  = self.Al[b*nλ_b:(b+1)*nλ_b, b]
            Aui  = self.Au[b*nλ_b:(b+1)*nλ_b, b]
            qi   = self.q[cols, b]

            m = gp.Model(env=self.env)
            x = m.addMVar(nx_b, lb=xli, ub=xui)
            x.Start = self.x_warm[:, b]
            y = m.addMVar(L_b.shape[0], lb=-GRB.INFINITY, ub=GRB.INFINITY)
            m.setObjective(0.5 * y @ y + qi @ x + self.regularization * x @ x, GRB.MINIMIZE)
            m.addConstr(L_b @ x == y)
            m.addConstr(A_b @ x <= Aui)
            m.addConstr(A_b @ x >= Ali)
            m.optimize()

            if m.Status == GRB.OPTIMAL:
                xk = np.clip(x.X, xli, xui)
                self.x_warm[:, b] = xk
                flags.append(True)
                xs_list.append((b, xk))
            else:
                print(f"Gurobi: env {b} infeasible (status {m.Status})")
                flags.append(False)
                xs_list.append((b, np.zeros(nx_b)))

        # Scatter per-env solutions into full-batch layout (nx_total, nb)
        xs_full = np.zeros((nx_total, nb))
        for b, xk in xs_list:
            xs_full[self._col_indices[b], b] = xk
        xs_out = xs_full[:, list(inds)]
        return xs_out, np.array(flags)
    

# solving a batch of qps using GPUs
# min 0.5 x^TPx + q^Tx
# s.t. Al <= Ax <= Au
#      xl <=  x <= xu
# eq: row indices where Al == Au

class ADMMSolver(QPSolver):
    def __init__(self, eps = 1E-6, reg = 0, verbose = False, **args):
        super(ADMMSolver, self).__init__(eps, reg, verbose, **args)

        # parameters
        self.sigma = args.get('sigma', 1E-6)
        self.rs = args.get('rs', [0.1])
        self.rs = tensor(self.rs)
        self.req = args.get('req', 1E3)

        self.alpha = args.get('alpha', 1.6)
        self.eps_conv = args.get('eps_conv', 1e-6)
        #self.eps_pinf = args.get('eps_pinf', 0)
        self.max_iter = args.get("max_iter", 5000)
        self.evaluate_iter = args.get("evaluate_iter", 50)
        torch.set_float32_matmul_precision('highest')

    @property
    def nbatch(self):
        return self.inds.shape[0]

    @property
    def nbatch_origin(self):
        return self.xl.shape[1]

    def clear(self):
        for var in ['x0', 'y0', 'z0', 'fea0']:
            if hasattr(self, var):
                del self.__dict__[var]

    def update_initial_guess(self, remain):
        remain = tensor(remain, data_type=bool)
        for var in ['x0', 'y0', 'z0']:
            if hasattr(self, var):
                self.__dict__[var] = self.__dict__[var][:, remain]

    def set_problem(self, L: sp.sparse.coo_matrix, A: sp.sparse.coo_matrix, eq: np.ndarray):
        self.clear()
        self.eq = eq

        # A & Ah
        self.A = sparse(A)
        Inx = sp.sparse.coo_matrix(sp.sparse.eye_array(self.nx, dtype=np.float64))
        Ah = sp.sparse.block_array([[A], [Inx]])
        self.Ah = sparse(Ah)
        self.AhT = self.Ah.t()

        # Q
        Q = sp.sparse.coo_matrix(L.T @ L)
        self.Q = sparse(Q, shape = (self.nx, self.nx))

        # rho
        self.prefactorize(L, A)

    def update_problem(self, q, xl, xu, Al, Au):
        self.xl, self.xu = tensor(xl), tensor(xu)
        self.Al, self.Au = tensor(Al), tensor(Au),
        self.q = tensor(q)
        self.zl = torch.vstack([self.Al, self.xl])
        self.zu = torch.vstack([self.Au, self.xu])

    def step(self):
        self.xk, self.yk, self.zk, self.dyk = self.admm(self.xk, self.yk, self.zk)

    def solve(self, inds):

        # initialize xk, yk, zk, qk, zlk, zuk, Rk, invRk, invLk
        self.initialize_solver(inds)

        # admm iterations
        for self.iter in np.arange(stop = self.max_iter, step = self.evaluate_iter):

            # compute
            self.step()

            # evualuation
            stable, conv = self.evaluate(self.xk, self.yk, self.zk, self.dyk)
            terminate = self.termination(stable, conv)
            if self.remove(terminate):
                break

        return self.process_result()

    # cpu
    def prefactorize(self, L, A):

        # local device, float type
        ndevice = torch.device('cpu') if device.type == 'mps' else device
        float64 = torch.float64

        # R
        R = eye(self.nAh, data_type=float64, device=ndevice)
        R[self.eq, self.eq] *= self.req

        # diagR
        diagR = ones(self.nAh, data_type=float64, device=ndevice)
        diagR[self.eq] *= self.req

        # Rs, invRs, Ls
        self.Rs = zeros(self.nr, self.nAh, data_type=float64, device=ndevice)
        self.invRs = zeros(self.nr, self.nAh, data_type=float64, device=ndevice)
        Ls = zeros(self.nr, self.nx, self.nx, data_type=float64, device=ndevice)

        # Q, A
        Ld = torch.tensor(L.todense(), dtype=float64, device=ndevice)
        Qd = Ld.T @ Ld
        Inx = eye(self.nx, data_type=float64, device=ndevice)
        Ad = torch.tensor(A.todense(),  dtype=float64, device=ndevice)
        Ahd = torch.vstack([Ad, Inx])
        AhTd = Ahd.t()

        for id in range(self.nr):
            r = self.rs[id].item()
            L = (Qd + self.sigma * Inx + AhTd @ (R * r) @ Ahd)
            Ls[id, :, :] = L
            self.Rs[id, :] = diagR * r
            self.invRs[id, :] = 1.0 / diagR / self.rs[id].item()

        cholesky_Ls = torch.linalg.cholesky(Ls)

        # collect
        self.Ls = Ls
        self.invLs = torch.cholesky_inverse(cholesky_Ls).type(floatType).to(device)
        self.Rs = self.Rs.type(floatType).to(device)
        self.invRs = self.invRs.type(floatType).to(device)

    def initialize_solver(self, inds):
        self.inds0 = tensor(inds, data_type=int)
        self.inds = tensor(inds, data_type=int)
        self.flags = zeros(self.nbatch_origin, data_type=bool)

        # init geuss
        if not hasattr(self, 'x0'):
            self.x0 = zeros(self.nx, self.nbatch_origin)
            self.z0 = zeros(self.nAh, self.nbatch_origin)
        else:
            self.z0 = self.Ah @ self.x0
        if not hasattr(self, 'y0'):
            self.y0 = zeros(self.nAh, self.nbatch_origin)
        self.fea0 = zeros(self.nbatch_origin)

        # set xk, yk, zk
        self.xk, self.yk, self.zk = self.x0, self.y0, self.z0
        self.qk, self.zlk, self.zuk = self.q, self.zl, self.zu

        # update variables
        remain = zeros(self.nbatch_origin, data_type=bool)
        remain[self.inds] = True
        self.update_variables(remain)

        # initialize matrices
        self.R, self.invR, self.invL = self.Rs[0, :], self.invRs[0, :], self.invLs[0, :, :]

    #@torch.compile
    def admm(self, x_k1, y_k1, z_k1):
        with torch.no_grad():
            for it in range(self.evaluate_iter):
                xk, yk, zk = x_k1, y_k1, z_k1
                rhs = self.sigma * xk - self.qk + torch.sparse.mm(self.AhT, self.R[:, None] * zk - yk)
                xh_k1 = (self.invL @ rhs)
                x_k1 = (xh_k1 * self.alpha + xk * (1 - self.alpha))
                zh_k1 = torch.sparse.mm(self.Ah, xh_k1)
                z_alpha = self.alpha * zh_k1 + (1 - self.alpha) * zk
                z_k1 = z_alpha + self.invR[:, None] * yk
                z_k1 = torch.clip(z_k1, self.zlk, self.zuk)
                y_k1 = yk + self.R[:, None] * (z_alpha - z_k1)
        return x_k1, y_k1, z_k1, y_k1 - yk

    def evaluate(self, xk, yk, zk, dyk):
        pfea, dfea = self.evaluate_prime_dual_fea(xk, yk, zk)
        #pinf = self.evaluate_prime_inf(dyk)

        if hasattr(self, 'callback'):
            stable = self.evaluate_callback(xk)
        else:
            stable = torch.maximum(pfea, dfea)

        if self.verbose:
            self.print_evaluation(stable, pfea, dfea)
        self.record_solution(xk, yk, zk, stable)

        return stable, torch.maximum(pfea, dfea)

    def evaluate_prime_dual_fea(self, xk, yk, zk):
        prim = torch.sparse.mm(self.Ah, xk) - self.zk
        dual = torch.sparse.mm(self.Q, xk) + self.qk + torch.sparse.mm(self.AhT, yk)
        prim_norm = self.inf_norm(prim)
        dual_norm = self.inf_norm(dual)
        return prim_norm, dual_norm

    def evaluate_prime_inf(self, dyk):
        pinf = torch.sparse.mm(self.AhT, dyk)
        pinf_norm = self.inf_norm(pinf)
        dy_norm = self.inf_norm(dyk)
        return pinf_norm / dy_norm

    def evaluate_callback(self, xk):
        #xclip = xk
        xclip = torch.clip(xk, self.zlk[self.nA:, :], self.zuk[self.nA:, :])
        return self.callback.eval(xclip, self.inds)

    def print_evaluation(self, stable, pfea, dfea):
        print(f"iter {self.iter}")
        print('stable:\t', stable)
        print("prime fea:\t", pfea)
        print("dual fea:\t", dfea)
        #print("prime inf:\t", pinf)
        print("instances", self.inds)

    def record_solution(self, xk, yk, zk, stable):
        self.x0[:, self.inds] = xk.detach().clone()
        self.y0[:, self.inds] = yk.detach().clone()
        self.z0[:, self.inds] = zk.detach().clone()
        self.fea0[self.inds] = stable.detach().clone()

    def termination(self, stable, conv):
        terminate = torch.logical_or(stable < self.eps_abs, conv < self.eps_conv)
        return terminate

    def remove(self, terminate):
        # remove
        if terminate.sum() > 0:
            self.flags[self.inds[terminate]] = True
            remain = torch.logical_not(terminate)
            self.inds = self.inds[remain]
            if remain.sum() == 0:
                return True
            else:
                self.update_variables(remain)
        return False

    def update_variables(self, remain):
        self.xk = self.xk[:, remain]
        self.yk = self.yk[:, remain]
        self.zk = self.zk[:, remain]
        self.zlk = self.zlk[:, remain]
        self.zuk = self.zuk[:, remain]
        self.qk = self.qk[:, remain]

    def process_result(self):
        x0 = self.x0[:, self.inds0]
        x0 =  torch.clip(x0, self.zl[self.nA:, self.inds0], self.zu[self.nA:, self.inds0])
        return array(x0), array(self.flags[self.inds0])

class ADMMFunction(nn.Module):
    def __init__(self, sigma, alpha, Ah, AhT, R, invR, invL, evaluate_iter):
        super(ADMMFunction, self).__init__()

        # Register non-learnable buffers that need to move with the model
        self.register_buffer('Ah', Ah)
        self.register_buffer('AhT', AhT)
        self.register_buffer('R', R)
        self.register_buffer('invR', invR)
        self.register_buffer('invL', invL)

        # Scalar does not need registration
        self.sigma = sigma
        self.alpha = alpha
        self.evaluate_iter = evaluate_iter
        self.nx = Ah.shape[1]
        self.ny = Ah.shape[0]

    def split(self, x):
        ns = [self.nx, self.ny, self.ny, self.nx, self.ny, self.ny]
        vars = []
        start = 0
        for n in ns:
            vars.append(x[:, start : start + n].T)
            start += n
        return vars

    def forward(self, x):
        x_k1, y_k1, z_k1, qk, zlk, zuk = self.split(x)
        for it in range(self.evaluate_iter):
            xk, yk, zk = x_k1, y_k1, z_k1
            rhs = self.sigma * xk - qk + torch.sparse.mm(self.AhT, self.R[:, None] * zk - yk)
            xh_k1 = (self.invL @ rhs)
            x_k1 = (xh_k1 * self.alpha + xk * (1 - self.alpha))
            zh_k1 = torch.sparse.mm(self.Ah, xh_k1)
            z_alpha = self.alpha * zh_k1 + (1 - self.alpha) * zk
            z_k1 = z_alpha + self.invR[:, None] * yk
            z_k1 = torch.clip(z_k1, zlk, zuk)
            y_k1 = yk + self.R[:, None] * (z_alpha - z_k1)
        return torch.hstack([xk.T, yk.T, zk.T, (y_k1 - yk).T])


class ADMMSolver_nGPUs(ADMMSolver):

    def initialize_solver(self, inds):
        super(ADMMSolver_nGPUs, self).initialize_solver(inds)
        admm_func = ADMMFunction(self.sigma, self.alpha, self.Ah, self.AhT, self.R, self.invR, self.invL, self.evaluate_iter)
        self.parallel_admm = torch.nn.DataParallel(admm_func)

    def split(self, x):
        ns = [self.nx, self.nAh, self.nAh, self.nAh]
        vars = []
        start = 0
        for n in ns:
            vars.append(x[:, start: start + n].T)
            start += n
        return vars

    def step(self):
        x = torch.hstack([self.xk.T, self.yk.T, self.zk.T, self.qk.T, self.zlk.T, self.zuk.T])
        y = self.parallel_admm(x)
        self.xk, self.yk, self.zk, self.dyk = self.split(y)

class ADMMSolver2(ADMMSolver):

    def initialize_solver(self, inds):
        self.inds0 = tensor(inds, data_type=int)
        self.inds = tensor(inds, data_type=int)
        self.flags = zeros(self.nbatch_origin, data_type=bool)

        # initialize guess
        if not hasattr(self, 'x0'):
            self.x0 = zeros(self.nx, self.nbatch_origin * self.nr)
            self.z0 = zeros(self.nAh, self.nbatch_origin * self.nr)
        else:
            self.z0 = self.Ah @ self.x0
        if not hasattr(self, 'y0'):
            self.y0 = zeros(self.nAh, self.nbatch_origin * self.nr)
        self.fea0 = zeros(self.nbatch_origin * self.nr)

        # initialize variables
        self.xk, self.yk, self.zk = self.x0, self.y0, self.z0
        self.Rk = self.Rs.T.repeat_interleave(self.nbatch_origin, dim=1)
        self.invRk = self.invRs.T.repeat_interleave(self.nbatch_origin, dim=1)
        self.qk = torch.tile(self.q, (1, self.nr))
        self.zlk = torch.tile(self.zl, (1, self.nr))
        self.zuk = torch.tile(self.zu, (1, self.nr))

        # remove varaibles not in inds
        remain = zeros(self.nbatch_origin, data_type=bool)
        remain[self.inds] = True
        self.update_variables(remain)

    @torch.compile
    def admm(self, x_k1, y_k1, z_k1):
        with torch.no_grad():
            # rhs
            for it in range(self.evaluate_iter):
                xk, yk, zk = x_k1, y_k1, z_k1

                rhs = self.sigma * xk - self.qk + self.AhT @ (self.Rk * zk - yk)

                # solving lhs^{-1} rhs
                rhs_rxb = rhs.view(self.nx, self.nr, self.nbatch).transpose(0, 1)
                xh_k1_rxb = torch.bmm(self.invLs, rhs_rxb)
                xh_k1 = xh_k1_rxb.transpose(0, 1).flatten(start_dim=1)

                # update x
                x_k1 = (xh_k1 * self.alpha + xk * (1 - self.alpha))

                # update z
                zh_k1 = torch.sparse.mm(self.Ah, xh_k1)
                z_alpha = self.alpha * zh_k1 + (1 - self.alpha) * zk
                z_k1 = z_alpha + self.invRk * yk
                z_k1 = torch.clip(z_k1, self.zlk, self.zuk)

                # update y
                y_k1 = yk + self.Rk * (z_alpha - z_k1)

        return [x_k1, y_k1, z_k1, y_k1 - yk]

    def evaluate_callback(self, xk):
        xclip = xk
        #xclip = torch.clip(xk, self.zlk[self.nA:, :], self.zuk[self.nA:, :])
        inds = self.inds.repeat(self.nr)
        return self.callback.eval(xclip, inds)

    def print_evaluation(self, stable, pfea, dfea):
        best_rind, best_r = self.evaluate_best_r(stable, self.nbatch)
        stable_best = stable.index_select(0, best_rind)
        pfea_best = pfea.index_select(0, best_rind)
        dfea_best = dfea.index_select(0, best_rind)
        super(ADMMSolver2, self).print_evaluation(stable_best, pfea_best, dfea_best)
        if self.verbose:
            print("best_r:\t", best_r)

    def record_solution(self, xk, yk, zk, stable):
        nbatch_now = int(xk.shape[1] / self.nr)
        offset = (arange(0, self.nr) * self.nbatch_origin).repeat_interleave(nbatch_now)
        batch_inds = self.inds.repeat(self.nr) + offset
        self.x0[:, batch_inds] = xk.detach().clone()
        self.y0[:, batch_inds] = yk.detach().clone()
        self.z0[:, batch_inds] = zk.detach().clone()
        self.fea0[batch_inds] = stable.detach().clone()

    def termination(self, stable, conv):
        best_rinds, best_r = self.evaluate_best_r(stable, self.nbatch)
        stable_best = stable.index_select(0, best_rinds)

        best_rinds, best_r = self.evaluate_best_r(conv, self.nbatch)
        conv_best = conv.index_select(0, best_rinds)

        terminate = torch.logical_or(stable_best < self.eps_abs, conv_best < self.eps_conv)
        return terminate

    def update_variables(self, remain):
        remain = remain.repeat(self.nr)
        super(ADMMSolver2, self).update_variables(remain)
        self.Rk = self.Rk[:, remain]
        self.invRk = self.invRk[:, remain]

    def evaluate_best_r(self, val, nbatch):
        val = torch.where(torch.isnan(val), torch.tensor(float('inf')), val)
        bval = val.view(self.nr, -1).T
        best_r = bval.argmin(1)
        return best_r * nbatch + arange(0, nbatch), best_r

    def process_result(self):
        best_rind, best_r = self.evaluate_best_r(self.fea0, self.nbatch_origin)
        xresult = self.x0.index_select(1, best_rind).detach().clone()
        return array(xresult[:, self.inds0]), array(self.flags[self.inds0])

# solving a batch of qps using GPUs
# min 0.5 x^TPx + q^Tx
# s.t. Gl <= Gx <= Gu
#      xl <=  x <= xu
#            Ax == b
class PIQPSolver(QPSolver):
    def __init__(self, eps = 1E-6, reg = 0, verbose = False, **args):
        super(PIQPSolver, self).__init__(eps, reg, verbose, **args)
        self.solver = piqp.SparseSolver()
    
    def solve(self, inds):
        flags, xs = [], []
        nx, nA = self.q.shape[0], self.Al.shape[0]
        for id in inds:
            xli, xui, Ali, Aui, qi = self.xl[:, id], self.xu[:, id], self.Al[:, id], self.Au[:, id], self.q[:, id]
            Pi = self.P
            if hasattr(self, 'L'):
                y = m.addMVar(self.L.shape[0], lb = -GRB.INFINITY, ub = GRB.INFINITY)

                m.setObjective(0.5 * y @ y + qi @ x + self.regularization * x @ x, gp.GRB.MINIMIZE)
                m.addConstr(self.L @ x[:self.L.shape[1]] == y)

                self.solver.update(P,Q, self.L, self.b, self.A, Ali, Aui, xli,xui)
                self.solver.solve()
                if self.solver.status == piqp.PIQP_SOLVED:
                    flags.append(True)
                    xk = np.clip(x.X, xli, xui)
                    xs.append(xk)
                    continue
                
            flags.append(False)
            xs.append(np.zeros(nx))

        xs = np.array(xs).T
        xs = xs.reshape(nx, -1)
        return xs, np.array(flags)