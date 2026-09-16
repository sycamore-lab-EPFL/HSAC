import copy
import math
from time import perf_counter
from typing import Union
import numpy as np
import torch
from scipy.sparse import coo_matrix
import trimesh
floatType = torch.float64
intType = torch.int32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NOBLOCK = 0
FREE = 1
FIXED = 2
NOCONTACT = -1
class StabilityModel():
    def __init__(self, rho = 1, nt = 4,mu=0.5, name = "",Ccp=1e5, qp_solver=None,n_block_max = 50, **args):
        self.rho = rho  # density N/m^3
        self.nt = nt  # num of discretized friction directions
        self.parts = []
        self.n_block_max = n_block_max
        self.part_states = np.full(n_block_max,NOBLOCK)  # states of the parts
        self.contacts = []
        self.name = name
        self.update_volume()
        self.mu = mu
        self.Ccp = Ccp

        # init solver
        self.qp_solver = copy.copy(qp_solver)
        self.init_solver()

        self.lb = args.get('lb', qp_solver.eps_abs)
        self.ub = args.get('ub', np.inf)
        self.max_iter = args.get('max_iter', 1)
        self.verbose = args.get('verbose', True)
    
    def add_part(self,part_mesh,part_id, contacts=[{}],state=FIXED):
        """
        Add a part to the model.
        :param part_mesh: trimesh object representing the part
        :param part_id: unique identifier for the part
        :param contact: dictionary containing contact information
        """
        assert part_id == len(self.parts), "Part ID must match the current number of parts"
        self.part_states[len(self.parts)]=state
        self.parts.append(part_mesh)
        if len(contacts) > 0:
            for contact in contacts:
                self.add_contact(contact)

        self.update_volume()
    def remove_part(self, part_id):
        """
        Remove a part from the model.
        :param part_id: unique identifier for the part to be removed
        """
        if part_id < 0 or part_id >= len(self.parts):
            raise ValueError("Invalid part ID")
        assert part_id == len(self.parts) - 1, "Only the last part can be removed to maintain contiguous part IDs"
        self.part_states[part_id] = NOBLOCK
        # Optionally, you can also remove the part from the parts list
        # self.parts.pop(part_id)
        # Adjust part_states for subsequent parts
        # self.part_states = np.delete(self.part_states, part_id)
        # Update contacts to remove any associated with the removed part
        self.contacts = [c for c in self.contacts if c["partIDA"] != part_id and c["partIDB"] != part_id]
        self.parts.pop(part_id)
        self.update_volume()
        
    def leave_part(self, part_id):
        self.part_states[part_id] = FREE
    def fix_part(self, part_id):
        self.part_states[part_id] = FIXED
    def add_contact(self, contact):
        """
        Add a contact to the model.
        :param contact: dictionary containing contact information
        """
        if not hasattr(self, 'contacts'):
            self.contacts = []
        if "partIDA" not in contact or "partIDB" not in contact:
            raise ValueError("Contact must contain 'partIDA' and 'partIDB' keys")
        if "points" not in contact:
            raise ValueError("Contact must contain 'points' key")
        if "normal" not in contact:
            raise ValueError("Contact must contain 'normal' key")
        self.contacts.append(contact)
    def reset(self):
        """
        Reset the model to its initial state.
        """
        self.part_states[:] =  NOBLOCK
        self.parts = []
        if hasattr(self, 'contacts'):
            del self.contacts
        self.contacts = []
        self.update_volume()
        if hasattr(self, 'qp_solver'):
            self.qp_solver.clear()
    def compute_velocity(self, part_states, λn, λt):
        r = self.Jn.T @ λn + self.Jt.T @ λt + self.g[:, None]
        p = self.p(part_states)
        r = p * r
        v = (self.invM @ r)
        return v
    @property
    def n_part(self):
        return len(self.parts)
    @property
    def nf(self):
        return self.n_part * 6

    @property
    def nλ(self):
        if not hasattr(self, '_nλ'):
            self._nλ = 0
            for contact in self.contacts:
                self._nλ += contact["points"].shape[0]
        return self._nλ

    @property
    def Jn(self):
        if not hasattr(self, '_Jn'):
            JnT = np.zeros((self.nf, self.nλ), dtype=np.float64)
            fn = 0
            for contact in self.contacts:
                partIDA = contact["partIDA"]
                partIDB = contact["partIDB"]
                ms = []
                t = contact["normal"]#[:3, 2]
                for part in [partIDB, partIDA]:
                    ct = self.parts[part].center_mass
                    for id, p in enumerate(contact["points"]):
                        m = np.hstack([t, np.cross(p - ct, t)])
                        JnT[part * 6: part * 6 + 6, fn + id] = m
                    t = -t
                fn += len(contact["points"])

            self._Jn = coo_matrix(JnT.T)
            self._Jn.eliminate_zeros()
        return self._Jn

    @property
    def Jt(self):
        if not hasattr(self, '_Jt'):
            _Jt = np.zeros((self.nλ * self.nt, self.nf), dtype=np.float64)
            for k in range(self.nt):
                fn = 0
                angle = np.pi / self.nt * k * 2
                JtkT = np.zeros((self.nf, self.nλ), dtype=np.float64)
                for contact in self.contacts:
                    partIDA = contact["partIDA"]
                    partIDB = contact["partIDB"]
                    ms = []
                    xaxis = np.cross(contact['normal'],np.array([1,0,0]))
                    if np.linalg.norm(xaxis) < 1e-4:
                        xaxis = np.cross(contact['normal'], np.array([0, 1, 0]))
                    xaxis = xaxis / np.linalg.norm(xaxis)
                    yaxis = np.cross(contact["normal"], xaxis)
                    t = xaxis * np.cos(angle) + yaxis * np.sin(angle)

                    for part in [partIDB, partIDA]:
                        ct = self.parts[part].center_mass
                        for id, p in enumerate(contact["points"]):
                            m = np.hstack([t, np.cross(p - ct, t)])
                            JtkT[part * 6: part * 6 + 6, fn + id] = m
                        t = -t
                    fn += len(contact["points"])
                _Jt[self.nλ * k: self.nλ * (k + 1), :] = JtkT.T
            self._Jt = coo_matrix(_Jt)
            self._Jt.eliminate_zeros()
        return self._Jt

    @property
    def g(self):
        if not hasattr(self, '_g'):
            self._g = np.zeros(self.nf, dtype=np.float64)
            for part in range(self.n_part):
                gf = np.array([0, 0, -self.volumes[part]], dtype=np.float64) * self.rho
                self._g[part * 6: part * 6 + 3] = gf
        return self._g
    @property
    def solver_inputs(self):
        return *self.qp_matrices(), *self.qp_bounds(self.part_states)
    def fix_part(self, part_id):
        """
        Fix a part in the model.
        :param part_id: unique identifier for the part to be fixed
        """
        if not hasattr(self, 'fixed_parts'):
            self.fixed_parts = set()
        self.fixed_parts.add(part_id)
        self.part_states[part_id] = FIXED
    @property
    def E(self):
        if not hasattr(self, '_E'):
            _E = np.zeros((self.nλ, self.nt * self.nλ), dtype=np.float64)
            for i in range(self.nλ):
                for k in range(self.nt):
                    _E[i, self.nλ * k + i] = 1
            self._E = coo_matrix(_E)
        return self._E

    @property
    def M(self):
        if not hasattr(self, '_M'):
            _M = np.zeros((self.nf, self.nf), dtype=np.float64)
            for id, part in enumerate(self.parts):
                I = self.inertias[id] * self.rho
                vM = np.identity(3) * self.volumes[id] * self.rho
                _M[id * 6: id * 6 + 3,
                   id * 6: id * 6 + 3] = vM
                _M[id * 6 + 3: id * 6 + 6,
                   id * 6 + 3: id * 6 + 6] = I
            self._M = coo_matrix(_M)
        return self._M
    @property
    def invML(self):
        if not hasattr(self, '_invML'):
            Mt = torch.tensor(self.M.todense(), dtype=floatType, device='cpu')
            L = torch.linalg.cholesky(Mt)
            invM = torch.cholesky_inverse(L)
            self._invML = torch.linalg.cholesky(invM)
            self._invML = coo_matrix(self._invML)
            self._invML.eliminate_zeros()
            self.changedML = False
        return self._invML

    @property
    def invM(self):
        if not hasattr(self, '_invM'):
            Mt = torch.tensor(self.M.todense(), dtype=floatType, device='cpu')
            L = torch.linalg.cholesky(Mt)
            self._invM = torch.cholesky_inverse(L)
            self._invM = coo_matrix(self._invM.numpy())
            self._invM.eliminate_zeros()
            self.changedM = False
        return self._invM
    
    def update_volume(self):
        if hasattr(self, '_invM'):
            del self._invM
        if hasattr(self, '_invML'):
            del self._invML
        if hasattr(self, '_M'):
            del self._M
        if hasattr(self, '_g'):
            del self._g
        if hasattr(self, '_Jn'):
            del self._Jn
        if hasattr(self, '_Jt'):
            del self._Jt
        if hasattr(self, '_E'):
            del self._E
        if hasattr(self, '_nλ'):
            del self._nλ 
        if hasattr(self,'_Knt'):
            del self._Knt
        if hasattr(self, '_L'):
            del self._L
        self.volumes = [0 for part in self.parts]
        self.inertias = [None for part in self.parts]
        for id, part in enumerate(self.parts):
            self.volumes[id] = part.volume
            self.inertias[id] = part.moment_inertia
            if part.moment_inertia is None:
                print(f"Warning: part.moment_inertia is None for part {id} of type {type(part)}, is_watertight: {getattr(part.is_watertight, None)}")

    def p(self, part_states: np.ndarray):
        if part_states.ndim == 1:
            part_states = part_states.reshape(1, -1)
        assert part_states.ndim == 2

        nbatch = part_states.shape[0]
        _p = np.zeros((self.nf, nbatch), dtype=np.float64)
        for partIDB, state in enumerate(part_states):
            for part in range(self.n_part):
                _p[part * 6: part * 6 + 6, partIDB] = (state[part] == 1)
        return _p

    def c(self, states: np.ndarray):
        if states.ndim == 1:
            states = states.reshape(1, -1)
        assert states.ndim == 2

        nbatch = states.shape[0]
        _c = np.zeros((self.nλ, nbatch), dtype=np.float64)
        for b, state in enumerate(states):
            nf = 0
            for contact in self.contacts:
                partIDA = contact["partIDA"]
                partIDB = contact["partIDB"]
                for p in contact["points"]:
                    if state[partIDA] > NOBLOCK and state[partIDB] > NOBLOCK and (state[partIDA] < FIXED or state[partIDB] < FIXED):
                        _c[nf, b] = 1
                    else:
                        _c[nf, b] = 0
                    nf = nf + 1
        return _c 
    def init_solver(self):
        self.qp_solver.clear()
        #L, A, eq = self.qp_matrices()
        self.qp_solver.update_problem(*self.solver_inputs)
    @property
    def nx(self):
        return (1 + self.nt) * self.nλ + self.nf
    @property
    def nA(self):
        return self.nf + self.nλ
    def qp_matrices(self):
        pass
    def qp_bounds(self, part_states):
        pass
    def simulate(self, part_states: Union[np.ndarray,None] = None):
        if part_states is None:
            part_states = self.part_states.reshape(1, -1)
        else:
            if hasattr(self.qp_solver, 'callback'):
                self.qp_solver.callback.set_part_states(part_states)
        start = perf_counter()
        nbatch = part_states.shape[0]
        inds = np.arange(nbatch)
        flags = np.zeros(nbatch)
        λn_star = np.zeros((self.nλ, nbatch), dtype=np.float64)
        λt_star = np.zeros((self.nλ * self.nt, nbatch), dtype=np.float64)
        #q, xl, xu, Al, Au = self.qp_bounds(part_states)
        self.qp_solver.update_problem(*self.solver_inputs)
        for self.iter in range(self.max_iter):
            xs, flag_terminate = self.qp_solver.solve(inds)
            λn_star[:, inds] = xs[:self.nλ, :]
            λt_star[:, inds] = xs[self.nλ: self.nλ * (self.nt + 1), :]
            flag_stable = self.verify_stability(part_states[inds, :], λn_star[:, inds], λt_star[:, inds] , qtol=self.lb)
            flag_consider = self.verify_stability(part_states[inds, :], λn_star[:, inds] , λt_star[:, inds] , qtol=self.ub)
            flag_nonconsider = np.logical_and(flag_stable, flag_terminate)
            flag_next = np.logical_and(flag_consider, np.logical_not(flag_nonconsider))
            flags[inds[flag_stable]] = True
            if np.sum(flag_next) == 0:
                break
            inds = inds[flag_next]

        end = perf_counter()
        if self.verbose:
            #print(f"time:\t {(end - start) / nbatch},\t nbatch\t {nbatch}")
            #print(f"solving time:\t {(end - start)}")
            self.verify_stability(part_states, λn_star, λt_star, qtol=self.lb, verbose = False)
        return λn_star, λt_star, flags.astype(np.bool_)

    def velocity(self, part_states, λn, λt):
        if λn.ndim == 1:
            λn = λn.reshape(-1, 1)
        if λt.ndim == 1:
            λt = λt.reshape(-1, 1)
        r = self.Jn.T @ λn + self.Jt.T @ λt + self.g[:, None]
        ps = self.p(part_states)
        r = r * ps
        vs = self.invM @ r
        return vs

    def verify_stability(self, part_states, λn, λt, qtol, verbose = False):
        vs = self.velocity(part_states, λn, λt)
        vs_inf = np.max(np.abs(vs), axis=0) #inf norm
        if verbose:
            print("vs_inf:\t", vs_inf)
        return vs_inf < qtol
class BatchStabilityModel():
    def __init__(self,n_batch=1,n_contact_max=512, *args, **kwargs):
        self.n_batch = n_batch
        self.n_block_max = kwargs.get('n_block_max', 50)
        self.max_contact = n_contact_max
        self.rho = kwargs.get('rho', 1)  # density N/m^3
        self.nt = kwargs.get('nt', 4)  # num of discretized friction directions
        self.mu = kwargs.get('mu', 0.5)
        self.Ccp = kwargs.get('Ccp', 1e5)
        self.name = kwargs.get('name', "BatchStabilityModel")
        self.qp_solver = copy.copy(kwargs.get('qp_solver', None))
        if self.qp_solver is not None:
            self.qp_solver.n_batch = n_batch
        self.part_states = np.full((n_batch, self.n_block_max), NOBLOCK)  # states of the parts
        self.contacts_n = np.zeros((n_batch, n_contact_max,3), dtype=np.float64)
        self.contacts_p = np.zeros((n_batch, n_contact_max,3), dtype=np.float64)
        self.contacts_partIDA = np.full((n_batch, n_contact_max),NOCONTACT, dtype=int)
        self.contacts_partIDB = np.full((n_batch, n_contact_max),NOCONTACT, dtype=int)
        self.contact_counts = np.zeros(n_batch, dtype=int)
        self.parts = np.zeros((n_batch, self.n_block_max), dtype=object)
        self.volumes = np.zeros((n_batch, self.n_block_max), dtype=np.float64)
        self.inertias = np.zeros((n_batch, self.n_block_max, 3, 3), dtype=np.float64)
        self.cm = np.zeros((n_batch, self.n_block_max, 3), dtype=np.float64)
    def add_part(self,batchid,part_mesh,part_id, contacts=[{}],state=FIXED):
        self.part_states[batchid, part_id]=state
        self.parts[batchid, part_id]=part_mesh
        if len(contacts) > 0:
            for contact in contacts:
                for c in contact:
                    self.add_contact(c)
        self.volumes[batchid, part_id] = np.array(
            [part.volume if hasattr(part, 'volume') else 0.0 for part in part_mesh],
            dtype=np.float64)
        self.inertias[batchid, part_id] = np.array(
            [part.moment_inertia if hasattr(part, 'moment_inertia') else np.zeros((3, 3), dtype=np.float64) for part in part_mesh],
            dtype=np.float64).reshape(-1, 3, 3)
        self.cm[batchid, part_id] = np.array(
            [part.center_mass if hasattr(part, 'center_mass') else np.zeros(3, dtype=np.float64) for part in part_mesh],
            dtype=np.float64).reshape(-1, 3)
    def add_contact(self, contact):
        if "batchID" not in contact:
            raise ValueError("Contact must contain 'batchID' key for batch stability model")
        if "partIDA" not in contact or "partIDB" not in contact:
            raise ValueError("Contact must contain 'partIDA' and 'partIDB' keys")
        if "points" not in contact:
            raise ValueError("Contact must contain 'points' key")
        if "normal" not in contact:
            raise ValueError("Contact must contain 'normal' key")
        contact_id = self.contact_counts[contact['batchID']]
        npoints = contact["points"].shape[0]
        if contact_id + npoints > self.contacts_n.shape[1]:
            raise ValueError(f"Exceeded maximum number of contacts for batch {contact['batchID']}")
        
        self.contacts_n[contact['batchID'],contact_id:contact_id + npoints, :] = contact["normal"][None]
        self.contacts_p[contact['batchID'],contact_id:contact_id + npoints, :] = contact["points"]
        self.contacts_partIDA[contact['batchID'],contact_id:contact_id + npoints] = contact["partIDA"]
        self.contacts_partIDB[contact['batchID'],contact_id:contact_id + npoints] = contact["partIDB"]
        self.contact_counts[contact['batchID']] += npoints
    def update_Jn(self,new_contacts):
        #TODO only update the Jn for the new contacts, and keep the old Jn for the unchanged contacts
        pass
    def update_Jt(self,new_contacts):
        #TODO only update the Jt for the new contacts, and keep the old Jt for the unchanged contacts
        pass
    def update_volume(self):
        raise NotImplementedError("BatchStabilityModel does not implement update_volume, as it is expected to be overridden by specific batch stability model implementations")
    def remove_part(self, batchid, part_id):
        """
        Remove a part from the model.
        batchid and part_id may each be an int or a 1-D numpy array of length N.
        """
        batchid = np.atleast_1d(np.asarray(batchid))
        part_id = np.atleast_1d(np.asarray(part_id))

        # Scalar fields: fancy indexing works directly
        self.part_states[batchid, part_id] = NOBLOCK
        self.volumes[batchid, part_id]     = 0.0
        self.inertias[batchid, part_id]    = 0
        self.cm[batchid, part_id]          = 0
        self.parts[batchid, part_id]       = None

        # Contact filtering is per-slot because each slot may retain a different
        # number of contacts — vectorising across slots would require ragged ops.
        for b, p in zip(batchid, part_id):
            nc   = int(self.contact_counts[b])
            keep = ~((self.contacts_partIDA[b, :nc] == p) |
                     (self.contacts_partIDB[b, :nc] == p))
            nk   = int(keep.sum())

            new_p   = self.contacts_p[b,         :nc][keep].copy()
            new_n   = self.contacts_n[b,         :nc][keep].copy()
            new_ida = self.contacts_partIDA[b,   :nc][keep].copy()
            new_idb = self.contacts_partIDB[b,   :nc][keep].copy()

            self.contacts_p[b]         = 0
            self.contacts_n[b]         = 0
            self.contacts_partIDA[b]   = 0
            self.contacts_partIDB[b]   = 0

            self.contacts_p[b,       :nk] = new_p
            self.contacts_n[b,       :nk] = new_n
            self.contacts_partIDA[b, :nk] = new_ida
            self.contacts_partIDB[b, :nk] = new_idb
            self.contact_counts[b]        = nk
    def leave_part(self,  batchid,part_id):
        self.part_states[batchid,part_id] = FREE
    def fix_part(self,  batchid,part_id):
        self.part_states[batchid,part_id] = FIXED
    def reset(self,batchid):
        """
        Reset the model to its initial state.
        """
        self.part_states[batchid] =  NOBLOCK
        self.parts[batchid] = None
        self.volumes[batchid] = 0.0
        self.inertias[batchid] = 0
        self.cm[batchid] = 0

        self.contacts_n[batchid] = 0
        self.contacts_p[batchid] = 0
        self.contacts_partIDA[batchid] = NOCONTACT
        self.contacts_partIDB[batchid] = NOCONTACT
        self.contact_counts[batchid] = 0
        
        if hasattr(self, 'qp_solver'):
            self.qp_solver.clear()
    def compute_velocity(self, part_states, λn, λt):
        r = self.Jn.T @ λn + self.Jt.T @ λt + self.g[:, None]
        p = self.p(part_states)
        r = p * r
        v = (self.invM @ r)
        return v
    @property
    def n_part(self):
        return self.n_block_max*self.n_batch
    @property
    def nf(self):
        return self.n_block_max * 6*self.n_batch

    @property
    def nλ(self):
        return self.max_contact * self.n_batch
    @property
    def Jn(self):
        if not hasattr(self, '_Jn'):
            Jn = np.zeros((self.nλ, self.n_block_max * self.n_batch, 6), dtype=np.float64)

            b2d = np.arange(self.n_batch)[:, None]   # (n_batch, 1) for cm indexing
            valid_mask = self.contacts_partIDA.flatten() != NOCONTACT
            b_per_contact = np.arange(self.n_batch).repeat(self.max_contact)

            dist_to_cmA = self.contacts_p - self.cm[b2d, self.contacts_partIDA]  # (n_batch, max_contact, 3)
            dist_to_cmB = self.contacts_p - self.cm[b2d, self.contacts_partIDB]

            momentA = np.cross(dist_to_cmA, self.contacts_n)
            momentB = np.cross(dist_to_cmB, self.contacts_n)

            gen_fA = np.concatenate([ self.contacts_n,  momentA], axis=-1).reshape(-1, 6)[valid_mask]
            gen_fB = np.concatenate([-self.contacts_n, -momentB], axis=-1).reshape(-1, 6)[valid_mask]

            batchpartIDA = (self.contacts_partIDA.flatten() + b_per_contact * self.n_block_max)[valid_mask]
            batchpartIDB = (self.contacts_partIDB.flatten() + b_per_contact * self.n_block_max)[valid_mask]
            batchcontactID = np.arange(self.n_batch * self.max_contact)[valid_mask]

            Jn[batchcontactID, batchpartIDA] = gen_fA
            Jn[batchcontactID, batchpartIDB] = gen_fB
            Jn = Jn.reshape(self.nλ, self.nf)
            self._Jn = coo_matrix(Jn)
            self._Jn.eliminate_zeros()
        return self._Jn

    @property
    def Jt(self):
        if not hasattr(self, '_Jt'):
            _Jt = np.zeros((self.nλ * self.nt, self.nf), dtype=np.float64)
            angle = np.linspace(0, np.pi * 2, self.nt, endpoint=False)

            b2d = np.arange(self.n_batch)[:, None]   # (n_batch, 1) for cm indexing
            valid_mask = self.contacts_partIDA.flatten() != NOCONTACT
            b_per_contact = np.arange(self.n_batch).repeat(self.max_contact)

            batchpartIDA = (self.contacts_partIDA.flatten() + b_per_contact * self.n_block_max)[valid_mask]
            batchpartIDB = (self.contacts_partIDB.flatten() + b_per_contact * self.n_block_max)[valid_mask]
            batchcontactID = np.arange(self.n_batch * self.max_contact)[valid_mask]

            xaxis = np.cross(self.contacts_n, np.array([1, 0, 0], dtype=np.float64))
            degenerate = np.linalg.norm(xaxis, axis=-1) < 1e-4
            xaxis[degenerate] = np.cross(self.contacts_n[degenerate], np.array([0, 1, 0], dtype=np.float64))
            norms = np.linalg.norm(xaxis, axis=-1, keepdims=True)
            xaxis = xaxis / np.maximum(norms, 1e-10)
            yaxis = np.cross(self.contacts_n, xaxis)   # (n_batch, max_contact, 3)

            dist_to_cmA = self.contacts_p - self.cm[b2d, self.contacts_partIDA]  # (n_batch, max_contact, 3)
            dist_to_cmB = self.contacts_p - self.cm[b2d, self.contacts_partIDB]

            cols_A = (batchpartIDA[:, None] * 6 + np.arange(6)).flatten()
            cols_B = (batchpartIDB[:, None] * 6 + np.arange(6)).flatten()

            for k, a in enumerate(angle):
                t = xaxis * np.cos(a) + yaxis * np.sin(a)   # (n_batch, max_contact, 3)
                t_v  = t.reshape(-1, 3)[valid_mask]          # (n_valid, 3)
                mA_v = np.cross(dist_to_cmA, t).reshape(-1, 3)[valid_mask]
                mB_v = np.cross(dist_to_cmB, t).reshape(-1, 3)[valid_mask]
                gen_fA = np.concatenate([ t_v,  mA_v], axis=-1)  # (n_valid, 6)
                gen_fB = np.concatenate([-t_v, -mB_v], axis=-1)
                rows = (k * self.nλ + batchcontactID)[:, None].repeat(6, axis=1).flatten()
                _Jt[rows, cols_A] = gen_fA.flatten()
                _Jt[rows, cols_B] = gen_fB.flatten()

            self._Jt = coo_matrix(_Jt)
            self._Jt.eliminate_zeros()
        return self._Jt

    @property
    def g(self):
        if not hasattr(self, '_g'):
            self._g = np.zeros(self.nf, dtype=np.float64)
            partidxs = np.arange(self.n_block_max*self.n_batch)*6+2#in the z direction
            partidxs = partidxs[self.part_states.flatten() != NOBLOCK]
            self._g[partidxs] = self.volumes.flatten()[self.part_states.flatten() != NOBLOCK] * self.rho
        return self._g
    @property
    def solver_inputs(self):
        return *self.qp_matrices(), *self.qp_bounds(self.part_states)
    @property
    def E(self):
        if not hasattr(self, '_E'):
            _E = np.zeros((self.nλ, self.nt * self.nλ), dtype=np.float64)
            for i in range(self.nλ):
                for k in range(self.nt):
                    _E[i, self.nλ * k + i] = 1
            self._E = coo_matrix(_E)
        return self._E

    @property
    def M(self):
        if not hasattr(self, '_M'):
            _M = np.eye(self.nf, dtype=np.float64)
            for id, part in enumerate(self.parts):
                if part is None:
                    continue
                batchid, partid = np.unravel_index(id, (self.n_batch, self.n_block_max))
                I = self.inertias[batchid, partid] * self.rho
                vM = np.identity(3) * self.volumes[batchid, partid] * self.rho
                _M[id * 6: id * 6 + 3,
                   id * 6: id * 6 + 3] = vM
                _M[id * 6 + 3: id * 6 + 6,
                   id * 6 + 3: id * 6 + 6] = I
            self._M = coo_matrix(_M)
        return self._M
    @property
    def invML(self):
        raise NotImplementedError("BatchStabilityModel does not implement invML, as no preprocessing is necessary")
        if not hasattr(self, '_invML'):
            Mt = torch.tensor(self.M.todense(), dtype=floatType, device='cpu')
            #L = torch.linalg.cholesky(Mt)
            #invM = torch.cholesky_inverse(L)
            #self._invML = torch.linalg.cholesky(invM)
            #self._invML = coo_matrix(self._invML)
            #self._invML.eliminate_zeros()
            #self.changedML = False
        return self._invML

    @property
    def invM(self):
        if not hasattr(self, '_invM'):
            import scipy.linalg
            block = self.n_block_max * 6
            M_arr = self.M.toarray()
            inv_blocks = [np.linalg.inv(M_arr[b*block:(b+1)*block, b*block:(b+1)*block])
                          for b in range(self.n_batch)]
            self._invM = coo_matrix(scipy.linalg.block_diag(*inv_blocks))
            self._invM.eliminate_zeros()
            self.changedM = False
        return self._invM
    
    def update_volume(self):
        if hasattr(self, '_invM'):
            del self._invM
        if hasattr(self, '_M'):
            del self._M
        if hasattr(self, '_g'):
            del self._g
        if hasattr(self, '_Jn'):
            del self._Jn
        if hasattr(self, '_Jt'):
            del self._Jt
        if hasattr(self, '_E'):
            del self._E
        if hasattr(self, '_nλ'):
            del self._nλ
        if hasattr(self, '_Knt'):
            del self._Knt
        if hasattr(self, '_L'):
            del self._L
        self.volumes = [0 for part in self.parts]
        self.inertias = [None for part in self.parts]
        for id, part in enumerate(self.parts):
            self.volumes[id] = part.volume
            self.inertias[id] = part.moment_inertia
            if part.moment_inertia is None:
                print(f"Warning: part.moment_inertia is None for part {id} of type {type(part)}, is_watertight: {getattr(part.is_watertight, None)}")

    def p(self, part_states: np.ndarray):
        if part_states.ndim == 1:
            part_states = part_states.reshape(1, -1)
        assert part_states.ndim == 2
        nbatch = part_states.shape[0]
        _p = np.zeros((self.nf, nbatch), dtype=np.float64)
        b_idx, part_idx = np.where(part_states == FREE)
        dof_starts = (b_idx * self.n_block_max + part_idx) * 6
        all_dofs = (dof_starts[:, None] + np.arange(6)).flatten()
        _p[all_dofs, np.repeat(b_idx, 6)] = 1.0
        return _p

    def c(self, states: np.ndarray):
        if states.ndim == 1:
            states = states.reshape(1, -1)
        assert states.ndim == 2
        nbatch = states.shape[0]
        _c = np.zeros((self.nλ, nbatch), dtype=np.float64)
        for b in range(nbatch):
            state = states[b]
            n_k = self.contact_counts[b]
            if n_k == 0:
                continue
            pA = self.contacts_partIDA[b, :n_k]
            pB = self.contacts_partIDB[b, :n_k]
            active = ((state[pA] > NOBLOCK) & (state[pB] > NOBLOCK) &
                      ((state[pA] < FIXED) | (state[pB] < FIXED)))
            _c[b * self.max_contact + np.where(active)[0], b] = 1.0
        return _c
    def init_solver(self):
        self.qp_solver.clear()
        #L, A, eq = self.qp_matrices()
        self.qp_solver.update_problem(*self.solver_inputs)
    @property
    def nx(self):
        return (1 + self.nt) * self.nλ + self.nf
    @property
    def nA(self):
        return self.nf + self.nλ
    def qp_matrices(self):
        pass
    def qp_bounds(self, part_states):
        pass
    def simulate(self, part_states: Union[np.ndarray, None] = None):
        # Accept either env_ids (1D int) or part_states (2D float)
        if part_states is None:
            env_ids = np.arange(self.n_batch)
        elif np.asarray(part_states).ndim == 1:
            env_ids = np.asarray(part_states, dtype=int)
        else:
            env_ids = np.arange(part_states.shape[0])

        if hasattr(self.qp_solver, 'callback'):
            self.qp_solver.callback.set_part_states(self.part_states)

        start = perf_counter()
        nλ_b = self.nλ // self.n_batch   # per-env contacts
        λn_star = np.zeros((self.nλ, self.n_batch), dtype=np.float64)
        λt_star = np.zeros((self.nλ * self.nt, self.n_batch), dtype=np.float64)

        self.qp_solver.update_problem(*self.solver_inputs)
        # Solve all n_batch envs as independent small QPs (the main speedup)
        xs, flag_terminate = self.qp_solver.solve(np.arange(self.n_batch))
        # xs: (nx_total, n_batch) in full-batch layout after scatter in solve()
        λn_star[:, :] = xs[:self.nλ, :]
        λt_star[:, :] = xs[self.nλ: self.nλ * (self.nt + 1), :]
        flags_all = self.verify_stability(self.part_states, λn_star, λt_star, qtol=1e-6)

        end = perf_counter()
        return λn_star[:, env_ids], λt_star[:, env_ids], flags_all[env_ids].astype(np.bool_)

    def velocity(self, part_states, λn, λt):
        if λn.ndim == 1:
            λn = λn.reshape(-1, 1)
        if λt.ndim == 1:
            λt = λt.reshape(-1, 1)
        r = self.Jn.T @ λn + self.Jt.T @ λt + self.g[:, None]
        ps = self.p(part_states)
        r = r * ps
        vs = self.invM @ r
        return vs

    def verify_stability(self, part_states, λn, λt, qtol, verbose = False):
        vs = self.velocity(part_states, λn, λt)
        vs_inf = np.max(np.abs(vs), axis=0) #inf norm
        if verbose:
            print("vs_inf:\t", vs_inf)
        return vs_inf < qtol