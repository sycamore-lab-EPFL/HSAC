import numpy as np
import gurobipy as gp
from gurobipy import GRB
gp.setParam('OutputFlag', 0)
def slide_direction(points, normals, mu, load, n_tangents=4,tol=1e-5):
    """
    Calculate the sliding direction based on contact points, normals, friction coefficient, and load.
    
    Args:
        points (np.ndarray): Contact points of shape (n_contacts, 3).
        normals (np.ndarray): Normals at the contact points of shape (n_contacts, 3).
        mu (float): Friction coefficient.
        application_point (np.ndarray): Point of application of the load vector, shape (3,).
        load (np.ndarray): Load vector of shape (3,).
        
    Returns:
        np.ndarray: Sliding direction vector of shape (3,).
    """
    assert points.shape[0] == normals.shape[0], "Points and normals must have the same number of contacts."
    assert points.shape[1] == 3 and normals.shape[1] == 3, "Points and normals must be 3D vectors."
    A_n = np.vstack([normals])
    nf = normals.shape[0]
    m = gp.Model()
    xn = m.addMVar(nf, lb=0, ub=GRB.INFINITY)
    m.setObjective(((A_n.T@xn+load)@(A_n.T@xn+load)).sum(), gp.GRB.MINIMIZE)
    #print("Frictionless optimization")
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        acceleration_frictionless = A_n.T@xn.X+load
    else:
        print("Optimization failed: Normal forces")
        return np.zeros(3),False
    if np.linalg.norm(acceleration_frictionless) < tol:
        return np.zeros(3),True
    # check if we can slide or not
    txy = np.stack([np.cross(normals, np.array([[1,0,0]]),axis=-1),
                     np.cross(normals, np.array([[0,1,0]]),axis=-1)])
    idx1 = np.argmax(np.linalg.norm(txy,axis=-1),axis=0)
    t1 =txy[idx1,np.arange(idx1.shape[0])]/ np.linalg.norm(txy[idx1,np.arange(idx1.shape[0])],axis=-1,keepdims=True)
    t2 = np.cross(normals, t1, axis=-1)
    t2 = t2 / np.linalg.norm(t2,axis=-1,keepdims=True)
    rot_t = np.linspace(0, 2 * np.pi, n_tangents, endpoint=False)
    A_f = np.vstack([np.cos(rot_t)[:,None,None]*t1[None] +np.sin(rot_t)[:,None,None]*t2[None]]).reshape(-1,3)
    A_f = np.round(A_f,decimals=10)
    nx = A_f.shape[0]

    xf = m.addMVar(A_f.shape[0], lb=0, ub=GRB.INFINITY)
    m.setObjective(((A_f.T@xf+acceleration_frictionless)@(A_f.T@xf+acceleration_frictionless)).sum(), gp.GRB.MINIMIZE)
    A_friction = np.zeros((normals.shape[0],nx), dtype=np.float64)
    idx_friction = np.arange(n_tangents)*nf
    idx_friction =( idx_friction[None] + np.arange(nf)[:,None]).flatten()
    A_friction[np.repeat(np.arange(nf),n_tangents),idx_friction] = 1
    m.addConstr(A_friction @ xf <= mu*xn.X)
    m.addConstr(A_friction @ xf >= 0)
    #print("Friction optimization")
    m.presolve()
    m.Params.NonConvex = 1
    m.Params.PSDTol = 1e-5
    if m.IsMIP:
        print("Model is a MIP, solving as LP")
        m.setParam('Method', 0)  # Use the simplex method for MIP
        m.setParam('TimeLimit', 10)
    if not m.IsQP:
        pass
    try:
        m.optimize()
    except gp.GurobiError as e:
        print("Optimization failed: friction")
        return np.zeros(3),False
    if m.Status == GRB.OPTIMAL:
        acceleration =A_f.T@xf.X+acceleration_frictionless
    else:
        print("Optimization failed: friction")
        return np.zeros(3),False
    return acceleration,True
def rotation_direction(points,normals,load,application_point,tol,n_tangents=4,mu=0.5):
    """
    Calculate the rotation direction based on inertia, contact normals, points, load, and application point.
    
    Args:
        normals (np.ndarray): Normals at the contact points of shape (n_contacts, 3).
        points (np.ndarray): Contact points of shape (n_contacts, 3).
        load (np.ndarray): Load vector of shape (3,).
        application_point (np.ndarray): Point of application of the load vector, shape (3,).
    Returns:
        np.ndarray: Rotation center point vector of shape (3,).
        np.ndarray: Rotation direction vector of shape (3,).
    """
    
    assert points.shape[0] == normals.shape[0], "Points and normals must have the same number of contacts."
    assert points.shape[1] == 3 and normals.shape[1] == 3, "Points and normals must be 3D vectors."
    n_contacts = normals.shape[0]
    points = points - application_point
    
    # check if we can slide or not
    txy = np.stack([np.cross(normals, np.array([[1,0,0]]),axis=-1),
                     np.cross(normals, np.array([[0,1,0]]),axis=-1)])
    idx1 = np.argmax(np.linalg.norm(txy,axis=-1),axis=0)
    t1 =txy[idx1,np.arange(idx1.shape[0])]/ np.linalg.norm(txy[idx1,np.arange(idx1.shape[0])],axis=-1,keepdims=True)
    t2 = np.cross(normals, t1, axis=-1)
    t2 = t2 / np.linalg.norm(t2,axis=-1,keepdims=True)
    rot_t = np.linspace(0, 2 * np.pi, n_tangents, endpoint=False)
    A_f = np.vstack([np.cos(rot_t)[None,:,None]*t1[:,None] +np.sin(rot_t)[None,:,None]*t2[:,None]])
    points_f = np.tile(points, (n_tangents, 1))
    A_points = np.vstack([points, points_f])
    m_sim = gp.Model()
    m_control = gp.Model()
    #m.setParam('NumericFocus',2)
    fi = m_sim.addMVar((n_contacts), lb=0, ub=100)
    ti = m_sim.addMVar((n_contacts,n_tangents), lb=0, ub=100)
    #forces = np.hstack([fi,ti.reshape(-1)])
    vi = m_control.addMVar((n_contacts,3), lb=-100, ub=100)
    vc = m_control.addMVar((3), lb=-100, ub=100)
    nomega = m_control.addVar(-100,100)
    #nvi = m.addMVar(n_contacts, lb=0, ub=100)
    #because we are not sliding: applied_force = load
    #applied_force = m.addMVar(3, lb=-10, ub=10)
    applied_moment = m_sim.addMVar((3), lb=-100, ub=100)
    #m.setObjective(((vc-load)*(vc-load)).sum(), gp.GRB.MINIMIZE)
    #m.setObjective(omega@vc, gp.GRB.MINIMIZE)
    #m.setObjective(((vc-load)*(vc-load)).sum()+(omega@load)*(omega@load), gp.GRB.MINIMIZE)
    #m.setObjective(((vc-load)*(vc-load)).sum(), gp.GRB.MINIMIZE)

    #m.setObjective((applied_moment*applied_moment).sum()+((vc-load)*(vc-load)).sum(), gp.GRB.MINIMIZE)
    m_sim.setObjective((applied_moment*applied_moment).sum(), gp.GRB.MINIMIZE)
    m_sim.addConstr(fi@normals+(ti[:,:,None]*A_f).sum(0).sum(0)==-load, "total_force")
    m_sim.addConstr(mu*fi >= ti.sum(axis=-1), "friction_forces")
    ft_i = fi[:,None]*normals+ (ti[:,:,None]*A_f).sum(1)
    
    m_sim.addConstr((points[:,1]*ft_i[:,2]-points[:,2]*ft_i[:,1]).sum(0)==applied_moment[0])
    m_sim.addConstr((points[:,2]*ft_i[:,0]-points[:,0]*ft_i[:,2]).sum(0)==applied_moment[1])
    m_sim.addConstr((points[:,0]*ft_i[:,1]-points[:,1]*ft_i[:,0]).sum(0)==applied_moment[2])
    #Makes the problem a QCP
    #m.addConstr(vi*fi[:,None] == np.zeros((n_contacts,3)), "force_velocity")


    

    #m.addConstr(vi[:,1]*normals[:,2]-vi[:,2]*normals[:,1]==0)
    #m.addConstr(vi[:,2]*normals[:,0]-vi[:,0]*normals[:,2]==0)
    #m.addConstr(vi[:,0]*normals[:,1]-vi[:,1]*normals[:,0]==0)

    #m.addConstr(vi[1]*normals[2]-vi[2]*normals[1]>=0)
    #m.addConstr(vi[2]*normals[0]-vi[0]*normals[2]>=0)
    #m.addConstr(vi[0]*normals[1]-vi[1]*normals[0]>=0)
    #m.addConstr(omega[1]*vc[2]-omega[2]*vc[1]==0)
    #m.addConstr(omega[2]*vc[0]-omega[0]*vc[2]==0)
    #m.addConstr(omega[0]*vc[1]-omega[1]*vc[0]==0)
    #m.addConstr(vc@load >=0, "center_of_mass")

    #looks good but is wrong
    #m.addConstr(omega@load == 0)
    m_sim.optimize()
    if m_sim.Status == GRB.OPTIMAL or m_sim.Status == GRB.SUBOPTIMAL:
    #This would be the measurement of the force sensors
        omega = applied_moment.X
    else:
        print("Optimization failed: Moment")
        return np.zeros(3), np.zeros(3), 0
    if np.linalg.norm(omega) < tol/100:
        return np.zeros(3), np.zeros(3), 0
    m_control.addConstr((vi*normals).sum(-1) >= 0, "noncolliding")
    #The force control has priority and thus is considered as a constraint
    cross_product(m_control,omega*nomega,points,vi-vc[None])
    m_control.setObjective(((vc-load)*(vc-load)).sum(), gp.GRB.MINIMIZE)
    cpivot = m_control.addConstr(vi[0]==0)
    #find the best pivot point
    objs = np.zeros(n_contacts)
    nOmegas = np.zeros(n_contacts)
    for idxpivot in range(n_contacts):
        m_control.remove(cpivot)
        #m.remove(comega)
        cpivot = m_control.addConstr(vi[idxpivot]==0,'pivot')
        #comega = m.addConstr(omega@normals[idxpivot] == 0)
        #print("rotation optimization")
        m_control.setParam('BarHomogeneous',1)
        m_control.optimize()
        if m_control.Status == GRB.OPTIMAL or m_control.Status == GRB.SUBOPTIMAL:
            #rotation_centeridx, = np.nonzero(np.linalg.norm(vi.X,axis=-1)<tol)
            #if np.linalg.norm(applied_moment.X) > tol and (vc.X/np.linalg.norm(vc.X))@load >0:
                objs[idxpivot] = m_control.ObjVal
                nOmegas[idxpivot] = nomega.X
                #omega.X[1]*points[:,2]-omega.X[2]*points[:,1]+vc.X[0]==vi.X[:,0]
        else:
            print("Optimization failed: FORCE CONTROL")
            objs[idxpivot] = np.inf
            continue #maybe other points are better
            return np.zeros(3),np.zeros(3),0
    #if np.min(nOmegas)*np.linalg.norm(omega) < tol:
        #return np.zeros(3), np.zeros(3), 0
    idx_best = np.argmin(objs)
    rotation_center = points[idx_best]+ application_point 
    return rotation_center, nOmegas[idx_best]*omega,idx_best

def cross_product(model,vec1,vec2,result):
    if len(vec1.shape) ==1:
        vec1 = vec1.reshape(1,-1)
    if len(vec2.shape) ==1:
        vec2 = vec2.reshape(1,-1)
    if len(result.shape) ==1:
        result = result.reshape(1,-1)
    model.addConstr(vec1[:,1]*vec2[:,2]-vec1[:,2]*vec2[:,1]==result[:,0])
    model.addConstr(vec1[:,2]*vec2[:,0]-vec1[:,0]*vec2[:,2]==result[:,1])
    model.addConstr(vec1[:,0]*vec2[:,1]-vec1[:,1]*vec2[:,0]==result[:,2])
if __name__ == "__main__":
    # Example usage
    points = np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0]])
    angle = np.pi / 8
    
    normals = np.array([[np.sin(angle), 0,np.cos(angle)], [np.sin(angle),0,np.cos(angle)], [np.sin(angle), 0, np.cos(angle)]])
    
    mu = 0.5
    load = np.array([0, 0, -1])