import numpy as np
import matplotlib.pyplot as plt
def project_vectors(vec,dir):
    perp_dir = np.cross(np.array([[1,0,0],
                                  [0,1,0],
                                  [0,0,1]]),dir[None],axis=-1)
    n = np.linalg.norm(perp_dir,axis=-1)
    idx_coords = np.argsort(n)[1:]
    perp_dir0 = perp_dir[idx_coords[0]]/n[idx_coords[0]]
    perp_dir1 = perp_dir[idx_coords[1]]/n[idx_coords[1]]
    proj_mat = np.vstack([perp_dir0,perp_dir1])
    proj_vec = vec @ proj_mat.T
    return proj_vec,proj_mat
def make_hull(vecs,tol=1e-9):
    """use the Jarvis algorithm to compute the convex hull of a set of 2D points"""
    assert vecs.shape[1]==2,"only works in 2D for now"
    if vecs.shape[0]==1:
        return np.array([0]),np.array([[0,0]])
    pot_first, = np.nonzero((vecs[:,0]-np.min(vecs[:,0]))<tol)
    idx_next = pot_first[np.argmin(vecs[pot_first,1])]
    convex_hull = vecs[idx_next,None]#take the lefmost point
    
    idx_hull = np.array([idx_next],dtype=int)
    last_line  = np.array([0,1])
    dir_side = np.zeros((0,2))#last_line[None]
    while convex_hull.shape[0]==1 or np.linalg.norm(convex_hull[-1]-convex_hull[0])>tol:
        start_p = convex_hull[-1]
        d = vecs-start_p
        normd = np.linalg.norm(d,axis=-1)
        d[normd>tol]= d[normd>tol]/normd[normd>tol,None]
        
        cos = np.einsum("ik,k->i",d,last_line)
        cos[normd<=tol]=-2
        
        if np.any(normd>tol) and np.all(cos[normd>tol]<-1+tol):
            #if we are doing a full circle, we break the angle in two
            pass
            #idx_hull = np.append(idx_hull,idx_next)
            #convex_hull = np.vstack([convex_hull,convex_hull[-1]])
            #dir_side = np.vstack([dir_side,last_line])
        normd[normd<=tol]=1
        #angle = np.arccos(cos)
        idx_next = np.argmax(cos)
        #idx_max, = np.nonzero(cos>cos[idx_max]-tol/normd)
        #idx_next = idx_max[np.argmax(normd[idx_max])]
        idx_hull = np.append(idx_hull,idx_next)
        convex_hull = np.vstack([convex_hull,vecs[idx_next]])
        last_line = d[idx_next]
        dir_side = np.vstack([dir_side,[-last_line[1],last_line[0]]])
    idx_red,dir_side = simplify_hull(vecs[idx_hull[1:]],tol=tol)
    return idx_hull[idx_red+1],dir_side
def isinhull(points,hullcorners,hullsides,tol=1e-5):
    if hullcorners.shape[0]==1:
        return (np.abs(points- hullcorners)<np.abs(tol)).all(1),~(np.abs(points- hullcorners)<np.abs(tol)).all(1)[:,None]
    violated = np.einsum("ik,jk->ij",points,hullsides)>np.einsum("jk,jk->j",hullcorners,hullsides)[None]-tol
    if (np.sum(violated,axis=1)>2).any():
        val = np.einsum("ik,jk->ij",points[np.sum(violated,axis=1)>2],hullsides)-np.einsum("jk,jk->j",hullcorners,hullsides)[None]-tol
        violated[np.sum(violated,axis=1)>2] = val>=val.max(axis=-1)[:,None]
    return ~violated.any(axis=-1),violated
def simplify_hull(hullcorners,tol=1e-5):
    """simplify the hull by removing points that are not needed"""
    #build the sides of the hull (skipping one point)
    if hullcorners.shape[0]==1:
        return np.array([0]),np.array([[0,0]])
    #first try to fit all the points in a segment hull
    idx_hull,new_hullconstrs = build_segment_hull(hullcorners)

    #assert hullcorners.shape[0]!=2,"180 degree angles should be broken in two"
    if hullcorners.shape[0]>2:
        hullside = np.vstack([hullcorners[1,:] - hullcorners[-1,:],
                            hullcorners[2:] - hullcorners[:-2],
                            hullcorners[0,:] - hullcorners[-2,:]
                            ])
        ref = np.vstack([hullcorners[1:],
                        hullcorners[0,:]
                        ])
        refp = np.vstack([hullcorners[-1,:],
                            hullcorners[:-1,:],
                        ])
        hullconstr = np.stack([-hullside[:,1],hullside[:,0]],axis=-1)
        n = np.linalg.norm(hullconstr,axis=-1)
        hullconstr[n>tol] = hullconstr[n>tol]/n[n>tol,None]
        hullconstr[n<=tol] = [0,0]
        val = np.einsum("ik,ik->i",hullcorners,hullconstr)-np.einsum("ik,ik->i",ref,hullconstr)
        to_keep = val>tol
    else:
        to_keep = [False]
    if not np.any(to_keep):
        idx_hull,new_hullconstrs = build_segment_hull(hullcorners)
        test,t = isinhull(hullcorners,hullcorners[idx_hull],new_hullconstrs,tol=-tol)
        assert test.all(),"the two points are not on the hull, something is wrong"
    else:
        #if we have two points that are redondant, we only remove the first one
        to_actual_keep = to_keep | ~np.roll(to_keep,1)
        if np.sum(to_actual_keep)<3:
            #the added constraint may break the hull
            idx_hull,new_hullconstrs = build_segment_hull(hullcorners)
            #idx_hull = np.concatenate([np.nonzero(to_actual_keep)[0],np.arange(hullcorners.shape[0])])[idx_hull_in]
            test,t = isinhull(hullcorners,hullcorners[idx_hull],new_hullconstrs,tol=-tol)
            assert test.all(),"some points are not on the hull, something is wrong"
        else:
            idx_hull = np.nonzero(to_actual_keep)[0]
            new_points = hullcorners[to_actual_keep]
            hullside =  np.vstack([new_points[0,:] - new_points[-1,:],
                                   new_points[1:] - new_points[:-1],
                                ])
            new_hullconstrs = np.stack([-hullside[:,1],hullside[:,0]],axis=-1)
            new_hullconstrs = new_hullconstrs/np.linalg.norm(new_hullconstrs,axis=-1)[:,None]
            """   dir_side = np.array([dir_side[0],
                                [dir_side[0,1],-dir_side[0,0]],
                                dir_side[1],
                                [dir_side[1,1],-dir_side[1,0]],
                                ])"""
            if not np.all(to_actual_keep==to_keep):
                idx_hull_in,new_hullconstrs = simplify_hull(new_points,tol=tol)
                idx_hull = idx_hull[idx_hull_in]
        test,t = isinhull(hullcorners,hullcorners[idx_hull],new_hullconstrs,tol=-tol)
        assert test.all(),"some points are not on the hull, something is wrong"
    return idx_hull,new_hullconstrs
def build_segment_hull(vecs):
    """Works in O(n^2) to build a hull from two points"""
    n = np.linalg.norm(vecs[None]-vecs[:,None],axis=-1)
    min_point,max_point = np.unravel_index(np.argmax(n,axis=None),n.shape)
    dir_side = vecs[max_point]-vecs[min_point]
    assert np.linalg.norm(dir_side)>0,"the two points are too close to each other"
    dir_side = dir_side/np.linalg.norm(dir_side)
    """dir_side = dir_side/np.linalg.norm(dir_side)
    max_point = np.argmax(np.einsum("ik,k->i",vecs,dir_side))
    min_point = np.argmin(np.einsum("ik,k->i",vecs,dir_side))"""
    idx_hull = np.stack([max_point,min_point,min_point,max_point])
                #np.repeat(idx_hull,2)#
    new_hullconstrs = np.array([dir_side,
                                    [dir_side[1],-dir_side[0]],
                            -dir_side,
                            [-dir_side[1],dir_side[0]],
                            ])
    return idx_hull,new_hullconstrs
def closest_point_in_hull(point,hullcorners,tol=1e-5):
    """Find the closest point in the hull to each point"""
    hullcorners = hullcorners-point[None,:]
    d_p = np.linalg.norm(hullcorners,axis=-1)
    S = [np.argmin(d_p,axis=-1)]
    R = np.array([np.sqrt(1+np.square(d_p[S]))])
    w = np.array([1])
    tolin = 1e-10
    iter = 0
    n_iter_max = 100
    while iter< n_iter_max:
        iter += 1
        X = w@hullcorners[S]
        J = np.argmin(hullcorners@X.T)
        if X@hullcorners[J].T > X.T@X -tol*np.square(max(np.linalg.norm(hullcorners[J]),np.linalg.norm(hullcorners[S],axis=-1).max())):
            break
        if np.isin(J,S):
            break
        #r = np.linalg.solve(R.T,1+hullcorners[S]@hullcorners[J].T)
        S = np.append(S,J)
        w = np.append(w,0)
        #rho = np.array(np.sqrt(1+ hullcorners[J].T@hullcorners[J]-r.T@r))[None,None]
        #R = np.hstack([np.vstack([R,np.zeros((R.shape[0]))]),np.vstack([r[:,None],rho])])
        while True:
            A = np.vstack([np.ones(S.shape[0]),hullcorners[S].T])
            b = np.zeros(A.shape[0])
            b[0] = 1
            u = np.linalg.inv(A.T@A)@A.T@b
            #u = np.linalg.lstsq(A,b,rcond=None)[0]
            #u_bar = np.linalg.solve(R.T,np.ones(R.shape[0]))
            #u = np.linalg.solve(R,u_bar)
            v = u/np.sum(u)
            #v = u
            if np.all(v>tolin):
                w = v
                break
            else:
                pos = w-v>tolin
                #theta = min(1,np.max(w[pos]/(w[pos]-v[pos])))
                theta = np.min(w[pos]/(w[pos]-v[pos]),initial=1)
                w = w*(1-theta) + v*theta
                w[w<tolin] = 0
                tokeep = w>tolin
                #tokeep = np.ones(S.shape[0],dtype=bool)
                #tokeep[np.argmin(w)] = False
                S = S[tokeep]
                w = w[tokeep]
                delidx =np.argmin(w)
                #A = np.vstack([np.ones(S.shape[0]),hullcorners[S].T])
                #R = np.linalg.cholesky(A.T@A).T

                """for delidx in np.nonzero(~tokeep)[0]:
                    R = np.delete(R,delidx,axis=1)
                    for i in range(delidx,R.shape[1]):
                        a = R[i,i]
                        b = R[i+1,i]
                        c = np.sqrt(np.square(a)+np.square(b))
                        R[i] = R[i]*a/c + R[i+1]*b/c
                        R[i+1] = -b/c*R[i] + a/c*R[i+1]"""
    if iter== n_iter_max:
        print(f"Warning: closest_point_in_hull did not converge,doubling the tolerance to {tol*5}")
        return closest_point_in_hull(point,hullcorners,tol=tol*5)
    weights = np.zeros(hullcorners.shape[0])
    weights[S] = w
    return X+point,weights
def rotmat_from_6D(vecs):
    """Compute the rotation matrices from their 6D representation"""
    assert vecs.shape[1]==6,"only works in 3D"
    rotmat = np.zeros((vecs.shape[0],3,3))
    a1 = vecs[:,:3]
    a2 = vecs[:,3:]
    a1n = np.linalg.norm(a1,axis=-1)
    a2n = np.linalg.norm(a2,axis=-1)
    
    
    a1[a1n<1e-9]=[1,0,0]
    a1n[a1n<1e-9] = 1
    a1 = a1/a1n[:,None]

    b1 = a1
    b2 = a2 - np.einsum("ik,ik->i",a1,a2)[:,None]*a1
    b2n = np.linalg.norm(b2,axis=-1)
    b2[b2n<1e-9]=[0,1,0]
    b2n[b2n<1e-9] = 1
    b2 = b2/b2n[:,None]
    
    b3 = np.cross(b1,b2)
    rotmat = np.stack([b1,b2,b3],axis=-1)
    return rotmat
def rotmat_from_2D(vecs,tol=1e-9):
    """Compute the rotation matrices along the z axis from their 2D representation"""
    assert vecs.shape[1]==2,"only works in 2D"
    rotmat = np.zeros((vecs.shape[0],3,3))
    n = np.linalg.norm(vecs,axis=-1)
    vecs[n<tol] = [1,0]
    n[n<tol] = 1
    nvec = vecs/n[:,None]
    rotmat[:,:2,0] = nvec
    rotmat[:,0,1] = -nvec[:,1]
    rotmat[:,1,1] = nvec[:,0]
    rotmat[:,2,2] = 1
    return rotmat
def rotmat_to_2D(rotmats,tol=1e-9):
    """Compute the 2D representation of the rotation matrices along the z axis"""
    assert rotmats.shape[1:]==(3,3),"only works in 2D"
    vecs = rotmats[:,:2,0]
    return vecs
def plot_hull(vecs,hullcorners,hullsides,ax=None,load=None,rot_point = None,rot_axis=None):
    if ax is None:
        fig,ax = plt.subplots()
    ax.scatter(vecs[:,0],vecs[:,1])
    ax.scatter(hullcorners[:,0],hullcorners[:,1],color='red')
    hullsides = np.stack([hullsides[:,1],
                          -hullsides[:,0]],axis=-1)
    for i in range(hullsides.shape[0]):
        ax.plot([hullcorners[i,0],hullcorners[(i+1)%hullsides.shape[0],0]],
                [hullcorners[i,1],hullcorners[(i+1)%hullsides.shape[0],1]],color='red')
        ax.plot([hullcorners[i,0]-hullsides[i,0],hullcorners[i,0]+hullsides[i,0]],
                [hullcorners[i,1]-hullsides[i,1],hullcorners[i,1]+hullsides[i,1]],color='blue')
    if load is not None:
        ax.scatter(load[0],load[1],color='green')
    if rot_point is not None:
        ax.scatter(rot_point[0],rot_point[1],color='orange')
        if rot_axis is not None:
            ax.plot([rot_point[0],rot_point[0]+rot_axis[1]],
                    [rot_point[1],rot_point[1]-rot_axis[0]],color='orange')
    ax.set_aspect('equal')
    return fig,ax
if __name__ == "__main__":
    p = np.array([[-0.6196901 ,  0.9339022 ,  0.        ],
                  [-0.61725704,  0.8947882 ,  0.        ],
                    [-0.50993554, -1.227233  ,  0.        ],
                    [-0.51274053, -1.27329642,  0.        ],
                    [-0.55801157, -0.28058535,  0.        ],
                    [-0.55801169, -0.28058392,  0.        ],
                    
                    [-0.61969046,  0.62897437,  0.        ],
                    [-0.61969034,  0.68323272,  0.        ],
                    [-0.61969034,  0.68323272,  0.        ],
                    [-0.51520829, -1.23362309,  0.        ],
                    [-0.51520888, -1.2336126 ,  0.        ],
                    [-0.61725692,  0.89478534,  0.        ]
                    ])
    
    """p = np.array([[0,0.60005],
                  [3,0],
                  [-2,1],
                  [-2,1.00001]])"""
    """p = np.array([[0.,1.0001],
                  [-3,1],
                  [3,1],
                  ])"""
    """p = np.array([[-2.9,0.9],
                  [-3,1],
                  [3,1],])"""
    plt.scatter(p[:,0],p[:,1])
    plt.scatter(0,0)
    plt.axis('equal')
    plt.show(block=False)
    x,w =closest_point_in_hull(np.zeros(p.shape[1]),p,tol=1e-5)
    plt.scatter(p[w>0,0],p[w>0,1])
    plt.plot(p[w>0,0],p[w>0,1],color='red')
    pass