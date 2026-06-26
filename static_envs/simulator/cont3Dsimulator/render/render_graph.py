import polyscope as ps
import numpy as np
import torch
def render_graph(graph):
    graph_group = ps.create_group("graph")
    blocks_group = ps.create_group("graph_blocks")
    forces_group = ps.create_group("graph_forces")
    objectives_group = ps.create_group("graph_objectives")
    action_group = ps.create_group("graph_actions")
    #if graph['block']
    bb = ps.register_curve_network(f"bb",
                                            nodes=graph['block'].centroid.cpu().numpy(),
                                            edges=graph['block','bb','block'].edge_index.cpu().numpy().T,
                                            color=(0, 0, 0))
    bb.add_to_group(graph_group)
    bb.add_to_group(blocks_group)
    blocks = ps.register_point_cloud("blocks", graph['block'].centroid.cpu().numpy(),color=(1,1,1),transparency=1,radius=bb.get_radius())
    colors = np.tile([[0.9,0.9,0.9]],(graph['block'].centroid.shape[0],1))
    colors[graph['block'].x.cpu().numpy()[:,1].astype(bool)]=[1,0.7,0.7]
    colors[graph['block'].x.cpu().numpy()[:,0].astype(bool)]=[0,0,0]
    #    Add the color data to the point cloud and make it visible immediately
    blocks.add_color_quantity("ground", colors, enabled=True)
    blocks.add_to_group(graph_group)
    if len(graph['cover'])>0:
        color_cover = (1,1,0)
        cc = ps.register_curve_network(f"cc",
                                        nodes=graph['cover'].pos.cpu().numpy(),
                                        edges=graph['cover','cc','cover'].edge_index.cpu().numpy().T,
                                        color=color_cover)
        cc.add_to_group(graph_group)
        corners = ps.register_point_cloud("cover_corners", graph['cover'].pos.cpu().numpy(),color=color_cover,transparency=1,radius=bb.get_radius()/2)
        corners.add_to_group(graph_group)
        bc_edges = graph['block','bc','cover'].edge_index.cpu().numpy().T
        bc_edges[:,1]+=graph['block'].centroid.shape[0]
        bc = ps.register_curve_network(f"bc",
                                        nodes=np.vstack([graph['block'].centroid.cpu().numpy(),np.hstack([graph['cover'].pos.cpu().numpy(),np.zeros((graph['cover'].pos.shape[0],1))])]),
                                        edges=bc_edges,
                                        color=(color_cover[0]/2,color_cover[1]/2,color_cover[2]/2),radius=bb.get_radius()/16)
        bc.add_to_group(graph_group)
        corners.add_to_group(objectives_group)
        bc.add_to_group(objectives_group)
        bc.add_to_group(blocks_group)
        cc.add_to_group(objectives_group)
    if len(graph['force'])>0:
        color = (0.1, 0.4, 0.4)
        pos =torch.stack([graph['torque'].global_frame[graph['force','ft','torque'].edge_index[1,graph['force','ft','torque'].edge_index[0,:]==i]].mean(0) for i in range(graph['force'].x.shape[0])])
        dir = graph['force'].normal.cpu().numpy()
        forces = ps.register_point_cloud(f"forces",
                              points=pos.cpu().numpy(),
                              radius=bb.get_radius()/4,
                              color=color)
        forces.add_vector_quantity(f"forces_normal", dir, color=color, vectortype='standard',enabled=True,length = 0.2
                                    )
        bf_edges = graph['block','bf','force'].edge_index.cpu().numpy().T
        bf_edges[:,1]+=graph['block'].centroid.shape[0]
        bf = ps.register_curve_network(f"bf",nodes=np.vstack([graph['block'].centroid.cpu().numpy(),
                                                         pos.cpu().numpy()]),
                                        edges=bf_edges,
                                        color=color,radius=bb.get_radius()/4)
        
        torque_pos = graph['torque'].global_frame.cpu().numpy()
        ft_edges = graph['force','ft','torque'].edge_index.cpu().numpy().T
        ft_edges[:,1]+=graph['force'].x.shape[0]
        ft = ps.register_curve_network(f"ft",nodes=np.vstack([pos.cpu().numpy(),
                                                         torque_pos]),
                                        edges=ft_edges,
                                        color=color,radius=bb.get_radius()/8)
        
        bt_edges = graph['block','bt','torque'].edge_index.cpu().numpy().T
        bt_edges[:,1]+=graph['block'].centroid.shape[0]
        bt = ps.register_curve_network(f"bt",nodes=np.vstack([graph['block'].centroid.cpu().numpy(),
                                                         torque_pos]),
                                        edges=bt_edges,
                                        color=color,radius=bb.get_radius()/8)
        forces.add_to_group(graph_group)
        forces.add_to_group(forces_group)
        bf.add_to_group(graph_group)
        #bf.add_to_group(blocks_group)
        ft.add_to_group(graph_group)
        ft.add_to_group(forces_group)
        bt.add_to_group(graph_group)
        #bt.add_to_group(blocks_group)
        bt.add_to_group(forces_group)
        bf.add_to_group(forces_group)        
    if len(graph['action_discrete'])>0:
        color = (0.8, 0., 0.)
        ad_pos = graph['block'].centroid[graph['block','bad','action_discrete'].edge_index[0]].cpu().numpy()
        add_offset = np.zeros_like(ad_pos)
        add_offset[:,1]+= 0.5
        add_offset[:,2] += np.linspace(-0.4,0.4,ad_pos.shape[0])
        actions = ps.register_point_cloud(f"actions_discrete",
                              points=ad_pos+add_offset,
                              radius=bb.get_radius()/2,
                              color=color)
        actionsb = ps.register_point_cloud(f"actions_attache",
                              points=ad_pos,
                              radius=0,
                              color=color)
        scale = (np.linalg.norm(add_offset,axis=1)-bb.get_radius()/4)/np.linalg.norm(add_offset,axis=1)
        actionsb.add_vector_quantity(f"bad", add_offset*scale[:,None], color=color, vectortype='ambient',enabled=True,radius=bb.get_radius()/4,)

        actions.add_to_group(graph_group)
        actions.add_to_group(action_group)
        actionsb.add_to_group(graph_group)
        actionsb.add_to_group(action_group)
    return graph_group
