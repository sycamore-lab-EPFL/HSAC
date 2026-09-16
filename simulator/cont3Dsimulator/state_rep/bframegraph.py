from .base_graph import BaseGraphConstructor
import torch
import numpy as np

import torch_geometric
from torch_geometric.data import HeteroData
import torch_geometric.utils
#from ..objectives.cover import circle_encode, corners_encode
from ..objectives.cover import delaunay_encode as corners_encode
device = "cpu"

class BFrameGraphConstructor(BaseGraphConstructor):
    def __init__(self,keep_covered_areas=False,action_node='none',action_metadata=None,floatType = torch.float32):
        super().__init__(keep_covered_areas=keep_covered_areas,action_node=action_node,action_metadata=action_metadata,floatType=floatType)
        self._current_graph = None

    def current_graph(self,blocks,contacts,is_ground,is_held,objectives):
         
        return self.compute_full_graph(blocks,contacts,is_ground,is_held,objectives)

    def compute_full_graph_batched(self,n_block_t,blocks,contacts,is_ground,is_held,objectives,covered=None,referencial=None):
        datal = self.base_data_list(blocks.shape[0])
        for batch in range(blocks.shape[0]):
            data =datal[batch]
            bid, = np.nonzero(blocks[batch,:n_block_t[batch]])
            data['block'].x = torch.tensor(np.hstack([is_ground[batch,bid][:,None],
                          is_held[batch,bid][:,None],
                          #np.array([b.vertices.flatten()for b in blocks[batch,bid]]),
                          #np.array([b.center_mass for b in blocks[batch,bid]]),
                          np.array([b.moment_inertia[2,2] for b in blocks[batch,bid]])[:,None]
                          ]),device=device,dtype=self.floatType)
            data['block'].part_id = torch.tensor(bid,dtype=int,device=device)
            data['block'].centroid = torch.tensor(np.array([b.center_mass for b in blocks[batch,bid]]),device=device,dtype=torch.float32)
            data['block'].transform = torch.tensor(np.array([np.block([[referencial[batch,i],blocks[batch,i].center_mass[:,None]],[0,0,0,1]
                                                                       ]) for i in bid]),device=device,dtype=torch.float32)
            data['block'].invtransform = torch.linalg.inv(data['block'].transform)
            n_block = data['block'].x.shape[0]
            bb_src = torch.arange(n_block, device=device).repeat_interleave(n_block)
            bb_dst = torch.arange(n_block, device=device).repeat(n_block)
            data['block','bb','block'].edge_index = torch.stack([bb_src, bb_dst], dim=0)
            rel_tf = torch.matmul(
                data['block'].invtransform.index_select(0, bb_src),
                data['block'].transform.index_select(0, bb_dst),
            )
            data['block','bb','block'].edge_attr = rel_tf[:, :3, :].to(self.floatType).reshape(-1, 12).contiguous()
            if self.action_node=='last':
                nad = self.action_metadata['discrete'][0]
                assert self.action_metadata['discrete'][1]==1
                data['action_discrete'].x = torch.eye(nad,device=device,dtype=self.floatType)
                data['action_discrete'].part_id = torch.arange(nad,device=device,dtype=int)
                data['block','bad','action_discrete'].edge_index = torch.vstack([torch.full((nad,),data['block'].x.shape[0]-1,device=device,dtype=int), torch.arange(nad,dtype=int,device=device)])
                data['block','bad','action_discrete'].edge_attr = torch.ones((nad,1),dtype=self.floatType,device=device)

                data['action_continuous'].x = torch.zeros((1,self.action_metadata['continuous']),device=device,dtype=self.floatType)
                data['uaction_continuous'].x = torch.zeros((1,self.action_metadata['continuous']),device=device,dtype=self.floatType)
                data['actiond_long'].x = torch.zeros((1),device=device,dtype=int)

                data['block','bac','action_continuous'].edge_index = torch.zeros((2,1),dtype=int,device=device)
                data['block','bac','action_continuous'].edge_attr = torch.ones((1,1),dtype=self.floatType,device=device)

                data['action_discrete','aa','action_continuous'].edge_index = torch.zeros((2,1),dtype=int,device=device)
                data['action_discrete','aa','action_continuous'].edge_attr = torch.ones((1,1),dtype=self.floatType,device=device)

                data['action_discrete','is','action_discrete'].edge_index = torch.tile(torch.arange(data['action_discrete'].x.shape[0],device=device),(2,1))          
                data['action_discrete','is','action_discrete'].edge_attr = data['action_discrete'].x
                data['action_continuous','is','action_continuous'].edge_index = torch.tile(torch.arange(data['action_continuous'].x.shape[0],device=device),(2,1))          
                data['action_continuous','is','action_continuous'].edge_attr = torch.ones((data['action_continuous'].x.shape[0],self.action_metadata['continuous']),dtype=self.floatType,device=device)
            elif self.action_node=='all':
                nad = self.action_metadata['discrete'][0]
                nb = data['block'].x.shape[0]
                data['action_discrete'].x = torch.eye(nad,device=device,dtype=self.floatType).repeat((nb,1))
                data['action_discrete'].part_id = torch.arange(nad*nb,device=device,dtype=int)
                data['block','bad','action_discrete'].edge_index = torch.vstack([torch.arange(nb,device = device).repeat(nad,1).flatten(),
                                                                                 torch.arange(nad*nb,dtype=int,device=device)])
                data['block','bad','action_discrete'].edge_attr = torch.ones((nad*nb,1),dtype=self.floatType,device=device)

                data['action_continuous'].x = torch.zeros((1,self.action_metadata['continuous']),device=device,dtype=self.floatType)
                data['uaction_continuous'].x = torch.zeros((1,self.action_metadata['continuous']),device=device,dtype=self.floatType)
                data['actiond_long'].x = torch.zeros((1),device=device,dtype=int)

                data['block','bac','action_continuous'].edge_index = torch.zeros((2,1),dtype=int,device=device)
                data['block','bac','action_continuous'].edge_attr = torch.ones((1,1),dtype=self.floatType,device=device)

                data['action_discrete','aa','action_continuous'].edge_index = torch.zeros((2,1),dtype=int,device=device)
                data['action_discrete','aa','action_continuous'].edge_attr = torch.ones((1,1),dtype=self.floatType,device=device)

                data['action_discrete','is','action_discrete'].edge_index = torch.tile(torch.arange(data['action_discrete'].x.shape[0],device=device),(2,1))          
                data['action_discrete','is','action_discrete'].edge_attr = data['action_discrete'].x
                data['action_continuous','is','action_continuous'].edge_index = torch.tile(torch.arange(data['action_continuous'].x.shape[0],device=device),(2,1))          
                data['action_continuous','is','action_continuous'].edge_attr = torch.ones((data['action_continuous'].x.shape[0],self.action_metadata['continuous']),dtype=self.floatType,device=device)
            elif self.action_node=='none':
                data['action_discrete'].x = torch.zeros((1,1),device=device,dtype=self.floatType)
                data['action_continuous'].x = torch.zeros((1,self.action_metadata['continuous']),device=device,dtype=self.floatType)
                data['actiond_long'].x = torch.zeros((1),device=device,dtype=int)
                #magic number

                data['block','bad','action_discrete'].edge_index = torch.zeros((2,1),dtype=int,device=device)
                data['block','bad','action_discrete'].edge_attr = torch.ones((1,1),dtype=self.floatType,device=device)

                data['block','bac','action_continuous'].edge_index = torch.zeros((2,1),dtype=int,device=device)
                data['block','bac','action_continuous'].edge_attr = torch.ones((1,1),dtype=self.floatType,device=device)

                data['action_discrete','aa','action_continuous'].edge_index = torch.zeros((2,1),dtype=int,device=device)
                data['action_discrete','aa','action_continuous'].edge_attr = torch.ones((1,1),dtype=self.floatType,device=device)
        

                data['action_discrete','is','action_discrete'].edge_index = torch.tile(torch.arange(data['action_discrete'].x.shape[0],device=device),(2,1))          
                data['action_discrete','is','action_discrete'].edge_attr = torch.ones((data['action_discrete'].x.shape[0],1),dtype=self.floatType,device=device)
                data['action_continuous','is','action_continuous'].edge_index = torch.tile(torch.arange(data['action_continuous'].x.shape[0],device=device),(2,1))          
                data['action_continuous','is','action_continuous'].edge_attr = torch.ones((data['action_continuous'].x.shape[0],self.action_metadata['continuous']),dtype=self.floatType,device=device)
            node_obj,edge_obj = corners_encode(objectives[batch])
            if node_obj.shape[0]>0:
                data['cover'].x = torch.zeros((node_obj.shape[0],1),dtype=self.floatType,device=device)
                data['cover','cc','cover'].edge_index = torch.tensor(edge_obj, device=device, dtype=torch.long).T
                node_obj = torch.tensor(node_obj,device=device,dtype=self.floatType)
                data['cover'].pos = node_obj
                # In 2D, relative offsets across cover edges encode the polygon.
                cc_src = data['cover','cc','cover'].edge_index[0]
                cc_dst = data['cover','cc','cover'].edge_index[1]
                data['cover','cc','cover'].edge_attr = (node_obj.index_select(0, cc_src) - node_obj.index_select(0, cc_dst)).contiguous()

                n_cover = data['cover'].x.shape[0]
                bc_src = torch.arange(n_block, device=device).repeat_interleave(n_cover)
                bc_dst = torch.arange(n_cover, device=device).repeat(n_block)
                data['block','bc','cover'].edge_index = torch.stack([bc_src, bc_dst], dim=0)
                inv_rot = data['block'].invtransform.index_select(0, bc_src)[:, :2, :2]
                delta = data['block'].centroid.index_select(0, bc_src)[:, :2] - node_obj.index_select(0, bc_dst)
                data['block','bc','cover'].edge_attr = torch.bmm(inv_rot, delta.unsqueeze(-1)).to(self.floatType).squeeze(-1).contiguous()
                data['cover','rev_bc','block'].edge_index = data['block','bc','cover'].edge_index[[1,0],:]
                data['cover','rev_bc','block'].edge_attr = -data['block','bc','cover'].edge_attr
            else:
                data['cover'].x = torch.zeros((0,1),dtype=self.floatType,device=device)
                data['cover'].pos = torch.zeros((0,2),dtype=self.floatType,device=device)
                data['cover','cc','cover'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['cover','cc','cover'].edge_attr = torch.ones((0,2),dtype=self.floatType,device=device)
                data['block','bc','cover'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['block','bc','cover'].edge_attr = torch.ones((0,2),dtype=self.floatType,device=device)
                data['cover','rev_bc','block'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['cover','rev_bc','block'].edge_attr = torch.ones((0,2),dtype=self.floatType,device=device)
            data['cover','is','cover'].edge_index = torch.tile(torch.arange(data['cover'].x.shape[0],device=device),(2,1))          
            data['cover','is','cover'].edge_attr = torch.ones((data['cover'].x.shape[0],1),dtype=self.floatType,device=device)
            data['block','is','block'].edge_index = torch.tile(torch.arange(data['block'].x.shape[0],device=device),(2,1))
            data['block','is','block'].edge_attr = torch.ones((data['block'].x.shape[0],1),dtype=self.floatType,device=device)
            if self.keep_covered_areas:
                data['covered'].x = torch.zeros((0,1),dtype=self.floatType,device=device)
                data['covered'].pos = torch.zeros((0,2),dtype=self.floatType,device=device)
                data['covered'].aidx = torch.zeros((0,1),dtype=int,device=device)
                data['covered','cc','covered'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['covered','cc','covered'].edge_attr = torch.ones((0,2),dtype=self.floatType,device=device)
                data['covered','cc','covered'].aidx = torch.zeros((0,1),dtype=int,device=device)
                n_prev = 0
                for step, covered_at_step in enumerate(covered[batch]):
                    covered_nodes, covered_edges =  corners_encode(covered_at_step)
                    if len(covered_nodes)>0:
                        data['covered'].x = torch.cat([data['covered'].x,
                                                       torch.zeros((covered_nodes.shape[0],1),dtype=self.floatType,device=device)])
                        data['covered','cc','covered'].edge_index = torch.cat([data['covered','cc','covered'].edge_index,
                                                                               n_prev+torch.tensor(covered_edges,device=device,dtype = int).T],dim=1)
                        covered_nodes = torch.tensor(covered_nodes,device=device,dtype=self.floatType)
                        data['covered'].pos =  torch.cat([data['covered'].pos,covered_nodes])
                        #in 2D having at least 3 points perfectly define our area with only relative distances
                        data['covered','cc','covered'].edge_attr = torch.cat([data['covered','cc','covered'].edge_attr,
                                                                   torch.vstack([covered_nodes[ni[0]]-covered_nodes[ni[1]] for ni in covered_edges])])
                        data['covered'].aidx = torch.cat([data['covered'].aidx,
                                                         torch.full((covered_nodes.shape[0],1),step,device=device,dtype=int)])
                        data['covered','cc','covered'].aidx = torch.cat([data['covered','cc','covered'].aidx,
                                                                         torch.full((len(covered_edges),1),step,device=device,dtype=int)])
                        n_prev += covered_nodes.shape[0]
            

            
        return torch_geometric.data.Batch.from_data_list(datal).contiguous()
    @property
    def metadata(self):
        if self.action_node!='none':
            return (['block', 'cover', 'action_discrete'], 
                    [('block', 'bb', 'block'),
                     ('cover', 'cc', 'cover'),
                     ('block', 'bc', 'cover'), ('cover', 'rev_bc', 'block'),
                     ('block', 'bad', 'action_discrete'),
                     #('action_discrete', 'aa', 'action_continuous'),
                     ('block', 'is', 'block'), ('cover', 'is', 'cover'),
                     ('action_discrete','is','action_discrete'),])
        return (['block', 'cover'], 
                [('block', 'bb', 'block'),
                 ('cover', 'cc', 'cover'),
                 ('block', 'bc', 'cover'), ('cover', 'rev_bc', 'block'),
                 ('block', 'is', 'block'), ('cover', 'is', 'cover')])
    @property
    def label_dim(self):
        return {'block':3,'cover':0,('block', 'bb', 'block'):12,
                 ('cover', 'cc', 'cover'):2,
                 ('block', 'bc', 'cover'):2, ('cover', 'rev_bc', 'block'):2,
                 ('block', 'is', 'block'):0, ('cover', 'is', 'cover'):0}
