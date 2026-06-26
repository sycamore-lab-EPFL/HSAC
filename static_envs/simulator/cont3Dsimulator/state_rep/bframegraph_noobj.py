from .base_graph import BaseGraphConstructor
import torch
import numpy as np

import torch_geometric
from torch_geometric.data import HeteroData
import torch_geometric.utils
#from ..objectives.cover import circle_encode, corners_encode
from ..objectives.cover import delaunay_encode as corners_encode
device = "cpu"

class BFrameNoObjGraphConstructor(BaseGraphConstructor):
    def __init__(self,keep_covered_areas=False,action_node='none',action_metadata=None,floatType = torch.float32):
        super().__init__(keep_covered_areas=keep_covered_areas,action_node=action_node,action_metadata=action_metadata,floatType=floatType)
        self._current_graph = None
    def add_to_graph(self,new_block,new_contacts):
        pass
    def current_graph(self,blocks,contacts,is_ground,is_held,objectives):
        if self._current_graph is None:
            self._current_graph = self.compute_full_graph(blocks,contacts,is_ground,is_held,objectives)
        return self._current_graph
    def add_block(self,new_block,new_contacts):
        raise NotImplementedError
    def leave_block(blockID):
        raise NotImplementedError

    def _int(self, array):
        return torch.tensor(array, device=device, dtype=torch.long)

    def _float(self, array):
        return torch.tensor(array, device=device, dtype=self.floatType)

    def _arange(self, n):
        return torch.arange(start = 0, end = n, dtype=torch.long, device=device)
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
            data['block'].centroid = torch.tensor(np.array([b.center_mass for b in blocks[batch,bid]]),device=device,dtype=self.floatType)
            data['block'].transform = torch.tensor(np.array([np.block([[referencial[batch,i],blocks[batch,i].center_mass[:,None]],[0,0,0,1]
                                                                       ]) for i in bid]),device=device,dtype=self.floatType)
            data['block'].invtransform = torch.linalg.inv(data['block'].transform)
            n_block = data['block'].x.shape[0]
            bb_src = torch.arange(n_block, device=device).repeat_interleave(n_block)
            bb_dst = torch.arange(n_block, device=device).repeat(n_block)
            data['block','bb','block'].edge_index = torch.stack([bb_src, bb_dst], dim=0)
            rel_tf = torch.matmul(
                data['block'].invtransform.index_select(0, bb_src),
                data['block'].transform.index_select(0, bb_dst),
            )
            data['block','bb','block'].edge_attr = rel_tf[:, :3, :].reshape(-1, 12).contiguous()
            if self.action_node=='last':
                nad = np.prod(self.action_metadata['discrete'])
                data['action_discrete'].x = torch.eye(nad,device=device,dtype=self.floatType)
                data['action_discrete'].part_id = torch.arange(nad,device=device,dtype=int)
                data['block','bad','action_discrete'].edge_index = torch.vstack([torch.full((nad,),data['block'].x.shape[0]-1,device=device,dtype=int), torch.arange(nad,dtype=int,device=device)])
                data['block','bad','action_discrete'].edge_attr = torch.ones((nad,1),dtype=self.floatType,device=device)

                data['action_continuous'].x = torch.zeros((1,self.action_metadata['continuous']),device=device,dtype=self.floatType)
                data['uaction_continuous'].x = torch.zeros((1,self.action_metadata['continuous']),device=device,dtype=self.floatType)

                data['actiond_long'].x = torch.zeros((1),device=device,dtype=int)

                data['block','bac','action_continuous'].edge_index = torch.zeros((2,1),dtype=int,device=device)
                data['block','bac','action_continuous'].edge_attr = torch.ones((1,1),dtype=self.floatType,device=device)

                data['action_discrete','aa','action_continuous'].edge_index = torch.full((2,1),-1,dtype=int,device=device)
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

                data['action_discrete','aa','action_continuous'].edge_index = torch.full((2,1),-1,dtype=int,device=device)
                data['action_discrete','aa','action_continuous'].edge_attr = torch.ones((1,1),dtype=self.floatType,device=device)

                data['action_discrete','is','action_discrete'].edge_index = torch.tile(torch.arange(data['action_discrete'].x.shape[0],device=device),(2,1))          
                data['action_discrete','is','action_discrete'].edge_attr = data['action_discrete'].x
                data['action_continuous','is','action_continuous'].edge_index = torch.tile(torch.arange(data['action_continuous'].x.shape[0],device=device),(2,1))          
                data['action_continuous','is','action_continuous'].edge_attr = torch.ones((data['action_continuous'].x.shape[0],self.action_metadata['continuous']),dtype=self.floatType,device=device)
            elif self.action_node=='none':
                #magic number
                data['action_discrete'].x = torch.zeros((1,1),device=device,dtype=self.floatType)
                data['action_continuous'].x = torch.zeros((1,self.action_metadata['continuous']),device=device,dtype=self.floatType)
                data['actiond_long'].x = torch.zeros((1),device=device,dtype=int)
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
            data['block','is','block'].edge_index = torch.tile(torch.arange(data['block'].x.shape[0],device=device),(2,1))
            data['block','is','block'].edge_attr = torch.ones((data['block'].x.shape[0],1),dtype=self.floatType,device=device)
        return torch_geometric.data.Batch.from_data_list(datal).contiguous()
    @property
    def metadata(self):
        if self.action_node!='none':
            return (['block', 'action_discrete'], 
                    [('block', 'bb', 'block'),
                     ('block', 'bad', 'action_discrete'),
                     #('action_discrete', 'aa', 'action_continuous'),
                     ('block', 'is', 'block'),
                     ('action_discrete','is','action_discrete'),])
        return (['block'], 
                [('block', 'bb', 'block'),
                 ('block', 'is', 'block'),])
    @property
    def label_dim(self):
        return {'block':3,('block', 'bb', 'block'):12,
                 ('block', 'is', 'block'):0}
