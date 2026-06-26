import torch
import numpy as np

import torch_geometric
from torch_geometric.data import HeteroData
import torch_geometric.utils
from .base_graph import BaseGraphConstructor
from ..objectives.cover import circle_encode, corners_encode
device = "cpu"

class NOIGraphConstructor(BaseGraphConstructor):
    def __init__(self,keep_covered_areas=False):
        super().__init__(keep_covered_areas = keep_covered_areas)
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
        return torch.tensor(array, device=device, dtype=floatType)

    def _arange(self, n):
        return torch.arange(start = 0, end = n, dtype=torch.long, device=device)
    def compute_full_graph_batched(self,n_block_t,blocks,contacts,is_ground,is_held,objectives,**kwargs):
        datal = self.base_data_list(blocks.shape[0])
        for batch in range(blocks.shape[0]):
            data =datal[batch]
            bid, = np.nonzero(blocks[batch,:n_block_t[batch]])
            data['block'].x = torch.tensor(np.hstack([is_ground[batch,bid][:,None],
                          is_held[batch,bid][:,None],
                          #np.array([b.vertices.flatten()for b in blocks[batch,bid]]),
                          #np.array([b.center_mass for b in blocks[batch,bid]]),
                          np.array([b.moment_inertia.flatten() for b in blocks[batch,bid]])
                          ]),device=device,dtype=floatType)
            data['block'].part_id = torch.tensor(bid,dtype=int,device=device)
            data['block'].centroid = torch.tensor(np.array([b.center_mass for b in blocks[batch,bid]]),device=device,dtype=floatType)

            data['block','bb','block'].edge_index = torch.tensor([[i,j] for i in range(data['block'].x.shape[0]) for j in range(data['block'].x.shape[0])],dtype=int,device=device).T
            data['block','bb','block'].edge_attr = torch.vstack([data['block'].centroid[ni[0]]-data['block'].centroid[ni[1]] for ni in data['block','bb','block'].edge_index.T])
            #obj_b = circle_encode(objectives[batch])
            #data['cover'].x = torch.tensor(obj_b,dtype=floatType,device=device)
            """ if len(obj_b)>0:
                data['block','bc','cover'].edge_index = torch.tensor([[i,j] for i in range(data['block'].x.shape[0]) for j in range(obj_b.shape[0])],dtype=int,device=device).T
                data['cover','rev_bc','block'].edge_index = data['block','bc','cover'].edge_index[[1,0],:]
            """
            node_obj,edge_obj = corners_encode(objectives[batch])
            if node_obj.shape[0]>0:
                data['cover'].x = torch.zeros((node_obj.shape[0],1),dtype=floatType,device=device)
                data['cover','cc','cover'].edge_index = torch.tensor(edge_obj,device=device,dtype = int).T
                node_obj = torch.tensor(node_obj,device=device,dtype=floatType)
                data['cover'].pos = node_obj
                data['cover','cc','cover'].edge_attr = torch.vstack([node_obj[ni[0]]-node_obj[ni[1]] for ni in data['cover','cc','cover'].edge_index.T])
                
                data['block','bc','cover'].edge_index = torch.tensor([[i,j] for i in range(data['block'].x.shape[0]) for j in range(data['cover'].x.shape[0])],dtype=int,device=device).T
                data['block','bc','cover'].edge_attr = torch.vstack([data['block'].centroid[ni[0],:2]-node_obj[ni[1]] for ni in data['block','bc','cover'].edge_index.T])
                data['cover','rev_bc','block'].edge_index = data['block','bc','cover'].edge_index[[1,0],:]
                data['cover','rev_bc','block'].edge_attr = -data['block','bc','cover'].edge_attr
                data['cover','is','cover'].edge_index = torch.tile(torch.arange(data['cover'].x.shape[0],device=device),(2,1))          
                data['cover','is','cover'].edge_attr = torch.ones((data['cover'].x.shape[0],1),dtype=floatType,device=device)
            else:
                data['cover'].x = torch.zeros((0,1),dtype=floatType,device=device)
                data['cover'].pos = torch.zeros((0,2),dtype=floatType,device=device)
                data['cover','cc','cover'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['cover','cc','cover'].edge_attr = torch.ones((0,2),dtype=floatType,device=device)
                data['block','bc','cover'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['block','bc','cover'].edge_attr = torch.ones((0,2),dtype=floatType,device=device)
                data['cover','rev_bc','block'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['cover','rev_bc','block'].edge_attr = torch.ones((0,2),dtype=floatType,device=device)
                data['cover','is','cover'].edge_index = torch.zeros((2,0),dtype=int,device=device)          
                data['cover','is','cover'].edge_attr = torch.ones((0,1),dtype=floatType,device=device)
            data['block','is','block'].edge_index = torch.tile(torch.arange(data['block'].x.shape[0],device=device),(2,1))
            data['block','is','block'].edge_attr = torch.ones((data['block'].x.shape[0],1),dtype=floatType,device=device)
            #magic number
            data['action_discrete'].x = torch.zeros((1,1),device=device,dtype=floatType)
            #m
            data['action_continuous'].x = torch.zeros((1,4),device=device,dtype=floatType)
            data['actiond_long'].x = torch.zeros((1),device=device,dtype=int)
            data['block','bad','action_discrete'].edge_index = torch.zeros((2,1),dtype=int,device=device)
            data['block','bad','action_discrete'].edge_attr = torch.ones((1,1),dtype=floatType,device=device)

            data['block','bac','action_continuous'].edge_index = torch.zeros((2,1),dtype=int,device=device)
            data['block','bac','action_continuous'].edge_attr = torch.ones((1,1),dtype=floatType,device=device)

            data['action_discrete','aa','action_continuous'].edge_index = torch.zeros((2,1),dtype=int,device=device)
            data['action_discrete','aa','action_continuous'].edge_attr = torch.ones((1,1),dtype=floatType,device=device)
    

            data['action_discrete','is','action_discrete'].edge_index = torch.tile(torch.arange(data['action_discrete'].x.shape[0],device=device),(2,1))          
            data['action_discrete','is','action_discrete'].edge_attr = torch.ones((data['action_discrete'].x.shape[0],1),dtype=floatType,device=device)
            data['action_continuous','is','action_continuous'].edge_index = torch.tile(torch.arange(data['action_continuous'].x.shape[0],device=device),(2,1))          
            data['action_continuous','is','action_continuous'].edge_attr = torch.ones((data['action_continuous'].x.shape[0],4),dtype=floatType,device=device)
        return torch_geometric.data.Batch.from_data_list(datal)
    @property
    def metadata(self):
        return (['block', 'cover'], 
                [('block', 'bb', 'block'),
                 ('cover', 'cc', 'cover'),
                 ('block', 'bc', 'cover'), ('cover', 'rev_bc', 'block'),
                 ('block', 'is', 'block'), ('cover', 'is', 'cover')])
    @property
    def label_dim(self):
        return {'block':2,'cover':0,('block', 'bb', 'block'):3,
                 ('cover', 'cc', 'cover'):2,
                 ('block', 'bc', 'cover'):2, ('cover', 'rev_bc', 'block'):2,
                 ('block', 'is', 'block'):0, ('cover', 'is', 'cover'):0}
