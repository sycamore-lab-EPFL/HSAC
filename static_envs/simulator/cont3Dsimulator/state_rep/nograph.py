import torch
import numpy as np

import torch_geometric
from torch_geometric.data import HeteroData
import torch_geometric.utils
from .base_graph import BaseGraphConstructor
from ..objectives.cover import circle_encode, corners_encode
device = "cuda" if torch.cuda.is_available() else "cpu"

class NOGraphConstructor(BaseGraphConstructor):
    def __init__(self):
        super().__init__()
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
    def compute_full_graph_batched(self,n_blocks,blocks,contacts,is_ground,is_held,objectives,**kwargs):
        datal = self.base_data_list(blocks.shape[0])
        for batch in range(blocks.shape[0]):
            data = datal[batch]
            bid, = np.nonzero(blocks[batch,:n_blocks[batch]])
            data['block'].x = torch.tensor(np.hstack([is_ground[batch,bid][:,None],
                          is_held[batch,bid][:,None],
                          np.array([b.vertices.flatten()for b in blocks[batch,bid]]),
                          #np.array([b.center_mass for b in blocks[batch,bid]]),
                          #np.array([b.moment_inertia.flatten() for b in blocks[batch,bid]])
                          ]),device=device,dtype=floatType)
            data['block'].part_id = torch.tensor(bid,dtype=int,device=device)
            data['block','bb','block'].edge_index = torch.tensor([[i,j] for i in range(data['block'].x.shape[0]) for j in range(data['block'].x.shape[0])],dtype=int,device=device).T
            #obj_b = circle_encode(objectives[batch])
            #data['cover'].x = torch.tensor(obj_b,dtype=floatType,device=device)
            """ if len(obj_b)>0:
                data['block','bc','cover'].edge_index = torch.tensor([[i,j] for i in range(data['block'].x.shape[0]) for j in range(obj_b.shape[0])],dtype=int,device=device).T
                data['cover','rev_bc','block'].edge_index = data['block','bc','cover'].edge_index[[1,0],:]
            """
            node_obj,edge_obj = corners_encode(objectives[batch])
            if node_obj.shape[0]>0:
                data['cover'].x = torch.tensor(node_obj,dtype=floatType,device=device)
                data['cover','cc','cover'].edge_index = torch.tensor(edge_obj,device=device,dtype = int).T
                data['block','bc','cover'].edge_index = torch.tensor([[i,j] for i in range(data['block'].x.shape[0]) for j in range(data['cover'].x.shape[0])],dtype=int,device=device).T
                data['cover','rev_bc','block'].edge_index = data['block','bc','cover'].edge_index[[1,0],:]
                data['cover','is','cover'].edge_index = torch.tile(torch.arange(data['cover'].x.shape[0],device=device),(2,1))
            contact = contacts[batch]
            
            data['block','is','block'].edge_index = torch.tile(torch.arange(data['block'].x.shape[0],device=device),(2,1))
        return torch_geometric.data.Batch.from_data_list(datal)
    @property
    def metadata(self):
        return (['block', 'cover'], 
                [('block', 'bb', 'block'),
                 ('cover', 'cc', 'cover'),
                 ('block', 'bc', 'cover'), ('cover', 'rev_bc', 'block'),
                 ('block', 'is', 'block'), ('cover', 'is', 'cover')])
    def typed_graph(self):
        data = HeteroData()

        # part nodes
        data["block"].x = self.part_x.detach().clone()
        #cover nodes
        data['cover'].x = self.obj_x.detach().clone()
        # part part edge
        data["block", "pp", "block"].edge_index = self.part_part_edge.detach().clone().t().contiguous()

        data["block","bc","cover"].edge_index = self.part_cover_ids.contiguous()
        data["cover","rev_bc","block"].edge_index = self.part_cover_ids[[1,0],:].contiguous()
        data["block", "is", "block"].edge_index = torch.stack([torch.arange(data["block"].x.shape[0],device=device),
                                                            torch.arange(data["block"].x.shape[0],device=device)])
       
        data["cover","is","cover"].edge_index = torch.stack([torch.arange(data["cover"].x.shape[0],device=device),
                                                            torch.arange(data["cover"].x.shape[0],device=device)])
        
        data['block'].batch = self.batch_x
        data['cover'].batch = self.batch_o
        return data