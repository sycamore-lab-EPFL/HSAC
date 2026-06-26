from .base_graph import BaseGraphConstructor
from .bframegraph import BFrameGraphConstructor
import torch
import numpy as np

import torch_geometric
from torch_geometric.data import HeteroData
import torch_geometric.utils
from ..objectives.cover import circle_encode, corners_encode
device = "cpu"
floatType = torch.float32

class FTBFrameGraphConstructor(BFrameGraphConstructor):
    def __init__(self,keep_covered_areas=False,action_node='none',action_metadata=None,floatType = torch.float32):
        super().__init__(keep_covered_areas = keep_covered_areas,action_node=action_node,action_metadata=action_metadata,floatType=floatType)
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
    def compute_full_graph_batched(self,n_block_t,blocks,contacts,is_ground,is_held,objectives,**kwargs):
        datal = super().compute_full_graph_batched(n_block_t,blocks,contacts,is_ground,is_held,objectives,**kwargs).to_data_list()
        #datal = self.base_data_list(blocks.shape[0])
        for batch in range(blocks.shape[0]):
            data =datal[batch]
            contact = contacts[batch]
            if len(contact)>0:
                data['force'].x = torch.ones((len(contact),1),device=device,dtype=self.floatType)
                data['force'].normal = torch.tensor(np.array([c['normal'] for c in contact]),device=device,dtype=self.floatType)
                data['block','bf','force'].edge_index = torch.hstack([torch.tensor([[c['partIDA'],i] for i,c in enumerate(contact)],dtype=int,device=device).T,
                                                                      torch.tensor([[c['partIDB'],i] for i,c in enumerate(contact)],dtype=int,device=device).T])
                data['block','bf','force'].toward = torch.vstack([torch.ones((len(contact),1),dtype=self.floatType,device=device),
                                                                -torch.ones((len(contact),1),dtype=self.floatType,device=device)])
                data['block','bf','force'].edge_attr = data['block','bf','force'].toward * torch.vstack([data['block'].invtransform[ni[0],:3,:3]@data['force'].normal[ni[1]] for ni in data['block','bf','force'].edge_index.T])
                data['force','rev_bf','block'].edge_index = data['block','bf','force'].edge_index[[1,0],:]
                data['force','rev_bf','block'].edge_attr = -data['block','bf','force'].edge_attr

                data['torque'].global_frame =  torch.tensor(np.stack([pt for c in contact for pt in c['points']]),device=device,dtype=self.floatType)
                data['torque'].x = torch.ones((data['torque'].global_frame.shape[0],1),device=device,dtype=self.floatType)
                partA = torch.tensor([c['partIDA'] for c in contact for _ in range(len(c['points']))],dtype=int,device=device)
                partB = torch.tensor([c['partIDB'] for c in contact for _ in range(len(c['points']))],dtype=int,device=device)
                data['block','bt','torque'].edge_index = torch.vstack([torch.hstack([partA,partB]),
                                                                     torch.arange(data['torque'].x.shape[0],device=device).repeat(2)])
                data['block','bt','torque'].edge_attr = torch.vstack(
                    [data['block'].invtransform[ni[0]]@torch.hstack([data['torque'].global_frame[ni[1]],torch.ones(1,device = device)])
                      for ni in data['block','bt','torque'].edge_index.T])[:,:3]
                
                data['torque','rev_bt','block'].edge_index = data['block','bt','torque'].edge_index[[1,0],:]
                data['torque','rev_bt','block'].edge_attr = -data['block','bt','torque'].edge_attr


                data['force','ft','torque'].edge_index = torch.stack([torch.hstack([torch.full((len(c['points']),),i,device=device) for i,c in enumerate(contact)]),
                                                                     torch.arange(data['torque'].x.shape[0],device=device)],dim=0)
                data['force','ft','torque'].edge_attr = torch.ones((data['force','ft','torque'].edge_index.shape[1],1),dtype=self.floatType,device=device)                             
                data['torque','rev_ft','force'].edge_index = data['force','ft','torque'].edge_index[[1,0],:]
                data['torque','rev_ft','force'].edge_attr = -data['force','ft','torque'].edge_attr

                data['force','is','force'].edge_index = torch.tile(torch.arange(data['force'].x.shape[0],device=device),(2,1))
                data['force','is','force'].edge_attr = torch.ones((data['force'].x.shape[0],1),dtype=self.floatType,device=device)
                data['torque','is','torque'].edge_index = torch.tile(torch.arange(data['torque'].x.shape[0],device=device),(2,1))
                data['torque','is','torque'].edge_attr = torch.ones((data['torque'].x.shape[0],1),dtype=self.floatType,device=device)
            else:
                data['force'].x = torch.ones((0,1),device=device,dtype=self.floatType)
                data['force'].normal = torch.ones((0,3),device=device,dtype=self.floatType)
                data['block','bf','force'].toward = torch.ones((0,1),device=device,dtype=self.floatType)
                data['torque'].x = torch.ones((0,1),device=device,dtype=self.floatType)
                data['torque'].global_frame = torch.ones((0,3),device=device,dtype=self.floatType)
                data['force','is','force'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['force','is','force'].edge_attr = torch.zeros((0,1),dtype=self.floatType,device=device)
                data['torque','is','torque'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['torque','is','torque'].edge_attr = torch.zeros((0,1),dtype=self.floatType,device=device)
                data['block','bf','force'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['block','bf','force'].edge_attr = torch.zeros((0,3),dtype=self.floatType,device=device)
                data['force','rev_bf','block'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['force','rev_bf','block'].edge_attr = torch.zeros((0,3),dtype=self.floatType,device=device)
                data['block','bt','torque'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['block','bt','torque'].edge_attr = torch.zeros((0,3),dtype=self.floatType,device=device)
                data['torque','rev_bt','block'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['torque','rev_bt','block'].edge_attr = torch.zeros((0,3),dtype=self.floatType,device=device)
                data['force','ft','torque'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['force','ft','torque'].edge_attr = torch.zeros((0,1),dtype=self.floatType,device=device)
                data['torque','rev_ft','force'].edge_index = torch.zeros((2,0),dtype=int,device=device)
                data['torque','rev_ft','force'].edge_attr = torch.zeros((0,1),dtype=self.floatType,device=device)
        batch = torch_geometric.data.Batch.from_data_list(datal)
        #batch.validate()
        return batch
    @property
    def metadata(self):
        if self.action_node=='none':
            return (['block', 'cover', 'force', 'torque'], 
                    [('block', 'bb', 'block'),
                    ('cover', 'cc', 'cover'),
                    ('block', 'bc', 'cover'),
                    ('cover', 'rev_bc', 'block'),
                    ('block', 'bf', 'force'),
                    ('force', 'rev_bf', 'block'),
                    ('block', 'bt', 'torque'),
                    ('torque', 'rev_bt', 'block'),
                    ('force', 'ft', 'torque'),
                    ('torque', 'rev_ft', 'force'),
                    ('block', 'is', 'block'),
                    ('cover', 'is', 'cover'),
                    ('force', 'is', 'force'),
                    ('torque','is','torque'),])
        else:
            return (['block', 'cover', 'force', 'torque', 'action_discrete'], 
                    [('block', 'bb', 'block'),
                     ('block', 'bad', 'action_discrete'),
                    ('cover', 'cc', 'cover'),
                    ('block', 'bc', 'cover'),
                    ('cover', 'rev_bc', 'block'),
                    ('block', 'bf', 'force'),
                    ('force', 'rev_bf', 'block'),
                    ('block', 'bt', 'torque'),
                    ('torque', 'rev_bt', 'block'),
                    ('force', 'ft', 'torque'),
                    ('torque', 'rev_ft', 'force'),
                    ('block', 'is', 'block'),
                    ('cover', 'is', 'cover'),
                    ('force', 'is', 'force'),
                    ('torque','is','torque'),
                    ('action_discrete','is','action_discrete')])
    @property
    def label_dim(self):
        return {'block':3,'cover':0,('block', 'bb', 'block'):12,
                 ('cover', 'cc', 'cover'):2,
                 ('block', 'bc', 'cover'):2, ('cover', 'rev_bc', 'block'):2,
                 ('block', 'is', 'block'):0, ('cover', 'is', 'cover'):0}
