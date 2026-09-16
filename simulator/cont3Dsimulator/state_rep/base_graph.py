import torch
import numpy as np

import torch_geometric
from torch_geometric.data import HeteroData
import torch_geometric.utils
from ..objectives.cover import circle_encode, corners_encode
device =  "cpu"


class BaseGraphConstructor:
    def __init__(self,keep_covered_areas=False,action_node='none',action_metadata=None,floatType = torch.float32):
        self._current_graph = None
        self.floatType = floatType
        self.action_node = action_node
        self.keep_covered_areas = keep_covered_areas
        self.action_metadata = action_metadata
        self.max_td_error = 1e6
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
    def base_data_list(self,n_batch):
        data_list = []
        for batch in range(n_batch):
            data = HeteroData()
            #data['value_exp'].x = torch.zeros((1,1),device=device,dtype=floatType)
            data['value_est'].x = torch.zeros((1,1),device=device,dtype=self.floatType)
            data['logprob'].x = torch.zeros((1,1),device=device,dtype=self.floatType)
            data['reward'].x = torch.zeros((1,1),device=device,dtype=self.floatType)
            data['tde'].x = torch.ones((1,1),device=device,dtype=self.floatType)*self.max_td_error
            data['weight'].x = torch.ones((1,1),device=device,dtype=self.floatType)
            data['terminal'].x = torch.zeros((1,1),device=device,dtype=bool)
            data['truncated'].x = torch.ones((1,1),device=device,dtype=bool)
            data['dcr'].x = torch.zeros((1,1),device=device,dtype=self.floatType)
            data['adv'].x = torch.zeros((1,1),device=device,dtype=self.floatType)
            data['rotationvec'].x = torch.zeros((1,2),device=device,dtype=self.floatType)
            data['action_discrete'].x = torch.zeros((1,self.action_metadata['discrete'][0]),device=device,dtype=self.floatType)
            data['action_continuous'].x = torch.zeros((1,4),device=device,dtype=self.floatType)
            data['uaction_continuous'].x = torch.zeros((1,4),device=device,dtype=self.floatType)
            data['actiond_long'].x = torch.zeros((1,1),device=device,dtype=int)
            data['block','bad','action_discrete'].edge_index = torch.zeros((2,1),dtype=int,device=device)
            data['block','bad','action_discrete'].edge_attr = torch.ones((1,1),dtype=self.floatType,device=device)
            data['block','bac','action_continuous'].edge_index = torch.zeros((2,1),dtype=int,device=device)
            data['block','bac','action_continuous'].edge_attr = torch.ones((1,self.action_metadata['continuous']),dtype=self.floatType,device=device)
            data['action_discrete','aa','action_continuous'].edge_index = torch.zeros((2,1),dtype=int,device=device)
            data['action_discrete','aa','action_continuous'].edge_attr = torch.ones((1,1),dtype=self.floatType,device=device)
            #has to be replaced later using the wanted graph  
            data['action_discrete','is','action_discrete'].edge_index = torch.tile(torch.arange(data['action_discrete'].x.shape[0],device=device),(2,1)) 
            data['action_discrete','is','action_discrete'].edge_attr = data['action_discrete'].x 
            data['action_continuous','is','action_continuous'].edge_index = torch.tile(torch.arange(data['action_continuous'].x.shape[0],device=device),(2,1))          
            data['action_continuous','is','action_continuous'].edge_attr = data['action_continuous'].x 
            data_list.append(data)
        return data_list
    def compute_full_graph_batched(self,blocks,contacts,is_ground,is_held,objectives):
        raise NotImplementedError
    @property
    def metadata(self):
        raise NotImplementedError
    @property
    def label_dim(self):
        raise NotImplementedError
