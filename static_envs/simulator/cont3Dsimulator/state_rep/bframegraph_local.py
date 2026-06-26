from .bframegraph import BFrameGraphConstructor
from .base_graph import BaseGraphConstructor
import torch
import numpy as np

import torch_geometric
from torch_geometric.data import HeteroData
import torch_geometric.utils
from ..objectives.cover import circle_encode, corners_encode
device = "cpu"
floatType = torch.float32

class BFrameGraphConstructorLocal(BFrameGraphConstructor):
    def __init__(self,radius = 2.7,keep_covered_areas=False,action_node='block',action_metadata=None):
        super().__init__(keep_covered_areas=keep_covered_areas,action_node=action_node,action_metadata=action_metadata)
        self.radius = radius
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
        datab = super().compute_full_graph_batched(n_block_t,blocks,contacts,is_ground,is_held,objectives,covered=covered,referencial=referencial)
        
        datal = []
        for i,data in enumerate(datab.to_data_list()):
            to_keep = torch.norm(data['block','bb','block'].edge_attr[:,3::4],dim=1)<=self.radius
            data['block','bb','block'].edge_index = data['block','bb','block'].edge_index[:,to_keep].contiguous()
            data['block','bb','block'].edge_attr = data['block','bb','block'].edge_attr[to_keep].contiguous()
            datal.append(data)
        return torch_geometric.data.Batch.from_data_list(datal)