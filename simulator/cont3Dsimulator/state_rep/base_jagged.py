
import torch
from torch_geometric.data import HeteroData
import torch_geometric.utils
device = "cpu"
ISBLOCK_FEATURE = 0
ISACTION_FEATURE = 1
ISOBJECTIVE_FEATURE = 1
TYPE_FEATURE = 1
class BaseJaggedConstructor:
    def __init__(self,keep_covered_areas=False,
                 action_node='none',
                 action_metadata=None,
                 floatType = torch.float32,
                 embedding_dim = 16,
                 n_nodes = 100):
        self._current_graph = None
        self.floatType = floatType
        self.action_node = action_node
        self.keep_covered_areas = keep_covered_areas
        self.action_metadata = action_metadata
        self.embedding_dim = embedding_dim
        self.n_nodes = n_nodes
    def add_to_graph(self,new_block,new_contacts):
        raise NotImplementedError
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
    def embed_blocks(self, blocks, contacts, is_ground, is_held, objectives,**kwargs):
        raise NotImplementedError
    def embed_objectives(self, blocks, contacts, is_ground, is_held, objectives,**kwargs):
        raise NotImplementedError
    def embed_actions(self, actions, blocks, contacts, is_ground, is_held, objectives,**kwargs):
        raise NotImplementedError
    def compute_full_graph_batched(self,nblocks,blocks,contacts,is_ground,is_held,objectives,referencial = None, **kwargs):
        blocknodes = self.embed_blocks(blocks, contacts, is_ground, is_held, objectives,referencial=referencial,**kwargs)
        objectivenodes = self.embed_objectives(blocks, contacts, is_ground, is_held, objectives,**kwargs)
        data = torch.nested.nested_tensor([ torch.cat([bi,oi],dim=0)  for bi, oi in zip(blocknodes.unbind(), objectivenodes.unbind())], layout= torch.jagged)
        
        return data
    @property
    def metadata(self):
        return self.embedding_dim
    @property
    def label_dim(self):
        raise NotImplementedError
