from .base_jagged import BaseJaggedConstructor,ISOBJECTIVE_FEATURE,TYPE_FEATURE,ISBLOCK_FEATURE,ISACTION_FEATURE
import torch
import numpy as np

#from ..objectives.cover import circle_encode, corners_encode
from ..objectives.cover import delaunay_encode as corners_encode
device = "cpu"

class NaiveJaggedConstructor(BaseJaggedConstructor):
    def __init__(self,keep_covered_areas=False,action_node='none',action_metadata=None,floatType = torch.float32, embedding_dim = 16, n_nodes = 100):
        super().__init__(keep_covered_areas=keep_covered_areas,action_node=action_node,action_metadata=action_metadata,floatType=floatType, embedding_dim=embedding_dim, n_nodes=n_nodes)
        self._current_graph = None
        assert self.embedding_dim >= 16
    def embed_blocks(self, blocks, contacts, is_ground, is_held, objectives,referencial=None,**kwargs):
        max_blocks = blocks.shape[1]
        embedding = torch.zeros((blocks.shape[0],max_blocks,self.embedding_dim), device=device, dtype=self.floatType)
        embedding[:,:,ISBLOCK_FEATURE]= embedding[:,:,0]= torch.tensor([b is not None for b in blocks.flat]).view(blocks.shape)
        embedding[:,:,TYPE_FEATURE+1]=torch.tensor(is_ground)
        embedding[:,:,TYPE_FEATURE+2]=torch.tensor(is_held)
        embedding[:,:,TYPE_FEATURE+3:TYPE_FEATURE+6]=torch.tensor(np.array([b.center_mass if b is not None else [0,0,0] for b in blocks.flat])).view(blocks.shape+(3,))
        embedding[:,:,TYPE_FEATURE+6:TYPE_FEATURE+15]=torch.tensor(referencial).flatten(2)
        embedding = torch.nested.nested_tensor([emb[emb[:,ISBLOCK_FEATURE].bool()] for emb in embedding],layout= torch.jagged)
        return embedding
    def embed_objectives(self, blocks, contacts, is_ground, is_held, objectives,**kwargs):
        #the objectives are not embedded in the naive constructor, just return zeros
        return torch.zeros((blocks.shape[0],0, self.embedding_dim), device=device, dtype=self.floatType)
    def embed_actions(self, actions, blocks, contacts, is_ground, is_held, objectives,**kwargs):
        #the actions are not embedded in the naive constructor, just return zeros
        return torch.zeros((blocks.shape[0],0, self.embedding_dim), device=device, dtype=self.floatType)
