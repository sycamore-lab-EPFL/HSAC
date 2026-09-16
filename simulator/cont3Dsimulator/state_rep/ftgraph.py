import torch
import numpy as np

import torch_geometric
from torch_geometric.data import HeteroData
import torch_geometric.utils
from ..objectives.cover import circle_encode, corners_encode
device = "cuda" if torch.cuda.is_available() else "cpu"

class FTGraphConstructor:
    def __init__(self):
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
        return torch.tensor(array, device=device, dtype=floatType)

    def _arange(self, n):
        return torch.arange(start = 0, end = n, dtype=torch.long, device=device)
    def compute_full_graph_batched(self,blocks,contacts,is_ground,is_held,objectives):
        datal = self.base_data_list(blocks.shape[0])
        for batch in range(blocks.shape[0]):
            data = datal[batch]
            bid, = np.nonzero(blocks[batch])
            data['block'].x = torch.tensor(np.hstack([is_ground[batch,bid][:,None],
                          is_held[batch,bid][:,None],
                          #np.array([b.vertices.flatten()for b in blocks[batch,bid]]),
                          np.array([b.center_mass for b in blocks[batch,bid]]),
                          np.array([b.moment_inertia.flatten() for b in blocks[batch,bid]])
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
            if len(contact)>0:
                data['force'].x = torch.tensor(np.array([c['normal'] for c in contact]),device=device,dtype=floatType)
                data['force','fbplus','block'].edge_index = torch.tensor([[i,c['partIDA']] for i,c in enumerate(contact)],dtype=int,device=device).T
                data['force','fbminus','block'].edge_index = torch.tensor([[i,c['partIDB']] for i,c in enumerate(contact)],dtype=int,device=device).T
                if data['force'].x.shape[0]>0:
                    data['block','rev_fbplus','force'].edge_index = data['force','fbplus','block'].edge_index[[1,0],:]
                    data['block','rev_fbminus','force'].edge_index = data['force','fbminus','block'].edge_index[[1,0],:]
                else:
                    data['block','rev_fbplus','force'].edge_index = data['force','fbplus','block'].edge_index
                    data['block','rev_fbminus','force'].edge_index = data['force','fbminus','block'].edge_index
                
                tauA = [pt - blocks[batch,c["partIDA"]].center_mass for c in contact for pt in c['points']]
                tauB = [pt - blocks[batch,c["partIDB"]].center_mass for c in contact for pt in c['points']]
                if len(tauA)>0:
                    data['torque'].x = torch.tensor(np.stack(tauA+tauB),device=device,dtype=floatType)
                else:
                    data['torque'].x = torch.zeros((0,3),device=device,dtype=floatType)
                data['torque','tt','torque'].edge_index = torch.tensor([[i,i+len(tauA)] for i in range(len(tauA))]+
                                                                    [[i+len(tauA),i] for i in range(len(tauA))],dtype=int,device=device).T
                data['torque','tb','block'].edge_index  = torch.vstack([torch.arange(len(tauA)*2,device=device),
                                                                    torch.tensor([c['partIDA'] for i,c in enumerate(contact) for _ in range(len(c['points']))]+
                                                                                    [c['partIDB'] for c in contact for i in range(len(c['points']))],dtype=int,device=device)])
                if data['torque'].x.shape[0]>0:
                    data['block','rev_tb','torque'].edge_index = data['torque','tb','block'].edge_index[[1,0],:]
                else:
                    data['block','rev_tb','torque'].edge_index = data['torque','tb','block'].edge_index
                data['force','ft','torque'].edge_index =  torch.vstack([torch.tile(torch.tensor([i for i,c in enumerate(contact) for j in range(len(c['points']))],dtype=int,device=device),(1,2)),
                                                                        torch.arange(len(tauA)*2,device=device)]
                                                                    )
                data['force','is','force'].edge_index = torch.tile(torch.arange(data['force'].x.shape[0],device=device),(2,1))
                data['torque','is','torque'].edge_index = torch.tile(torch.arange(data['torque'].x.shape[0],device=device),(2,1))
            else:
                data['force'].x = torch.zeros((0,3),device=device,dtype=floatType)
                data['torque'].x = torch.zeros((0,3),device=device,dtype=floatType)
                data['force','fbplus','block'].edge_index = torch.zeros((2,0),device=device,dtype=int)
                data['force','fbminus','block'].edge_index = torch.zeros((2,0),device=device,dtype=int)
                data['block','rev_fbplus','force'].edge_index = torch.zeros((2,0),device=device,dtype=int)
                data['block','rev_fbminus','force'].edge_index = torch.zeros((2,0),device=device,dtype=int)
                data['torque','tt','torque'].edge_index = torch.zeros((2,0),device=device,dtype=int)
                data['torque','tb','block'].edge_index  = torch.zeros((2,0),device=device,dtype=int)
                data['block','rev_tb','torque'].edge_index = torch.zeros((2,0),device=device,dtype=int)
                data['force','ft','torque'].edge_index = torch.zeros((2,0),device=device,dtype=int)
                data['force','is','force'].edge_index = torch.zeros((2,0),device=device,dtype=int)
                data['torque','is','torque'].edge_index = torch.zeros((2,0),device=device,dtype=int)
            data['torque','rev_ft','force'].edge_index = data['force','ft','torque'].edge_index[[1,0],:]
            data['block','is','block'].edge_index = torch.tile(torch.arange(data['block'].x.shape[0],device=device),(2,1))
            
        return torch_geometric.data.Batch.from_data_list(datal)
    @property
    def metadata(self):
        return (['block', 'force', 'torque', 'cover'], 
                [('block', 'bb', 'block'),
                 ('cover', 'cc', 'cover'),
                 ('block', 'bc', 'cover'), ('cover', 'rev_bc', 'block'),
                 ('force', 'fbplus', 'block'), ('force', 'fbminus', 'block'), 
                 ('block', 'rev_fbplus', 'force'), ('block', 'rev_fbminus', 'force'), 
                 ('torque', 'tt', 'torque'),
                 ('torque', 'tb', 'block'), ('block', 'rev_tb', 'torque'), 
                 ('force', 'ft', 'torque'), ('torque', 'rev_ft', 'force'),
                 ('block', 'is', 'block'), ('force', 'is', 'force'), ('torque', 'is', 'torque'), ('cover', 'is', 'cover')])
def draw_pygdata(PyGdata: HeteroData, k = 1, iter = 10000):
    graph = to_networkx(PyGdata, to_undirected=False)

    node_shapes = {}
    for node, attrs in graph.nodes(data=True):
        if attrs["type"] == "part":
            node_shapes[node] = "o"
        elif attrs["type"] == "force":
            node_shapes[node] = "s"
        elif attrs["type"] == "torque":
            node_shapes[node] = "^"

    # Define colors for the edges
    edge_type_colors = {
        ("part", "pfplus", "force"): "#FF7777",
        ("force", "rev_pfplus", "part"): "#FF7777",
    }

    edge_colors = []
    for from_node, to_node, attrs in graph.edges(data=True):
        edge_type = attrs["type"]
        if edge_type in edge_type_colors:
            edge_colors.append(edge_type_colors[edge_type])
        else:
            edge_colors.append("#000000")

    # pos
    pos = nx.spring_layout(graph, k=k, iterations=iter)

    for node, shape in node_shapes.items():
        nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_shape=shape, node_size=100)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, arrows=True)

    plt.show()

def draw_pygdata(PyGdata: HeteroData, k = 1, iter = 10000):
    graph = to_networkx(PyGdata, to_undirected=False)

    node_shapes = {}
    for node, attrs in graph.nodes(data=True):
        if attrs["type"] == "part":
            node_shapes[node] = "o"
        elif attrs["type"] == "force":
            node_shapes[node] = "s"
        elif attrs["type"] == "torque":
            node_shapes[node] = "^"

    # Define colors for the edges
    edge_type_colors = {
        ("part", "pfplus", "force"): "#773434",
        ("force", "rev_pfplus", "part"): "#FF7777",
    }

    edge_colors = []
    for from_node, to_node, attrs in graph.edges(data=True):
        edge_type = attrs["type"]
        if edge_type in edge_type_colors:
            edge_colors.append(edge_type_colors[edge_type])
        else:
            edge_colors.append("#000000")

    # pos
    pos = nx.spring_layout(graph, k=k, iterations=iter, seed = 10)

    for node, shape in node_shapes.items():
        nx.draw_networkx_nodes(graph, pos, nodelist=[node], node_shape=shape, node_size=100)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, arrows=True)

    plt.show()
