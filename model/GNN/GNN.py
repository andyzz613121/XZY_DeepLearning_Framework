from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='./data/Cora', name='Cora')
print(dataset)