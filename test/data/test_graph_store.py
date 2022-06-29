import pytest
import torch
from torch_sparse import SparseTensor

from torch_geometric.data.graph_store import EdgeLayout
from torch_geometric.testing.graph_store import MyGraphStore
from torch_geometric.utils.sort_edge_index import sort_edge_index


def get_edge_index(num_src_nodes, num_dst_nodes, num_edges):
    row = torch.randint(num_src_nodes, (num_edges, ), dtype=torch.long)
    col = torch.randint(num_dst_nodes, (num_edges, ), dtype=torch.long)
    return torch.stack([row, col], dim=0)


def test_graph_store():
    graph_store = MyGraphStore()
    edge_index = torch.LongTensor([[0, 1], [1, 2]])
    adj = SparseTensor(row=edge_index[0], col=edge_index[1])

    def assert_equal_tensor_tuple(expected, actual):
        assert len(expected) == len(actual)
        for i in range(len(expected)):
            assert torch.equal(expected[i], actual[i])

    # We put all three tensor types: COO, CSR, and CSC, and we get them back
    # to confirm that `GraphStore` works as intended.
    coo = adj.coo()[:-1]
    csr = adj.csr()[:-1]
    csc = adj.csc()[-2::-1]  # (row, colptr)

    # Put:
    graph_store['edge', EdgeLayout.COO] = coo
    graph_store['edge', 'csr'] = csr
    graph_store['edge', 'csc'] = csc

    # Get:
    assert_equal_tensor_tuple(coo, graph_store['edge', 'coo'])
    assert_equal_tensor_tuple(csr, graph_store['edge', 'csr'])
    assert_equal_tensor_tuple(csc, graph_store['edge', 'csc'])

    # Get attrs:
    edge_attrs = graph_store.get_all_edge_attrs()
    assert len(edge_attrs) == 3

    with pytest.raises(KeyError):
        _ = graph_store['edge_2', 'coo']


def test_graph_store_conversion():
    graph_store = MyGraphStore()
    edge_index = sort_edge_index(get_edge_index(100, 100, 300), num_nodes=100,
                                 sort_by_row=False)
    num_nodes = (100, 100)
    coo = (edge_index[0], edge_index[1])
    csr = SparseTensor.from_edge_index(edge_index,
                                       sparse_sizes=num_nodes).csr()[:2]
    csc = SparseTensor.from_edge_index(edge_index.flip([0]), sparse_sizes=list(
        reversed(num_nodes))).csr()[-2::-1]

    # Put all edge indices:
    graph_store.put_edge_index(edge_index=coo, edge_type=('1', 'to', '2'),
                               layout='coo', num_nodes=num_nodes,
                               is_sorted=True)

    graph_store.put_edge_index(edge_index=csr, edge_type=('2', 'to', '3'),
                               layout='csr', num_nodes=num_nodes)

    graph_store.put_edge_index(edge_index=csc, edge_type=('3', 'to', '1'),
                               layout='csc', num_nodes=num_nodes)

    def _assert_edge_index_equal(expected: torch.Tensor, actual: torch.Tensor):
        assert torch.equal(sort_edge_index(expected), sort_edge_index(actual))

    # Convert to COO:
    row_dict, col_dict, perm_dict = graph_store.coo()
    for key in row_dict:
        actual = torch.stack((row_dict[key], col_dict[key]))
        _assert_edge_index_equal(actual, edge_index)
        assert perm_dict[key] is None

    # Convert to CSR:
    row_dict, col_dict, perm_dict = graph_store.csr()
    for key in row_dict:
        assert torch.equal(row_dict[key], csr[0])
        assert torch.equal(col_dict[key], csr[1])
        if key == ('1', 'to', '2'):
            assert perm_dict[key] is not None

    # Convert to CSC:
    row_dict, col_dict, perm_dict = graph_store.csc()
    for key in row_dict:
        assert torch.equal(row_dict[key], csc[0])
        assert torch.equal(col_dict[key], csc[1])
        assert perm_dict[key] is None
