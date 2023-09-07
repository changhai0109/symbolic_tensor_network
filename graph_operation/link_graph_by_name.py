import copy
from .graph_operation_base import GraphOPBase
from symbolic_graph import SymbolicGraph
from tensor import get_tensor_size


class LinkGraphByName(GraphOPBase):
    def __init__(self):
        super(LinkGraphByName, self).__init__(("tensor_name", "op_type"), (), ())

    def _check_same_keys(self, graph1, graph2):
        keys1 = graph1.keys
        keys2 = copy.deepcopy(graph2.keys)
        for key1 in keys1:
            assert key1 in keys2
            keys2.remove(key1)
        assert len(keys2) == 0

    def process(self, graph1, graph2, links):
        self._check_same_keys(graph1, graph2)
        name_tensor_map = dict()
        for tensor in graph1.tensors:
            assert not tensor.tensor_name in name_tensor_map
            name_tensor_map[tensor.tensor_name] = tensor
        for tensor in graph2.tensors:
            assert not tensor.tensor_name in name_tensor_map
            name_tensor_map[tensor.tensor_name] = tensor

        for from_, to_ in links.items():
            assert from_ in name_tensor_map and to_ in name_tensor_map
            from_tensor, to_tensor = name_tensor_map[from_], name_tensor_map[to_]
            assert to_tensor.op_type == "T"
            if "shape" in from_tensor.keys():
                assert "shape" in to_tensor.keys()
                assert get_tensor_size(from_tensor.shape) == get_tensor_size(
                    to_tensor.shape
                )
            if "hidden" in from_tensor.keys():
                assert "hidden" in from_tensor.keys()
                assert get_tensor_size(from_tensor.hidden) == get_tensor_size(
                    to_tensor.hidden
                )
            for key in from_tensor.keys():
                if key == "tensor_name":
                    # id do not change, and its children might reference the tensor via name
                    pass
                else:
                    to_tensor[key] = from_tensor[key]
            # after copying every key, the to_tensor will replace from_tensor in from_graph, then no need of from_tensor
            del name_tensor_map[from_]
        # create empty graph
        merged_graph = SymbolicGraph(None)
        # then load it with keys and tensors
        merged_graph.keys = copy.deepcopy(graph1.keys)
        merged_graph.tensors = list()
        for tensor in name_tensor_map.values():
            merged_graph.tensors.append(tensor)
        return merged_graph
