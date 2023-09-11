from .graph_operation_base import GraphOPBase
from symbolic_graph import SymbolicGraph


class GetParentShape(GraphOPBase):
    def __init__(self, with_hidden=True):
        if with_hidden:
            super(GetParentShape, self).__init__(
                ("tensor_id", "x1", "x2", "shape", "hidden"),
                (),
                ("x1_shape", "x1_hidden", "x2_shape", "x2_hidden"),
            )
        else:
            super(GetParentShape, self).__init__(
                ("tensor_id", "x1", "x2", "shape"), (), ("x1_shape", "x2_shape")
            )
        self.with_hidden = with_hidden

    def process(self, graph: SymbolicGraph):
        id_tensor_map = dict()
        for tensor in graph.tensors:
            assert not tensor.id in id_tensor_map
            id_tensor_map[tensor.id] = tensor
        for tensor in graph.tensors:
            if tensor.x1 is not None:
                tensor["x1_shape"] = id_tensor_map[tensor.x1].shape
            else:
                tensor["x1_shape"] = None
            if tensor.x2 is not None:
                tensor["x2_shape"] = id_tensor_map[tensor.x2].shape
            else:
                tensor["x2_shape"] = None

            if self.with_hidden:
                if tensor.x1 is not None:
                    tensor["x1_hidden"] = id_tensor_map[tensor.x1].hidden
                else:
                    tensor["x1_hidden"] = None
                if tensor.x2 is not None:
                    tensor["x2_hidden"] = id_tensor_map[tensor.x2].hidden
                else:
                    tensor["x2_hidden"] = None
        return graph
