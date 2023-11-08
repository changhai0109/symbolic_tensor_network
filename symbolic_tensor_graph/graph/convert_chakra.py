import copy
import sympy as sp
from ..tensor import Tensor
from .graph import HybridGraph
from .coll_comm_matcher import CommunicationMatcher
from ..chakra.node import Node


class ConvertChakra:
    @classmethod
    def _insert_comp(cls, tensor, symbol_map_value, nodes_this_tensor):
        tensor_ops = Tensor.eval_expr(tensor.ops, symbol_map_value)
        ## TODO: change implementation
        ## some tensors with identical op will translate no comp/comms, and leading issues when connecting tensor's nodes as the connection cannot skip the empty tensor.
        ## to avoid this, new insert zero size comp as place holder, and remove them later.
        ## should be fixed
        # if tensor.op > 0:
        if True:
            comp_node = Node()
            comp_node.node_type = Node.NodeType.COMP_NODE
            comp_node.name = f"{tensor.id}_COMP"
            comp_node.num_ops = tensor_ops
            y_tensor_size = Tensor.eval_expr(
                Tensor.eval_size(tensor.y_shape), symbol_map_value
            )
            x1_tensor_size = 0
            if tensor.x1_shape is not None:
                x1_tensor_size = Tensor.eval_expr(
                    Tensor.eval_size(tensor.x1_shape), symbol_map_value
                )
            x2_tensor_size = 0
            if tensor.x2_shape is not None:
                x2_tensor_size = Tensor.eval_expr(
                    Tensor.eval_size(tensor.x2_shape), symbol_map_value
                )
            tensor_size = y_tensor_size + x1_tensor_size + x2_tensor_size
            comp_node.tensor_size = tensor_size
            nodes_this_tensor[HybridGraph.NodeType.COMP] = comp_node

    @classmethod
    def _insert_comm_x1(
        cls, tensor, symbol_map_value, parallel_syms, nodes_this_tensor
    ):
        if tensor.x1 is not None:
            matched_comms = CommunicationMatcher.match_comms(
                tensor.x1.y_shape,
                tensor.x1.y_hidden,
                tensor.x1_shape,
                tensor.x1_hidden,
                parallel_syms,
            )

            comm_nodes = list()
            for comm in matched_comms:
                comm_size = Tensor.eval_expr(
                    Tensor.eval_size(tensor.x1.y_shape), symbol_map_value
                )
                x1_comm_node = Node()
                x1_comm_node.node_type = Node.NodeType.COLL_COMM_NODE
                x1_comm_node.name = f"{tensor.id}_X1COMM"
                x1_comm_node.comm_size = comm_size
                x1_comm_node._comm_meta_data = comm
                if comm[0] == CommunicationMatcher.CommType.ALL_REDUCE:
                    x1_comm_node.comm_type = Node.CollectiveType.ALL_REDUCE
                elif comm[0] == CommunicationMatcher.CommType.ALL_GATHER:
                    x1_comm_node.comm_type = Node.CollectiveType.ALL_GATHER
                elif comm[0] == CommunicationMatcher.CommType.ALL_TO_ALL:
                    x1_comm_node.comm_type = Node.CollectiveType.ALL_TO_ALL
                elif comm[0] == CommunicationMatcher.CommType.REDUCE_SCATTER:
                    x1_comm_node.comm_type = Node.CollectiveType.REDUCE_SCATTER
                else:
                    assert False
                if len(comm_nodes) > 0:
                    x1_comm_node.data_deps.append(comm_nodes[-1].id)
                comm_nodes.append(x1_comm_node)
            if HybridGraph.NodeType.COMP in nodes_this_tensor and len(comm_nodes) > 0:
                nodes_this_tensor[HybridGraph.NodeType.COMP].data_deps.append(
                    comm_nodes[-1].id
                )
            for i, comm_node in enumerate(comm_nodes):
                if i == 0:
                    nodes_this_tensor[HybridGraph.NodeType.X1_COMM] = comm_node
                else:
                    nodes_this_tensor[f"{HybridGraph.NodeType.X1_COMM}{i}"] = comm_node

    @classmethod
    def _insert_comm_x2(
        cls, tensor, symbol_map_value, parallel_syms, nodes_this_tensor
    ):
        if tensor.x2 is not None:
            matched_comms = CommunicationMatcher.match_comms(
                tensor.x2.y_shape,
                tensor.x2.y_hidden,
                tensor.x2_shape,
                tensor.x2_hidden,
                parallel_syms,
            )

            comm_nodes = list()
            for comm in matched_comms:
                comm_size = Tensor.eval_expr(
                    Tensor.eval_size(tensor.x2.y_shape), symbol_map_value
                )
                x2_comm_node = Node()
                x2_comm_node.node_type = Node.NodeType.COLL_COMM_NODE
                x2_comm_node.name = f"{tensor.id}_X2_COMM"
                x2_comm_node.comm_size = comm_size
                x2_comm_node._comm_meta_data = comm
                if comm[0] == CommunicationMatcher.CommType.ALL_REDUCE:
                    x2_comm_node.comm_type = Node.CollectiveType.ALL_REDUCE
                elif comm[0] == CommunicationMatcher.CommType.ALL_GATHER:
                    x2_comm_node.comm_type = Node.CollectiveType.ALL_GATHER
                elif comm[0] == CommunicationMatcher.CommType.ALL_TO_ALL:
                    x2_comm_node.comm_type = Node.CollectiveType.ALL_TO_ALL
                elif comm[0] == CommunicationMatcher.CommType.REDUCE_SCATTER:
                    x2_comm_node.comm_type = Node.CollectiveType.REDUCE_SCATTER
                else:
                    assert False
                if len(comm_nodes) > 0:
                    x2_comm_node.data_deps.append(comm_nodes[-1].id)
                comm_nodes.append(x2_comm_node)
            if HybridGraph.NodeType.COMP in nodes_this_tensor and len(comm_nodes) > 0:
                nodes_this_tensor[HybridGraph.NodeType.COMP].data_deps.append(
                    comm_nodes[-1].id
                )
            for i, comm_node in enumerate(comm_nodes):
                if i == 0:
                    nodes_this_tensor[HybridGraph.NodeType.X2_COMM] = comm_node
                else:
                    nodes_this_tensor[f"{HybridGraph.NodeType.X2_COMM}{i}"] = comm_node

    @classmethod
    def _get_output_node(cls, nodes_this_tensor):
        if HybridGraph.NodeType.COMP in nodes_this_tensor:
            return nodes_this_tensor[HybridGraph.NodeType.COMP]
        elif HybridGraph.NodeType.X1_COMM in nodes_this_tensor:
            x1_keys = list()
            for key in nodes_this_tensor.keys():
                if HybridGraph.NodeType.X1_COMM in key:
                    x1_keys.append(key)
            x1_comm_id = -1
            for key in x1_keys:
                key = key.replace(HybridGraph.NodeType.X1_COMM, "")
                if key == "":
                    id_ = 0
                else:
                    id_ = int(key)
                if id_ > x1_comm_id:
                    x1_comm_id = id_
            if x1_comm_id == 0:
                return nodes_this_tensor[HybridGraph.NodeType.X1_COMM]
            else:
                return nodes_this_tensor[f"{HybridGraph.NodeType.X1_COMM}{x1_comm_id}"]
        else:
            assert False

    @classmethod
    def _get_x1_input_node(cls, nodes_this_tensor):
        if HybridGraph.NodeType.X1_COMM in nodes_this_tensor:
            return nodes_this_tensor[HybridGraph.NodeType.X1_COMM]
        elif HybridGraph.NodeType.COMP in nodes_this_tensor:
            return nodes_this_tensor[HybridGraph.NodeType.COMP]
        else:
            assert False

    @classmethod
    def _get_x2_input_node(cls, nodes_this_tensor):
        if HybridGraph.NodeType.X2_COMM in nodes_this_tensor:
            return nodes_this_tensor[HybridGraph.NodeType.X2_COMM]
        elif HybridGraph.NodeType.COMP in nodes_this_tensor:
            return nodes_this_tensor[HybridGraph.NodeType.COMP]
        else:
            assert False

    @classmethod
    def _tensor_to_nodes(cls, tensor, symbol_map_value, parallel_syms):
        nodes_this_tensor = dict()
        cls._insert_comp(tensor, symbol_map_value, nodes_this_tensor)
        cls._insert_comm_x1(tensor, symbol_map_value, parallel_syms, nodes_this_tensor)
        cls._insert_comm_x2(tensor, symbol_map_value, parallel_syms, nodes_this_tensor)
        return nodes_this_tensor

    @classmethod
    def _connect_tensors_node(cls, tensors, tensor_map_nodes):
        for tensor in tensors:
            if not tensor.x1 is None:
                x1_to = cls._get_x1_input_node(tensor_map_nodes[tensor])
                x1_from = cls._get_output_node(tensor_map_nodes[tensor.x1])
                if not x1_from is None:
                    print(tensor)
                    x1_to.data_deps.append(x1_from.id)
            if not tensor.x2 is None:
                x2_to = cls._get_x2_input_node(tensor_map_nodes[tensor])
                x2_from = cls._get_output_node(tensor_map_nodes[tensor.x2])
                if not x2_from is None:
                    x2_to.data_deps.append(x2_from.id)

    @classmethod
    def _sanity_check(cls, tensor_graph, symbol_map_value, parallel_syms):
        for symbol in tensor_graph.get_symbols():
            assert symbol in symbol_map_value
        for parallel_sym in parallel_syms:
            assert parallel_sym in symbol_map_value

    @classmethod
    def _clean_empty_comp(cls, hybrid_graph):
        while True:
            nodes = hybrid_graph.get_nodes()
            assert all(len(node.ctrl_deps) == 0 for node in nodes)
            empty_comp_nodes = list()
            node_parent_to_child_link = hybrid_graph.get_node_parent_to_child_link()
            node_id_map_node = hybrid_graph.get_node_id_map_node()
            node_id_map_tensor = hybrid_graph.get_node_id_map_tensor()
            for node in nodes:
                if node.node_type == Node.NodeType.COMP_NODE:
                    if node.num_ops == 0:
                        empty_comp_nodes.append(node)
            if len(empty_comp_nodes) == 0:
                break
            for empty_comp_node in empty_comp_nodes:
                print(f" ask for {empty_comp_node.id}")
                tensor = node_id_map_tensor[empty_comp_node.id]
                nodes_this_tensor = hybrid_graph.tensor_map_nodes[tensor]
                nodes_this_tensor_keys = copy.copy(list(nodes_this_tensor.keys()))
                for key in nodes_this_tensor_keys:
                    if nodes_this_tensor[key].id == empty_comp_node.id:
                        del nodes_this_tensor[key]
                print(f"cleaned {empty_comp_node.id} {tensor.id}")
                for child_id in node_parent_to_child_link[empty_comp_node.id]:
                    child = node_id_map_node[child_id]
                    child.data_deps.remove(empty_comp_node.id)
                    if len(empty_comp_node.data_deps) == 0:
                        continue
                    elif len(empty_comp_node.data_deps) >= 2:
                        assert False
                    parent_id = empty_comp_node.data_deps[0]
                    child.data_deps.append(parent_id)

    @classmethod
    def apply(cls, tensor_graph, symbol_map_value, parallel_syms):
        cls._sanity_check(tensor_graph, symbol_map_value, parallel_syms)
        tensor_map_nodes = dict()
        for tensor in tensor_graph.tensors:
            nodes_this_tensor = cls._tensor_to_nodes(
                tensor, symbol_map_value, parallel_syms
            )
            tensor_map_nodes[tensor] = nodes_this_tensor
        cls._connect_tensors_node(tensor_graph.tensors, tensor_map_nodes)
        graph = HybridGraph(tensor_graph.tensors, tensor_map_nodes, symbol_map_value)
        cls._clean_empty_comp(graph)
        return graph