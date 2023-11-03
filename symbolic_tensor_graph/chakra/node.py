from .backends.chakra_00_1_backend import Chakra001Backend
from .backends.json_backend import JsonBackend


class Node:
    class NodeType:
        COMP_NODE = 1
        COLL_COMM_NODE = 2
        COMM_SEND_NODE = 3
        COMM_RECV_NODE = 4
        MEM_LOAD_NODE = 5
        MEM_STORE_NODE = 6

    class CollectiveType:
        ALL_REDUCE = 1
        REDUCE_SCATTER = 2
        ALL_GATHER = 3
        ALL_TO_ALL = 4

    backend = Chakra001Backend
    node_id = 0

    @staticmethod
    def get_node_id():
        Node.node_id += 1
        return Node.node_id

    def __init__(self, create_invalid=True):
        if not create_invalid:
            raise NotImplementedError()
        self.node_type = 0
        self.name = ""
        self.id = Node.get_node_id()
        self.ctrl_deps = list()
        self.data_deps = list()

    def sanity_check(self):
        if self.node_type == Node.NodeType.COMP_NODE:
            assert hasattr(self, "num_ops")
            assert hasattr(self, "tensor_size")
        elif self.node_type == Node.NodeType.COLL_COMM_NODE:
            assert hasattr(self, "comm_size")
            assert hasattr(self, "comm_type")
        elif self.node_type == Node.NodeType.COMM_SEND_NODE:
            assert hasattr(self, "comm_size")
            assert hasattr(self, "comm_tag")
            assert hasattr(self, "comm_dst")
        elif self.node_type == Node.NodeType.COMM_RECV_NODE:
            assert hasattr(self, "comm_size")
            assert hasattr(self, "comm_tag")
            assert hasattr(self, "comm_src")
        elif (
            self.node_type == Node.NodeType.MEM_LOAD_NODE
            or self.node_type == Node.NodeType.MEM_STORE_NODE
        ):
            assert hasattr(self, "tensor_size")
        else:
            assert False

    def readout(node, backend=None):
        if backend is None:
            backend = Node.backend
        return backend.readout(node)