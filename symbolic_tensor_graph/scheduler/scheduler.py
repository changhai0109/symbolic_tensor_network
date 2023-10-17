import copy


class SchedulerBase:
    def __init__(self, eg_nodes, node_runtime=None, num_queue=1, inplace=False):
        if inplace:
            self.nodes = eg_nodes
        else:
            self.nodes = copy.deepcopy(eg_nodes)
        self.node_runtime = node_runtime
        self.num_queue = num_queue

        self.queues = list()
        self.queues_tick = list()
        for _ in range(self.num_queue):
            self.queues.append(list())
            self.queues_tick.append(0)

        self._id_node_map = None
        self._parent_child_map = None

    def issue_node(self, node):
        if self.node_runtime == None:
            delta_tick = 1
        else:
            assert node.id in self.node_runtime
            delta_tick = self.node_runtime[node.id]

        min_queue_tick = 1e90
        min_queue = None
        min_queue_index = -1
        for i, (queue, queue_tick) in enumerate(zip(self.queues, self.queues_tick)):
            if queue_tick < min_queue_tick:
                min_queue = queue
                min_queue_index = i
                min_queue_tick = min(min_queue_tick, queue_tick)
        # print(
        #     f"issue node={node.name} queue={min_queue_index} begin_queue_tick={min_queue_tick} after_queue_tick={min_queue_tick+delta_tick}"
        # )
        min_queue.append(node)
        self.queues_tick[min_queue_index] += delta_tick

    @property
    def id_node_map(self):
        if self._id_node_map is not None:
            if len(self._id_node_map) != len(self.nodes):
                self._id_node_map = None
        if self._id_node_map is None:
            self._id_node_map = dict()
            for node in self.nodes:
                self._id_node_map[node.id] = node
        return self._id_node_map

    @property
    def parent_child_map(self):
        if self._parent_child_map is not None:
            if len(self._parent_child_map) != len(self.nodes):
                self._parent_child_map = None
        if self._parent_child_map == None:
            self._parent_child_map = dict()
            for node in self.nodes:
                self._parent_child_map[node.id] = list()
            for node in self.nodes:
                for parent in node.parent:
                    self._parent_child_map[parent].append(node.id)
        return self._parent_child_map

    def resolve_queue(self):
        raise NotImplementedError()

    def apply(self):
        self.resolve_queue()
        for queue in self.queues:
            for i in range(len(queue) - 1):
                from_ = queue[i]
                to_ = queue[i + 1]
                if not from_.id in to_.parent:
                    to_.parent.append(from_.id)
        return self.nodes
