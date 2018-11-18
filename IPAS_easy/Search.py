"""Search Algorithms and functions

Focusing on A* search, with implementations for:
Basic Astar: search on Graph generated from Environment/Threats
Time-Varying A*: search on Graphs with time-varying Environment/Threats
    - AStarNoWait - time-varying graph, but does not consider waiting Nodes
    - AStarWait - considers additional 'waiting' Node neighbors"""
import heapq
import itertools

from Graph import Graph, Vertex, XYNode


class PriorityQueue:
    """Collection of items with priorities, such that items can be
    efficiently retrieved in order of their priority, and removed. The
    items must be hashable.
    https://codereview.stackexchange.com/questions/152757/python-priority-queue-class-wrapper
    """
    _REMOVED = object()         # placeholder for a removed entry

    def __init__(self, iterable=()):
        """Construct a priority queue from the iterable, whose elements are
        pairs (item, priority) where the items are hashable and the
        priorities are orderable.

        """
        self._entry_finder = {}  # mapping of items to entries

        # Iterable generating unique sequence numbers that are used to
        # break ties in case the items are not orderable.
        self._counter = itertools.count()

        self._data = []
        for item, priority in iterable:
            self.add(item, priority)

    def add(self, item, priority):
        """Add item to the queue with the given priority. If item is already
        present in the queue then its priority is updated.

        """
        if item in self._entry_finder:
            self.remove(item)
        entry = [priority, next(self._counter), item]
        self._entry_finder[item] = entry
        heapq.heappush(self._data, entry)

    def remove(self, item):
        """Remove item from the queue. Raise KeyError if not found."""
        entry = self._entry_finder.pop(item)
        entry[-1] = self._REMOVED

    def pop(self):
        """Remove the item with the lowest priority from the queue and return
        it. Raise KeyError if the queue is empty.

        """
        while self._data:
            _, _, item = heapq.heappop(self._data)
            if item is not self._REMOVED:
                del self._entry_finder[item]
                return item
        raise KeyError('pop from an empty priority queue')

    def is_empty(self):
        """Check if priority queue is empty"""
        return not self._entry_finder


def AstarPath(env, start, goal):
    start_node = XYNode(0, *start)
    goal_node = XYNode(env.n_grid - 1, *goal)
    graph = Graph(env=env)
    graph.add_vertex(start_node)
    start_vertex = graph.get_vertex(start_node)

    goal_vertex = Vertex(node=goal_node)
    goal_vertex_found = Astar(graph, start_vertex, goal_vertex)
    path = [goal_vertex_found.node]
    reconstruct_path(goal_vertex_found, path)
    return path


def Astar(graph, start_vertex, goal_vertex):
    """
    A* search on a Graph from a start Vertex to goal Vertex
    Usage:

    graph = Graph(env=env)
    graph.add_vertex(start_node)
    start_vertex = graph.get_vertex(start_node)

    goal_vertex = Vertex(node=goal_node)
    goal_vertex_found = Astar(graph, start_vertex, goal_vertex)
    path = [goal_vertex_found.node]
    reconstruct_path(goal_vertex_found, path)
    """
    found_path = False
    # Put start Vertex into priority queue
    open_list = PriorityQueue()

    open_list.add(start_vertex, 0)
    start_vertex.is_in_openlist = True
    start_vertex.g_cost = 0

    while not open_list.is_empty() and not found_path:
        v_current = open_list.pop()
        if v_current.is_visited:
            continue

        if v_current.node == goal_vertex.node:
            #print("GOAL FOUND!!!")
            found_path = True
            return v_current

        v_current.is_in_openlist = False
        v_current.is_visited = True

        # Expand current vertex/node
        neighbor_list = graph.env.get_neighbors(v_current.node)
        for nbr in neighbor_list:
            graph.add_edge(v_current.node, nbr, 1.0)

        # Check all neighbors of current vertex
        for neighbor in v_current.neighbors:
            if not neighbor.is_visited:
                nbr_cost = graph.env.threat_field.threat_value(neighbor.node.pos_x, neighbor.node.pos_y)
                new_cost = v_current.g_cost + nbr_cost

                if not neighbor.is_in_openlist or new_cost < neighbor.g_cost:
                    neighbor.parent = v_current
                    neighbor.g_cost = new_cost
                    neighbor.h_cost = neighbor.node.get_heuristic(goal_node=goal_vertex.node)
                    neighbor.f_cost = neighbor.g_cost + neighbor.h_cost

                    open_list.add(neighbor, neighbor.f_cost)
                    neighbor.is_in_openlist = True
    print("GOAL NOT FOUND???")
    return None  # If goal not found and open_list becomes empty return None for goal_vertex_found


def reconstruct_path(vertex, path):
    """Make shortest path from vertex.parent

    Used after calling a search algorithm, typical usage:
    goal_vertex_found = SOME_SEARCH_ALG(input params)
    path = [goal_vertex_found.node]
    reconstruct_path(goal_vertex_found, path)

    path now contains list of nodes from start to goal"""
    if vertex.parent:
        path.append(vertex.parent.node)
        reconstruct_path(vertex=vertex.parent, path=path)
    return
