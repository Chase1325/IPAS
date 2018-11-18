"""Graph and Vertex (Node) classes

This file contains classes for creating a searchable Graph based on
an Environment object. Environment object specifies vertex locations
and spacing.

Each Vertex has attributes used in Astar search including:
f: (g + h) expected cost from start to goal
g: cost from start vertex to current vertex
h: the heuristic cost of going from current vertex to goal
is_in_openlist: marker if in the openlist/frontier
is_visited: marker if vertex has been explored (closed/expanded vertex list)"""

import sys
from functools import total_ordering


class Node(object):
    """Base class for Nodes. Nodes become a parameter for Vertex's. Nodes describe
    the physical state of a system such as location and threat value. Each Node has
    a unique node_id and corresponds to a grid point in the Environment class.

    Typically there should be no need to directly generate Node objects, except perhaps
    for the start and goal node locations. """
    def __init__(self, node_id):
        self.node_id = node_id
        self.is_goal = False

    def get_heuristic(self, goal_node):
        return NotImplementedError

    def __eq__(self, other):
        return self.node_id == other.node_id


class XYNode(Node):
    """XYNode is the 2D Node which should be used with XYEnvironment

    Most use cases have no need to generate XYNode directly.
    See 'test_search.py' for example of generating all XYNodes in a given XYEnvironment."""
    def __init__(self, node_id, pos_x=0, pos_y=0, threat_value=0):
        super(XYNode, self).__init__(node_id=node_id)

        self.pos_x = pos_x
        self.pos_y = pos_y
        self.threat_value = threat_value

    def get_heuristic(self, goal_node):
        return 0

    def __str__(self):
        selfstring = "id = {0}, pos_x = {1:.2f}, pos_y = {2:.2f}, threat = {3:.2f}".format(
               self.node_id, self.pos_x, self.pos_y, self.threat_value)
        return "XYNode: " + selfstring


class XYTNode(XYNode):
    """Time dependent version of XYNode"""
    def __init__(self, node_id, pos_x=0, pos_y=0, threat_value=0, time=0, time_idx=0):
        super(XYTNode, self).__init__(node_id=node_id, pos_x=pos_x, pos_y=pos_y, threat_value=threat_value)

        self.time = time
        self.time_idx = time_idx

    def get_heuristic(self, goal_node):
        return 0


@total_ordering
class Vertex(object):
    """Vertex objects are used in the Search algorithms. They are 'interface' between
    the physical Nodes and the Graph class. Vertex's store all the typical search algorithm
    parameters such as: parent/previous/backpointer, neighbors, (g,h,f) costs, and flags
    marking if the Vertex is in the open list/frontier/fringe and a flag for explored/
    visited/closed list.

    Takes a Node object to initialize:
    start_node = XYNode(node_id, pos_x, pos_y)
    start_vertex = Vertex(start_node)

    Although most Vertex's are generated internally by the Graph class."""
    def __init__(self, node):
        self.vert_id = node.node_id
        self.node = node
        self.parent = None
        self.neighbors = {}
        self.is_in_openlist = False
        self.is_visited = False
        self.f_cost = sys.maxsize
        self.g_cost = sys.maxsize
        self.h_cost = 0

    def add_neighbor(self, neighbor, edge_cost=0):
        self.neighbors[neighbor] = edge_cost

    def get_neighbors_ids(self):
        return self.neighbors.keys()

    def get_node_id(self):
        return self.node.node_id

    def get_neighbor_cost(self, neighbor):  # edge weight, edge cost
        return self.neighbors[neighbor]

    def get_g_cost(self):
        return self.g_cost

    def set_g_cost(self, g_cost):
        self.g_cost = g_cost

    def get_f_cost(self):
        return self.f_cost

    def set_f_cost(self, f_cost):
        self.f_cost = f_cost

    def set_parent(self, parent):
        self.parent = parent

    def set_visited(self):
        self.is_visited = True

    def set_is_in_open(self, value=True):
        self.is_in_openlist = value

    def __str__(self):
        return str(self.node.node_id) + ' neighbors: ' + str([x.node.node_id for x in self.neighbors])

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.f_cost == other.f_cost
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, self.__class__):
            return self.f_cost < other.f_cost

    def __hash__(self):
        return id(self)


class Graph(object):
    """Used by the Search algorithm class functions. Typical use cases only call
    for initializing a graph and adding a start vertex. The rest of the vertex and edge
    creation is handled by and during the Search algorithm.

    graph = Graph(env=env)
    graph.add_vertex(start_node)
    start_vertex = graph.get_vertex(start_node)

    goal_vertex = Vertex(node=goal_node)
    goal_vertex_found = Astart(graph=graph, start_vertex=start_vertex, goal_vertex=goal_vertex)
    """
    def __init__(self, env):
        self.vert_dict = {}
        self.num_vertices = 0
        self.env = env

    def __iter__(self):
        return iter(self.vert_dict.values())

    def add_vertex(self, node):
        """Vertices are added/created in the Graph by supplying a Node object
        start_node = XYNode(node_id, x_loc, y_loc)
        graph.add_vertex(start_node)"""
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node=node)
        self.vert_dict[node.node_id] = new_vertex
        return new_vertex

    def get_vertex(self, node):
        """Get a Vertex from the Graph corresponding to a Node object

        start_vertex = graph.get_vertex(start_node)"""
        if isinstance(node, Node):  # pass in a Node class object or node.node_id itself
            node_id = node.node_id  # I will check for both cases
        else:
            node_id = node
        if node_id in self.vert_dict:
            return self.vert_dict[node_id]
        else:
            return None

    def add_edge(self, src, dst, cost=0):
        """Used internally by Search algorithms."""
        if src.node_id not in self.vert_dict:
            self.add_vertex(src)
        if dst.node_id not in self.vert_dict:
            self.add_vertex(dst)

        self.vert_dict[src.node_id].add_neighbor(self.vert_dict[dst.node_id], cost)
        self.vert_dict[dst.node_id].add_neighbor(self.vert_dict[src.node_id], cost)

    def get_vertices(self):
        return self.vert_dict.keys()
