"""Environments

Develop the associated World/Map/Environment for searching on. Should define the
spatial relationships: size, dimension, number of grid points, definition of
neighbors.

See Threat.py; A Threat can be associated with the Environment and encodes some
'cost' which is used by the searching functions.
"""
try:
    from .Graph import Node, XYNode, XYTNode
except ImportError:
    from Graph import Node, XYNode, XYTNode


class Environment(object):
    """Base class for Environments

    Environments are the reference point for threats and graph generation.
    Subclasses that extend Environment should define their form in more explicit
    detail (i.e. it's number of dimensions, size, resolution, relationships
    adjacent points.)
    It should define a way to add a threat field and define a way to get the neighbors
    of a given location

    Threat fields are accessible through an Environment instance (env.threat_field)
    and can be added at the creation of an environment:
    env = Environment(dim=2, threat_field = my_field)
    OR after Environment established
    env.add_threat_field(my_field)"""

    def __init__(self, dim, threat_field=None):
        self.dim = dim
        self.threat_field = threat_field

    def get_neighbors(self, node):
        return NotImplementedError

    def add_threat_field(self, threat_field):
        self.threat_field = threat_field


class XYEnvironment(Environment):
    """This class is for 2D environments where locations are given by (x, y) points

    Generate a XY (2D) Environment example:
    env = XYEnvironment(x_size=10, y_size=10, n_grid_x=20, n_grid_y=30)
    to generate a 10 by 10 environment with resolution of 20 grid points in x direction
    and 30 grid points in y direction

    get_neighbors function uses 4-way connectivity"""

    def __init__(self, x_size, y_size, n_grid_x, n_grid_y, threat_field=None):
        super().__init__(dim=2, threat_field=threat_field)

        self.x_size = x_size
        self.y_size = y_size
        self.n_grid_x = n_grid_x
        self.n_grid_y = n_grid_y

        self.grid_sep_x = self.x_size / (self.n_grid_x - 1)
        self.grid_sep_y = self.y_size / (self.n_grid_y - 1)
        self.n_grid = self.n_grid_x * self.n_grid_y

    def get_neighbors(self, node):
        neighbors = []  # neighbors is a list of Nodes
        curr_id = node.node_id
        # Add neighbor to the RIGHT
        if (curr_id + 1 < self.n_grid) and ((curr_id + 1) % self.n_grid_x != 0):
            new_id = curr_id + 1
            new_node = XYNode(new_id)
            mx = new_id % self.n_grid_x
            my = int(new_id / self.n_grid_x)
            new_node.pos_x = mx * self.grid_sep_x
            new_node.pos_y = my * self.grid_sep_y
            neighbors.append(new_node)
        # Add neighbor to the LEFT
        if (curr_id - 1 >= 0) and (curr_id % self.n_grid_x != 0):
            new_id = curr_id - 1
            new_node = XYNode(new_id)
            mx = new_id % self.n_grid_x
            my = int(new_id / self.n_grid_x)
            new_node.pos_x = mx * self.grid_sep_x
            new_node.pos_y = my * self.grid_sep_y
            neighbors.append(new_node)
        # Add neighbor ABOVE
        if curr_id + self.n_grid_x < self.n_grid:
            new_id = curr_id + self.n_grid_x
            new_node = XYNode(new_id)
            mx = new_id % self.n_grid_x
            my = int(new_id / self.n_grid_x)
            new_node.pos_x = mx * self.grid_sep_x
            new_node.pos_y = my * self.grid_sep_y
            neighbors.append(new_node)
        # Add neighbor BELOW
        if curr_id - self.n_grid_x >= 0:
            new_id = curr_id - self.n_grid_x
            new_node = XYNode(new_id)
            mx = new_id % self.n_grid_x
            my = int(new_id / self.n_grid_x)
            new_node.pos_x = mx * self.grid_sep_x
            new_node.pos_y = my * self.grid_sep_y
            neighbors.append(new_node)
        return neighbors

    def get_location_from_gridpt(self, gridpt):
        """Get an x, y location from a grid point id number

        x, y = env.get_location_from_gridpt(5)"""
        mx = gridpt % self.n_grid_x
        my = int(gridpt / self.n_grid_x)
        pos_x = mx * self.grid_sep_x
        pos_y = my * self.grid_sep_y
        return pos_x, pos_y

    def get_gridpt_from_location(self, pos_x, pos_y):
        """Get the associated grid point id from an x, y location

        grid_pt = env.get_gridpt_from_location(x_loc, y_loc)

        TODO: Check bounds of env for valid x, y locations"""
        mx = int(pos_x / self.grid_sep_x)
        my = int(pos_y / self.grid_sep_y)
        gridpt = my * self.n_grid_x + mx
        return gridpt

    def __str__(self):
        selfstring = "x_size = {0}, y_size = {1}, n_grid_x = {2}, n_grid_y = {3}".format(
            self.x_size, self.y_size, self.n_grid_x, self.n_grid_y)
        return "XYEnv: " + selfstring


class SquareXYEnvironment(XYEnvironment):
    """Simple way to define a Square XYEnvironment

    Define a strictly square environment by calling XYEnvironment constructor with
    x_size = y_size and n_grid_x = n_grid_y

    square_env = SquareXYEnvironment(wksp_size = 10, grid_pts = 20)"""

    def __init__(self, wksp_size, grid_pts, threat_field=None):
        super(SquareXYEnvironment, self).__init__(x_size=wksp_size, y_size=wksp_size, n_grid_x=grid_pts, n_grid_y=grid_pts, threat_field=threat_field)


class XYTEnvironment(XYEnvironment):
    """This defines a Time-varying 2D environment, therefore locations are (x, y, t) points"""

    def __init__(self, x_size, y_size, n_grid_x, n_grid_y, t_final):
        super(XYTEnvironment, self).__init__(x_size=x_size, y_size=y_size, n_grid_x=n_grid_x, n_grid_y=n_grid_y)

        self.t_final = t_final
